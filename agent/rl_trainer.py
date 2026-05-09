import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from abr_env import ABREnv
from checkpoint_utils import load_netllm_checkpoint, save_netllm_checkpoint
from network import NetLLMABR
from return_utils import load_return_to_go_processor
from settings import (
    ACTION_DIM,
    BASE_MODEL_PATH,
    RL_MODEL_PATH,
    SFT_MODEL_PATH,
    TRAINING_STATS_PATH,
)
from training_eval import build_eval_trace_files, evaluate_model_policy


def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def mask_logits_for_jump_limit(logits, last_action, jump_limit):
    if jump_limit is None or int(jump_limit) < 0:
        return logits

    jump_limit = int(jump_limit)
    lower = max(0, int(last_action) - jump_limit)
    upper = min(logits.numel() - 1, int(last_action) + jump_limit)
    mask = torch.full_like(logits, -1e9)
    mask[lower:upper + 1] = 0.0
    return logits + mask


def categorical_kl(policy_logits, reference_logits):
    log_probs = F.log_softmax(policy_logits, dim=-1)
    reference_probs = F.softmax(reference_logits, dim=-1)
    return F.kl_div(log_probs, reference_probs, reduction="sum")


def _set_module_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad_(bool(trainable))


def configure_rl_trainable_scope(model, train_scope):
    """Freeze most of the SFT model by default for safer RL fine-tuning."""
    scope = str(train_scope or "action_head").strip().lower().replace("-", "_")
    for param in model.parameters():
        param.requires_grad_(False)

    if scope in {"all", "full"}:
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        _set_module_trainable(model.action_head, True)

        projection_scopes = {
            "projection",
            "head_projection",
            "projection_lora",
            "head_projection_lora",
            "non_plm",
            "non_plm_lora",
        }
        if scope in projection_scopes:
            for module_name in (
                "embed_state1",
                "embed_state2",
                "embed_state3",
                "embed_state4",
                "embed_state5",
                "embed_state6",
                "embed_return",
                "embed_action",
                "embed_timestep",
                "embed_ln",
            ):
                _set_module_trainable(getattr(model, module_name), True)

        if scope in {"non_plm", "non_plm_lora"}:
            _set_module_trainable(model.state_encoder, True)

        if "lora" in scope:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    if trainable == 0:
        raise RuntimeError(f"No trainable parameters for RL train_scope={train_scope}")
    return scope, trainable, total


def selection_score(
    summary,
    mean_weight=1.0,
    p10_weight=0.10,
    stall_weight=0.25,
    switch_weight=0.05,
):
    return (
        float(mean_weight) * float(summary["mean_qoe"])
        + float(p10_weight) * float(summary["p10_qoe"])
        - float(stall_weight) * float(summary["mean_stall_time"])
        - float(switch_weight) * float(summary["quality_switch_count"])
    )


def metric_deltas(summary, baseline):
    if baseline is None:
        return {}
    return {
        "mean_qoe": float(summary["mean_qoe"] - baseline["mean_qoe"]),
        "p10_qoe": float(summary["p10_qoe"] - baseline["p10_qoe"]),
        "mean_stall_time": float(summary["mean_stall_time"] - baseline["mean_stall_time"]),
        "quality_switch_count": float(
            summary["quality_switch_count"] - baseline["quality_switch_count"]
        ),
    }


def validation_is_safe(
    summary,
    baseline,
    mean_tolerance,
    p10_tolerance,
    stall_tolerance,
    switch_tolerance,
):
    if baseline is None:
        return True
    return (
        summary["mean_qoe"] >= baseline["mean_qoe"] - float(mean_tolerance)
        and summary["p10_qoe"] >= baseline["p10_qoe"] - float(p10_tolerance)
        and summary["mean_stall_time"] <= baseline["mean_stall_time"] + float(stall_tolerance)
        and summary["quality_switch_count"] <= baseline["quality_switch_count"] + float(switch_tolerance)
    )


def train_rl(
    sft_model_path=SFT_MODEL_PATH,
    rl_save_path=RL_MODEL_PATH,
    episodes=200,
    lr=1e-5,
    gamma=0.99,
    target_return=None,
    qoe_profile="pensieve",
    trace_split="train",
    trace_dir=None,
    seed=100003,
    eval_interval=10,
    validation_episodes=10,
    validation_split=None,
    validation_trace_dir=None,
    validation_sample_mode="stratified",
    validation_seed=20260507,
    validation_jump_limit=1,
    validation_batch_size=8,
    rollout_batch_size=1,
    validation_min_delta=1e-3,
    early_stop_patience=3,
    min_episodes=20,
    action_jump_limit=1,
    entropy_coef=0.01,
    kl_coef=0.0,
    advantage_momentum=0.95,
    advantage_clip=5.0,
    grad_clip=0.25,
    train_scope="action_head",
    exclude_validation_traces=True,
    validation_mean_tolerance=5.0,
    validation_p10_tolerance=2.0,
    validation_stall_tolerance=0.25,
    validation_switch_tolerance=1.0,
    selection_mean_weight=1.0,
    selection_p10_weight=0.10,
    selection_stall_weight=0.25,
    selection_switch_weight=0.05,
    save_candidates=True,
    train_history_action_embedding=True,
    stats_path=TRAINING_STATS_PATH,
    history_save_path=None,
    progress_interval=10,
    base_model_path=BASE_MODEL_PATH,
):
    """REINFORCE fine-tuning with evaluation gates and safety regularizers."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats_path = Path(stats_path or TRAINING_STATS_PATH)
    rtg = load_return_to_go_processor(
        target_return=target_return,
        stats_path=stats_path,
    )
    target_return = rtg.target_return
    rl_save_path = Path(rl_save_path)
    rl_save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = rl_save_path.parent
    baseline_save_path = checkpoint_dir / f"{rl_save_path.name}_baseline"
    best_safe_save_path = checkpoint_dir / f"{rl_save_path.name}_best_safe"
    latest_save_path = checkpoint_dir / f"{rl_save_path.name}_latest"

    validation_split = validation_split or trace_split
    validation_trace_dir = validation_trace_dir or trace_dir
    validation_trace_files = build_eval_trace_files(
        trace_split=validation_split,
        trace_dir=validation_trace_dir,
        episodes=int(validation_episodes),
        sample_mode=validation_sample_mode,
        seed=int(validation_seed),
        qoe_profile=qoe_profile,
    )
    same_validation_pool = (
        str(validation_split).strip().lower() == str(trace_split).strip().lower()
        and (validation_trace_dir or None) == (trace_dir or None)
    )
    train_exclude_trace_files = (
        validation_trace_files
        if bool(exclude_validation_traces) and same_validation_pool
        else []
    )
    def make_train_env():
        return ABREnv(
            qoe_profile=qoe_profile,
            trace_split=trace_split,
            trace_dir=trace_dir,
            exclude_trace_files=train_exclude_trace_files,
        )

    env = make_train_env()

    agent = NetLLMABR(
        model_name_or_path=str(base_model_path),
        action_dim=ACTION_DIM,
        lora_rank=128,
    )

    sft_model_path = Path(sft_model_path)
    loaded_sft_path = load_netllm_checkpoint(
        agent,
        sft_model_path,
        map_location=device,
        strict=False,
    )
    if loaded_sft_path is not None:
        print(f"[RL] loaded SFT weights: {loaded_sft_path}")

    agent.to(device)
    agent.train()
    resolved_scope, trainable_params, total_params = configure_rl_trainable_scope(
        agent,
        train_scope,
    )
    print(
        f"[RL] train_scope={resolved_scope} "
        f"trainable_params={trainable_params:,}/{total_params:,}"
    )

    reference_agent = None
    if float(kl_coef) > 0.0:
        reference_agent = NetLLMABR(
            model_name_or_path=str(base_model_path),
            action_dim=ACTION_DIM,
            lora_rank=128,
        )
        loaded_reference_path = load_netllm_checkpoint(
            reference_agent,
            sft_model_path,
            map_location=device,
            strict=False,
        )
        reference_agent.to(device)
        reference_agent.eval()
        for param in reference_agent.parameters():
            param.requires_grad_(False)
        print(f"[RL] KL anchor enabled; reference={loaded_reference_path}")

    optimizer = optim.AdamW(
        filter(lambda param: param.requires_grad, agent.parameters()),
        lr=lr,
    )

    print(
        f"[RL] device={device}, episodes={episodes}, "
        f"qoe_profile={env.qoe_profile}, {rtg.describe()}, "
        f"eval_traces={len(validation_trace_files)}, "
        f"excluded_train_traces={len(train_exclude_trace_files)}, "
        f"rollout_batch_size={int(rollout_batch_size or 1)}, "
        f"jump_limit={action_jump_limit}, entropy_coef={entropy_coef}, "
        f"kl_coef={kl_coef}, "
        f"history_action_grad={bool(train_history_action_embedding)}"
    )

    baseline_qoe = None
    baseline_summary = None
    best_selection_score = -float("inf")
    best_eval_qoe = -float("inf")
    evals_without_improvement = 0
    evals_below_baseline = 0
    validation_history = []

    def flush_validation_history():
        if not validation_trace_files:
            return None
        history_path = (
            Path(history_save_path)
            if history_save_path
            else rl_save_path.parent / "rl_validation_history.json"
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(validation_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return history_path

    if validation_trace_files:
        baseline_summary, _ = evaluate_model_policy(
            model=agent,
            trace_files=validation_trace_files,
            trace_split=validation_split,
            trace_dir=validation_trace_dir,
            qoe_profile=qoe_profile,
            target_return=target_return,
            device=device,
            jump_limit=validation_jump_limit,
            label="rl-sft-baseline",
            stats_path=stats_path,
            batch_size=validation_batch_size,
        )
        baseline_qoe = baseline_summary["mean_qoe"]
        baseline_score = selection_score(
            baseline_summary,
            mean_weight=selection_mean_weight,
            p10_weight=selection_p10_weight,
            stall_weight=selection_stall_weight,
            switch_weight=selection_switch_weight,
        )
        best_selection_score = baseline_score
        best_eval_qoe = baseline_qoe
        baseline_path = save_netllm_checkpoint(
            agent,
            baseline_save_path,
            metadata={"stage": "rl", "episode": 0, "safe_initial": True},
        )
        saved_path = save_netllm_checkpoint(
            agent,
            rl_save_path,
            metadata={"stage": "rl", "episode": 0, "safe_initial": True, "role": "best_safe"},
        )
        best_safe_path = save_netllm_checkpoint(
            agent,
            best_safe_save_path,
            metadata={"stage": "rl", "episode": 0, "safe_initial": True, "role": "best_safe"},
        )
        validation_history.append({
            "episode": 0,
            "safe": True,
            "improved": False,
            "selection_score": baseline_score,
            "best_selection_score": best_selection_score,
            "baseline_delta": metric_deltas(baseline_summary, baseline_summary),
            "save_paths": {
                "baseline": str(baseline_path),
                "best_safe": str(best_safe_path),
                "official": str(saved_path),
            },
            **baseline_summary,
        })
        flush_validation_history()
        print(
            f"[RL] baseline mean_qoe={baseline_qoe:.3f} "
            f"p10_qoe={baseline_summary['p10_qoe']:.3f} "
            f"score={baseline_score:.3f}; "
            f"saved baseline -> {baseline_path}; official -> {saved_path}"
        )

    running_return_mean = 0.0
    running_return_var = 1.0
    running_return_ready = False
    train_start = time.time()
    progress_interval = int(progress_interval or 0)

    rollout_batch_size = max(1, int(rollout_batch_size or 1))
    train_envs = [env] + [make_train_env() for _ in range(rollout_batch_size - 1)]
    episode = 0

    while episode < int(episodes):
        previous_completed = episode
        batch_size = min(rollout_batch_size, int(episodes) - episode)
        batch_envs = train_envs[:batch_size]

        histories = [agent.new_history() for _ in range(batch_size)]
        reference_histories = (
            [reference_agent.new_history() for _ in range(batch_size)]
            if reference_agent is not None
            else [None] * batch_size
        )
        states = []
        for batch_env in batch_envs:
            state, _ = batch_env.reset()
            states.append(state)

        dones = [False] * batch_size
        steps = [0] * batch_size
        last_actions = [0] * batch_size
        current_target_returns = [target_return] * batch_size
        rewards_by_episode = [[] for _ in range(batch_size)]
        log_probs_by_episode = [[] for _ in range(batch_size)]
        entropies_by_episode = [[] for _ in range(batch_size)]
        kl_terms_by_episode = [[] for _ in range(batch_size)]

        while not all(dones):
            active_indices = [idx for idx, done in enumerate(dones) if not done]
            state_batch = np.stack([states[idx] for idx in active_indices], axis=0)
            state_tensor = torch.from_numpy(state_batch).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ).unsqueeze(1)
            target_return_batch = [current_target_returns[idx] for idx in active_indices]
            timestep_batch = [steps[idx] for idx in active_indices]
            history_batch = [histories[idx] for idx in active_indices]

            with autocast_context(device):
                logits_batch, return_emb, state_emb, time_emb = agent.predict_logits_batch(
                    states=state_tensor,
                    target_returns=target_return_batch,
                    timesteps=timestep_batch,
                    histories=history_batch,
                )
                logits_batch = torch.stack([
                    mask_logits_for_jump_limit(
                        logits_batch[local_idx],
                        last_actions[episode_idx],
                        action_jump_limit,
                    )
                    for local_idx, episode_idx in enumerate(active_indices)
                ])

                if reference_agent is not None:
                    reference_history_batch = [
                        reference_histories[idx] for idx in active_indices
                    ]
                    with torch.no_grad():
                        (
                            ref_logits_batch,
                            ref_return_emb,
                            ref_state_emb,
                            ref_time_emb,
                        ) = reference_agent.predict_logits_batch(
                            states=state_tensor,
                            target_returns=target_return_batch,
                            timesteps=timestep_batch,
                            histories=reference_history_batch,
                        )
                        ref_logits_batch = torch.stack([
                            mask_logits_for_jump_limit(
                                ref_logits_batch[local_idx],
                                last_actions[episode_idx],
                                action_jump_limit,
                            )
                            for local_idx, episode_idx in enumerate(active_indices)
                        ])
                else:
                    ref_return_emb = ref_state_emb = ref_time_emb = None
                    ref_logits_batch = None

            dist = Categorical(logits=logits_batch)
            actions = dist.sample()
            log_prob_batch = dist.log_prob(actions)
            entropy_batch = dist.entropy()

            for local_idx, episode_idx in enumerate(active_indices):
                action_item = int(actions[local_idx].item())
                log_probs_by_episode[episode_idx].append(log_prob_batch[local_idx])
                entropies_by_episode[episode_idx].append(entropy_batch[local_idx])
                if ref_logits_batch is not None:
                    kl_terms_by_episode[episode_idx].append(
                        categorical_kl(
                            logits_batch[local_idx],
                            ref_logits_batch[local_idx],
                        )
                    )

                next_state, reward, done, _, _ = batch_envs[episode_idx].step(action_item)
                rewards_by_episode[episode_idx].append(float(reward))
                current_target_returns[episode_idx] = rtg.update(
                    current_target_returns[episode_idx],
                    reward,
                )

                agent.append_history(
                    history=histories[episode_idx],
                    return_emb=return_emb[local_idx:local_idx + 1],
                    state_emb=state_emb[local_idx:local_idx + 1],
                    time_emb=time_emb[local_idx:local_idx + 1],
                    action=action_item,
                    device=device,
                    detach_action=not bool(train_history_action_embedding),
                )
                if reference_agent is not None:
                    reference_agent.append_history(
                        history=reference_histories[episode_idx],
                        return_emb=ref_return_emb[local_idx:local_idx + 1],
                        state_emb=ref_state_emb[local_idx:local_idx + 1],
                        time_emb=ref_time_emb[local_idx:local_idx + 1],
                        action=action_item,
                        device=device,
                    )

                states[episode_idx] = next_state
                dones[episode_idx] = bool(done)
                last_actions[episode_idx] = action_item
                steps[episode_idx] += 1

        return_tensors = []
        reward_sums = []
        for rewards in rewards_by_episode:
            returns = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + gamma * running
                returns.insert(0, running)
            return_tensors.append(torch.tensor(returns, dtype=torch.float32, device=device))
            reward_sums.append(float(np.sum(rewards)))

        all_returns = torch.cat(return_tensors)
        episode_mean = float(all_returns.mean().detach().cpu())
        episode_var = float(all_returns.var(unbiased=False).detach().cpu())
        if not running_return_ready:
            running_return_mean = episode_mean
            running_return_var = max(episode_var, 1e-6)
            running_return_ready = True
        else:
            momentum = float(advantage_momentum)
            running_return_mean = (
                momentum * running_return_mean + (1.0 - momentum) * episode_mean
            )
            running_return_var = (
                momentum * running_return_var + (1.0 - momentum) * max(episode_var, 1e-6)
            )

        policy_losses = []
        for returns, log_probs in zip(return_tensors, log_probs_by_episode):
            advantages = (returns - running_return_mean) / np.sqrt(running_return_var + 1e-6)
            if advantage_clip and advantage_clip > 0:
                advantages = torch.clamp(
                    advantages,
                    -float(advantage_clip),
                    float(advantage_clip),
                )
            log_prob_tensor = torch.stack(log_probs)
            policy_losses.append(-(log_prob_tensor * advantages.detach()).mean())

        policy_loss = torch.stack(policy_losses).mean()
        entropy_tensor = torch.cat([
            torch.stack(entropies)
            for entropies in entropies_by_episode
            if entropies
        ])
        entropy_loss = -float(entropy_coef) * entropy_tensor.mean()
        flat_kl_terms = [
            torch.stack(kl_terms)
            for kl_terms in kl_terms_by_episode
            if kl_terms
        ]
        kl_mean = torch.cat(flat_kl_terms).mean() if flat_kl_terms else None
        kl_loss = (
            float(kl_coef) * kl_mean
            if kl_mean is not None
            else torch.tensor(0.0, dtype=torch.float32, device=device)
        )
        loss = policy_loss + entropy_loss + kl_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
        optimizer.step()

        episode += batch_size
        completed = episode

        if (
            progress_interval > 0
            and (
                completed == int(episodes)
                or completed // progress_interval > previous_completed // progress_interval
            )
        ):
            elapsed = time.time() - train_start
            eta = elapsed / max(completed, 1) * max(int(episodes) - completed, 0)
            print(
                f"[RL] episode={completed}/{episodes} "
                f"batch={batch_size} "
                f"reward_mean={np.mean(reward_sums):.2f} "
                f"loss={loss.item():.4f} "
                f"pg={policy_loss.item():.4f} "
                f"entropy={entropy_tensor.mean().item():.4f} "
                f"kl={(kl_mean.item() if kl_mean is not None else 0.0):.4f} "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(eta)}"
            )

        should_eval = (
            validation_trace_files
            and int(eval_interval) > 0
            and (
                completed == int(episodes)
                or completed // int(eval_interval) > previous_completed // int(eval_interval)
            )
        )
        if should_eval:
            summary, _ = evaluate_model_policy(
                model=agent,
                trace_files=validation_trace_files,
                trace_split=validation_split,
                trace_dir=validation_trace_dir,
                qoe_profile=qoe_profile,
                target_return=target_return,
                device=device,
                jump_limit=validation_jump_limit,
                label="rl-validation",
                stats_path=stats_path,
                batch_size=validation_batch_size,
            )
            candidate_path = None
            latest_path = None
            if save_candidates:
                candidate_path = save_netllm_checkpoint(
                    agent,
                    checkpoint_dir / f"{rl_save_path.name}_candidate_ep{completed:04d}",
                    metadata={"stage": "rl", "episode": completed, "role": "candidate"},
                )
                latest_path = save_netllm_checkpoint(
                    agent,
                    latest_save_path,
                    metadata={"stage": "rl", "episode": completed, "role": "latest"},
                )
            eval_qoe = summary["mean_qoe"]
            current_score = selection_score(
                summary,
                mean_weight=selection_mean_weight,
                p10_weight=selection_p10_weight,
                stall_weight=selection_stall_weight,
                switch_weight=selection_switch_weight,
            )
            safe_to_save = validation_is_safe(
                summary,
                baseline_summary,
                validation_mean_tolerance,
                validation_p10_tolerance,
                validation_stall_tolerance,
                validation_switch_tolerance,
            )
            improved = (
                current_score > best_selection_score + float(validation_min_delta)
                and safe_to_save
            )
            below_baseline = baseline_summary is not None and not safe_to_save
            history_item = {
                "episode": completed,
                "safe": bool(safe_to_save),
                "improved": bool(improved),
                "selection_score": float(current_score),
                "best_selection_score": float(best_selection_score),
                "baseline_delta": metric_deltas(summary, baseline_summary),
                "save_paths": {
                    "candidate": str(candidate_path) if candidate_path else None,
                    "latest": str(latest_path) if latest_path else None,
                },
                **summary,
            }

            if improved:
                best_selection_score = current_score
                best_eval_qoe = eval_qoe
                evals_without_improvement = 0
                evals_below_baseline = 0
                saved_path = save_netllm_checkpoint(
                    agent,
                    rl_save_path,
                    metadata={
                        "stage": "rl",
                        "episode": completed,
                        "role": "official_best_safe",
                        "selection_score": current_score,
                    },
                )
                best_safe_path = save_netllm_checkpoint(
                    agent,
                    best_safe_save_path,
                    metadata={
                        "stage": "rl",
                        "episode": completed,
                        "role": "best_safe",
                        "selection_score": current_score,
                    },
                )
                history_item["save_paths"]["official"] = str(saved_path)
                history_item["save_paths"]["best_safe"] = str(best_safe_path)
                history_item["best_selection_score"] = float(best_selection_score)
                print(
                    f"[RL] saved new best model at episode={completed}: "
                    f"mean_qoe={eval_qoe:.3f} p10_qoe={summary['p10_qoe']:.3f} "
                    f"stall={summary['mean_stall_time']:.3f} "
                    f"switch={summary['quality_switch_count']:.2f} "
                    f"score={current_score:.3f} -> {saved_path}"
                )
            else:
                evals_without_improvement += 1
                evals_below_baseline = evals_below_baseline + 1 if below_baseline else 0
                print(
                    f"[RL] validation episode={completed}: "
                    f"mean_qoe={eval_qoe:.3f} "
                    f"p10_qoe={summary['p10_qoe']:.3f} "
                    f"stall={summary['mean_stall_time']:.3f} "
                    f"switch={summary['quality_switch_count']:.2f} "
                    f"score={current_score:.3f} "
                    f"safe={safe_to_save} "
                    f"best_score={best_selection_score:.3f} "
                    f"baseline_mean={(baseline_qoe if baseline_qoe is not None else 0.0):.3f}"
                )
            validation_history.append(history_item)
            flush_validation_history()

            if (
                completed >= int(min_episodes)
                and int(early_stop_patience) > 0
                and (
                    evals_below_baseline >= int(early_stop_patience)
                    or evals_without_improvement >= int(early_stop_patience)
                )
            ):
                print(
                    f"[RL] early stop at episode {completed}; "
                    f"best_eval_qoe={best_eval_qoe:.3f}, "
                    f"best_score={best_selection_score:.3f}, "
                    f"below_baseline_evals={evals_below_baseline}, "
                    f"evals_without_improvement={evals_without_improvement}"
                )
                break

    if validation_trace_files:
        history_path = flush_validation_history()
        print(f"[RL] kept best validation weights: {rl_save_path}")
        print(f"[RL] saved validation history: {history_path}")
    else:
        saved_path = save_netllm_checkpoint(
            agent,
            rl_save_path,
            metadata={"stage": "rl", "episode": "final"},
        )
        print(f"[RL] saved final weights without validation gate: {saved_path}")
    return rl_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NetLLM ABR with safe REINFORCE.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-return", type=float, default=None)
    parser.add_argument("--qoe-profile", default="pensieve", choices=["pensieve", "log", "linear3"])
    parser.add_argument("--trace-split", default="train")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--seed", type=int, default=100003)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--validation-split", default=None)
    parser.add_argument("--validation-trace-dir", default=None)
    parser.add_argument(
        "--validation-sample-mode",
        default="stratified",
        choices=["first", "random", "stratified"],
    )
    parser.add_argument("--validation-seed", type=int, default=20260507)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--min-episodes", type=int, default=20)
    parser.add_argument("--action-jump-limit", type=int, default=1)
    parser.add_argument("--validation-jump-limit", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--advantage-momentum", type=float, default=0.95)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=0.25)
    parser.add_argument(
        "--train-scope",
        default="action_head",
        choices=[
            "action_head",
            "projection",
            "projection_lora",
            "non_plm",
            "non_plm_lora",
            "all",
        ],
    )
    parser.add_argument("--no-exclude-validation-traces", action="store_true")
    parser.add_argument("--validation-mean-tolerance", type=float, default=5.0)
    parser.add_argument("--validation-p10-tolerance", type=float, default=2.0)
    parser.add_argument("--validation-stall-tolerance", type=float, default=0.25)
    parser.add_argument("--validation-switch-tolerance", type=float, default=1.0)
    parser.add_argument("--selection-mean-weight", type=float, default=1.0)
    parser.add_argument("--selection-p10-weight", type=float, default=0.10)
    parser.add_argument("--selection-stall-weight", type=float, default=0.25)
    parser.add_argument("--selection-switch-weight", type=float, default=0.05)
    parser.add_argument("--no-save-candidates", action="store_true")
    parser.add_argument("--detach-history-action", action="store_true")
    parser.add_argument("--stats-path", default=TRAINING_STATS_PATH)
    parser.add_argument("--history-save-path", default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    args = parser.parse_args()
    train_rl(
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        target_return=args.target_return,
        qoe_profile=args.qoe_profile,
        trace_split=args.trace_split,
        trace_dir=args.trace_dir,
        seed=args.seed,
        eval_interval=args.eval_interval,
        validation_episodes=args.validation_episodes,
        validation_split=args.validation_split,
        validation_trace_dir=args.validation_trace_dir,
        validation_sample_mode=args.validation_sample_mode,
        validation_seed=args.validation_seed,
        validation_jump_limit=args.validation_jump_limit,
        validation_batch_size=args.validation_batch_size,
        rollout_batch_size=args.rollout_batch_size,
        early_stop_patience=args.early_stop_patience,
        min_episodes=args.min_episodes,
        action_jump_limit=args.action_jump_limit,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        advantage_momentum=args.advantage_momentum,
        advantage_clip=args.advantage_clip,
        grad_clip=args.grad_clip,
        train_scope=args.train_scope,
        exclude_validation_traces=not args.no_exclude_validation_traces,
        validation_mean_tolerance=args.validation_mean_tolerance,
        validation_p10_tolerance=args.validation_p10_tolerance,
        validation_stall_tolerance=args.validation_stall_tolerance,
        validation_switch_tolerance=args.validation_switch_tolerance,
        selection_mean_weight=args.selection_mean_weight,
        selection_p10_weight=args.selection_p10_weight,
        selection_stall_weight=args.selection_stall_weight,
        selection_switch_weight=args.selection_switch_weight,
        save_candidates=not args.no_save_candidates,
        train_history_action_embedding=not args.detach_history_action,
        stats_path=args.stats_path,
        history_save_path=args.history_save_path,
        progress_interval=args.progress_interval,
        base_model_path=args.base_model_path,
    )
