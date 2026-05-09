import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from abr_env import ABREnv
from data_collector import mpc_expert_policy
from pensieve_torch import PensieveActorCritic, save_pensieve_checkpoint
from settings import ACTION_DIM, PENSIEVE_MODEL_PATH
from training_eval import build_eval_trace_files, summarize_model_episodes


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


def discounted_returns(rewards, gamma):
    values = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + float(gamma) * running
        values.append(running)
    values.reverse()
    return torch.tensor(values, dtype=torch.float32)


def evaluate_pensieve_model(
    model,
    trace_files,
    trace_split,
    qoe_profile,
    device,
    trace_dir=None,
    progress_interval=0,
    label="pensieve-validation",
):
    was_training = model.training
    model.eval()

    episodes = []
    total_chunks = 0
    total_bitrate = 0.0
    total_stall = 0.0
    total_smooth = 0.0
    total_duration = 0.0
    rebuffer_chunks = 0
    action_counts = np.zeros(ACTION_DIM, dtype=np.int64)

    with torch.no_grad():
        for trace_file in trace_files:
            env = ABREnv(
                trace_split=trace_split,
                trace_dir=trace_dir,
                trace_file=trace_file,
                random_start=False,
                qoe_profile=qoe_profile,
            )
            state, _ = env.reset()
            done = False
            prev_action = None
            ep_qoe = 0.0
            ep_bitrate = 0.0
            ep_stall = 0.0
            ep_smooth = 0.0
            ep_switches = 0
            ep_steps = 0
            ep_action_counts = np.zeros(ACTION_DIM, dtype=np.int64)

            while not done:
                state_tensor = torch.from_numpy(state).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                ).unsqueeze(0)
                action, _, _ = model.act(state_tensor, deterministic=True)
                action = max(0, min(env.num_bitrates - 1, int(action)))
                next_state, reward, done, _, info = env.step(action)

                action_counts[action] += 1
                ep_action_counts[action] += 1
                ep_qoe += float(reward)
                ep_bitrate += float(info["bitrate_kbps"])
                ep_stall += float(info["stall_time"])
                ep_smooth += float(info["smoothness_penalty"])
                ep_steps += 1

                if info["stall_time"] > 1e-6:
                    rebuffer_chunks += 1
                if prev_action is not None and action != prev_action:
                    ep_switches += 1
                prev_action = action
                state = next_state

            total_chunks += ep_steps
            total_bitrate += ep_bitrate
            total_stall += ep_stall
            total_smooth += ep_smooth
            total_duration += ep_steps * env.segment_duration
            episodes.append({
                "trace": Path(trace_file).name,
                "qoe": ep_qoe,
                "mean_bitrate": ep_bitrate / max(ep_steps, 1),
                "stall_time": ep_stall,
                "smoothness_penalty": ep_smooth,
                "quality_switch_count": ep_switches,
                "chunks": ep_steps,
                "action_counts": {
                    str(idx): int(ep_action_counts[idx])
                    for idx in range(len(ep_action_counts))
                },
            })

            if progress_interval and len(episodes) % int(progress_interval) == 0:
                print(
                    f"[Pensieve] {label} episode={len(episodes)}/{len(trace_files)} "
                    f"mean_qoe={np.mean([item['qoe'] for item in episodes]):.3f}"
                )

    summary = summarize_model_episodes(
        episodes=episodes,
        label=label,
        qoe_profile=qoe_profile,
        total_chunks=total_chunks,
        total_bitrate=total_bitrate,
        total_stall=total_stall,
        total_smooth=total_smooth,
        total_duration=total_duration,
        rebuffer_chunks=rebuffer_chunks,
        action_counts=action_counts,
    )
    if was_training:
        model.train()
    return summary, episodes


def pretrain_with_mpc(
    model,
    optimizer,
    device,
    episodes,
    qoe_profile,
    trace_split,
    trace_dir,
    progress_interval,
):
    if int(episodes) <= 0:
        return []

    history = []
    model.train()
    start_time = time.time()
    for episode in range(1, int(episodes) + 1):
        env = ABREnv(qoe_profile=qoe_profile, trace_split=trace_split, trace_dir=trace_dir)
        state, _ = env.reset()
        done = False
        losses = []

        while not done:
            target_action = mpc_expert_policy(state, env, safety_factor=0.9, jump_limit=1)
            state_tensor = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
            logits, _ = model(state_tensor)
            target = torch.tensor([target_action], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            state, _, done, _, _ = env.step(target_action)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"episode": episode, "loss": mean_loss})
        if progress_interval and episode % int(progress_interval) == 0:
            elapsed = time.time() - start_time
            print(
                f"[Pensieve] pretrain={episode}/{episodes} "
                f"loss={mean_loss:.4f} elapsed={format_duration(elapsed)}"
            )

    return history


def train_one_episode(model, optimizer, device, env, gamma, value_coef, entropy_coef, grad_clip):
    state, _ = env.reset()
    done = False
    log_probs = []
    values = []
    rewards = []
    entropies = []
    ep_qoe = 0.0
    ep_steps = 0

    while not done:
        state_tensor = torch.from_numpy(state).to(device=device, dtype=torch.float32).unsqueeze(0)
        dist, value = model.action_distribution(state_tensor)
        action_tensor = dist.sample()
        action = int(action_tensor.item())
        next_state, reward, done, _, _ = env.step(action)

        log_probs.append(dist.log_prob(action_tensor).squeeze(0))
        values.append(value.squeeze(0))
        rewards.append(float(reward))
        entropies.append(dist.entropy().squeeze(0))
        ep_qoe += float(reward)
        ep_steps += 1
        state = next_state

    returns = discounted_returns(rewards, gamma).to(device)
    values_t = torch.stack(values)
    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)
    advantages = returns - values_t.detach()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    actor_loss = -(log_probs_t * advantages).mean()
    value_loss = F.mse_loss(values_t, returns)
    entropy_bonus = entropies_t.mean()
    loss = actor_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_bonus

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    optimizer.step()

    return {
        "qoe": ep_qoe,
        "steps": ep_steps,
        "loss": float(loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_bonus.detach().cpu()),
    }


def train_pensieve(
    model_save_path=PENSIEVE_MODEL_PATH,
    pretrain_episodes=300,
    pretrain_lr=1e-4,
    episodes=2000,
    lr=1e-4,
    gamma=0.99,
    value_coef=0.5,
    entropy_coef=0.01,
    grad_clip=0.5,
    qoe_profile="pensieve",
    trace_split="train",
    trace_dir=None,
    seed=100003,
    eval_interval=100,
    validation_episodes=30,
    validation_split=None,
    validation_trace_dir=None,
    validation_sample_mode="stratified",
    validation_seed=20260507,
    early_stop_patience=8,
    min_episodes=400,
    progress_interval=20,
    history_save_path=None,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

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

    model = PensieveActorCritic(action_dim=ACTION_DIM).to(device)
    print(f"[Pensieve] device={device}")
    print(f"[Pensieve] validation_traces={len(validation_trace_files)} split={validation_split}")

    pretrain_optimizer = optim.Adam(model.parameters(), lr=float(pretrain_lr))
    pretrain_history = pretrain_with_mpc(
        model=model,
        optimizer=pretrain_optimizer,
        device=device,
        episodes=pretrain_episodes,
        qoe_profile=qoe_profile,
        trace_split=trace_split,
        trace_dir=trace_dir,
        progress_interval=progress_interval,
    )

    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    train_history = []
    validation_history = []
    best_score = -float("inf")
    episodes_without_improvement = 0
    start_time = time.time()

    for episode in range(1, int(episodes) + 1):
        env = ABREnv(qoe_profile=qoe_profile, trace_split=trace_split, trace_dir=trace_dir)
        metrics = train_one_episode(
            model=model,
            optimizer=optimizer,
            device=device,
            env=env,
            gamma=gamma,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            grad_clip=grad_clip,
        )
        metrics["episode"] = episode
        train_history.append(metrics)

        if progress_interval and episode % int(progress_interval) == 0:
            recent = train_history[-int(progress_interval):]
            elapsed = time.time() - start_time
            print(
                f"[Pensieve] episode={episode}/{episodes} "
                f"mean_qoe={np.mean([item['qoe'] for item in recent]):.3f} "
                f"loss={np.mean([item['loss'] for item in recent]):.4f} "
                f"elapsed={format_duration(elapsed)}"
            )

        should_validate = validation_trace_files and (
            episode == 1 or episode % int(eval_interval) == 0 or episode == int(episodes)
        )
        if should_validate:
            summary, _ = evaluate_pensieve_model(
                model=model,
                trace_files=validation_trace_files,
                trace_split=validation_split,
                trace_dir=validation_trace_dir,
                qoe_profile=qoe_profile,
                device=device,
                progress_interval=0,
            )
            record = {"episode": episode, **summary}
            validation_history.append(record)
            print(
                f"[Pensieve] validation episode={episode} "
                f"mean_qoe={summary['mean_qoe']:.3f} "
                f"p10_qoe={summary['p10_qoe']:.3f} "
                f"stall={summary['mean_stall_time']:.3f}"
            )
            if summary["mean_qoe"] > best_score + 1e-3:
                best_score = summary["mean_qoe"]
                episodes_without_improvement = 0
                saved_path = save_pensieve_checkpoint(
                    model,
                    model_save_path,
                    metadata={"stage": "pensieve", "episode": episode, "best_mean_qoe": best_score},
                )
                print(f"[Pensieve] saved best checkpoint -> {saved_path}")
            else:
                episodes_without_improvement += 1

            if (
                early_stop_patience > 0
                and episode >= int(min_episodes)
                and episodes_without_improvement >= int(early_stop_patience)
            ):
                print(
                    f"[Pensieve] early stop at episode={episode}, "
                    f"best_mean_qoe={best_score:.3f}"
                )
                break

    if not validation_trace_files:
        save_pensieve_checkpoint(model, model_save_path, metadata={"stage": "pensieve"})

    if history_save_path:
        history_path = Path(history_save_path)
    else:
        history_path = model_save_path.parent / "pensieve_validation_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {
                "pretrain": pretrain_history,
                "train": train_history,
                "validation": validation_history,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[Pensieve] saved history: {history_path}")
    return model_save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PyTorch Pensieve baseline.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--pretrain-episodes", type=int, default=300)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--save-path", type=Path, default=PENSIEVE_MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    train_pensieve(
        model_save_path=args.save_path,
        pretrain_episodes=args.pretrain_episodes,
        episodes=args.episodes,
        validation_episodes=args.validation_episodes,
    )


if __name__ == "__main__":
    main()
