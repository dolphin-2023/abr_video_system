from pathlib import Path

import numpy as np
import torch

from abr_env import ABREnv
from return_utils import load_return_to_go_processor
from settings import ACTION_DIM
from trace_utils import sample_trace_files


def build_eval_trace_files(
    trace_split="train",
    trace_dir=None,
    episodes=10,
    sample_mode="stratified",
    seed=20260507,
    qoe_profile="pensieve",
):
    if episodes == 0:
        return []

    probe_env = ABREnv(
        trace_split=trace_split,
        trace_dir=trace_dir,
        random_start=False,
        qoe_profile=qoe_profile,
    )
    trace_files = sorted(probe_env.trace_files)
    if not trace_files:
        return []
    return sample_trace_files(
        trace_files,
        episodes=episodes,
        mode=sample_mode,
        seed=seed,
    )


def constrained_action(action, last_action, num_actions, jump_limit=1):
    action = int(action)
    if jump_limit is not None and jump_limit >= 0:
        lower = max(0, int(last_action) - int(jump_limit))
        upper = min(num_actions - 1, int(last_action) + int(jump_limit))
        action = max(lower, min(upper, action))
    return max(0, min(num_actions - 1, action))


def summarize_model_episodes(
    episodes,
    label,
    qoe_profile,
    total_chunks,
    total_bitrate,
    total_stall,
    total_smooth,
    total_duration,
    rebuffer_chunks,
    action_counts,
):
    qoes = np.asarray([item["qoe"] for item in episodes], dtype=np.float32)
    stall_times = np.asarray([item["stall_time"] for item in episodes], dtype=np.float32)
    smoothness_penalties = np.asarray(
        [item["smoothness_penalty"] for item in episodes],
        dtype=np.float32,
    )
    switch_counts = np.asarray(
        [item["quality_switch_count"] for item in episodes],
        dtype=np.float32,
    )

    return {
        "policy": label,
        "qoe_profile": qoe_profile,
        "episodes": len(episodes),
        "chunks": total_chunks,
        "mean_qoe": float(np.mean(qoes)) if qoes.size else 0.0,
        "median_qoe": float(np.median(qoes)) if qoes.size else 0.0,
        "p10_qoe": float(np.percentile(qoes, 10)) if qoes.size else 0.0,
        "worst_qoe": float(np.min(qoes)) if qoes.size else 0.0,
        "best_qoe": float(np.max(qoes)) if qoes.size else 0.0,
        "mean_bitrate": total_bitrate / max(total_chunks, 1),
        "mean_stall_time": float(np.mean(stall_times)) if stall_times.size else 0.0,
        "mean_smoothness_penalty": (
            float(np.mean(smoothness_penalties)) if smoothness_penalties.size else 0.0
        ),
        "rebuffer_ratio": total_stall / max(total_duration, 1e-8),
        "stall_chunk_ratio": rebuffer_chunks / max(total_chunks, 1),
        "quality_switch_count": float(np.mean(switch_counts)) if switch_counts.size else 0.0,
        "action_counts": {
            str(idx): int(action_counts[idx])
            for idx in range(len(action_counts))
        },
        "action_distribution": {
            str(idx): float(action_counts[idx] / max(int(action_counts.sum()), 1))
            for idx in range(len(action_counts))
        },
    }


def evaluate_model_policy_batched(
    model,
    trace_files,
    trace_split,
    qoe_profile,
    target_return,
    device,
    trace_dir=None,
    jump_limit=1,
    progress_interval=0,
    label="model",
    stats_path=None,
    batch_size=8,
):
    was_training = model.training
    model.eval()
    rtg = load_return_to_go_processor(
        target_return=target_return,
        stats_path=stats_path,
    )

    episodes = []
    total_chunks = 0
    total_bitrate = 0.0
    total_stall = 0.0
    total_smooth = 0.0
    total_duration = 0.0
    rebuffer_chunks = 0
    action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
    batch_size = max(1, int(batch_size))

    for start_idx in range(0, len(trace_files), batch_size):
        batch_trace_files = trace_files[start_idx:start_idx + batch_size]
        envs = [
            ABREnv(
                trace_split=trace_split,
                trace_dir=trace_dir,
                trace_file=trace_file,
                random_start=False,
                qoe_profile=qoe_profile,
            )
            for trace_file in batch_trace_files
        ]
        states = []
        histories = []
        for env in envs:
            state, _ = env.reset()
            states.append(state)
            histories.append(model.new_history())

        active = [True] * len(envs)
        steps = [0] * len(envs)
        last_actions = [0] * len(envs)
        prev_actions = [None] * len(envs)
        current_returns = [rtg.target_return] * len(envs)
        ep_qoes = [0.0] * len(envs)
        ep_bitrates = [0.0] * len(envs)
        ep_stalls = [0.0] * len(envs)
        ep_smooths = [0.0] * len(envs)
        ep_switches = [0] * len(envs)
        ep_steps = [0] * len(envs)
        ep_action_counts = [np.zeros(ACTION_DIM, dtype=np.int64) for _ in envs]

        while any(active):
            active_indices = [idx for idx, is_active in enumerate(active) if is_active]
            state_tensor = torch.from_numpy(
                np.stack([states[idx] for idx in active_indices], axis=0)
            ).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ).unsqueeze(1)
            logits, return_emb, state_emb, time_emb = model.predict_logits_batch(
                states=state_tensor,
                target_returns=[current_returns[idx] for idx in active_indices],
                timesteps=[steps[idx] for idx in active_indices],
                histories=[histories[idx] for idx in active_indices],
            )
            selected_actions = torch.argmax(logits, dim=1).detach().cpu().tolist()

            for local_idx, env_idx in enumerate(active_indices):
                env = envs[env_idx]
                action = constrained_action(
                    selected_actions[local_idx],
                    last_actions[env_idx],
                    env.num_bitrates,
                    jump_limit,
                )
                action_counts[action] += 1
                ep_action_counts[env_idx][action] += 1

                next_state, reward, done, _, info = env.step(action)
                current_returns[env_idx] = rtg.update(current_returns[env_idx], reward)
                model.append_history(
                    history=histories[env_idx],
                    return_emb=return_emb[local_idx:local_idx + 1],
                    state_emb=state_emb[local_idx:local_idx + 1],
                    time_emb=time_emb[local_idx:local_idx + 1],
                    action=action,
                    device=device,
                )

                ep_qoes[env_idx] += float(reward)
                ep_bitrates[env_idx] += float(info["bitrate_kbps"])
                ep_stalls[env_idx] += float(info["stall_time"])
                ep_smooths[env_idx] += float(info["smoothness_penalty"])
                ep_steps[env_idx] += 1

                if info["stall_time"] > 1e-6:
                    rebuffer_chunks += 1
                if prev_actions[env_idx] is not None and action != prev_actions[env_idx]:
                    ep_switches[env_idx] += 1
                prev_actions[env_idx] = action
                last_actions[env_idx] = action
                states[env_idx] = next_state
                steps[env_idx] += 1
                active[env_idx] = not bool(done)

        for env_idx, trace_file in enumerate(batch_trace_files):
            env = envs[env_idx]
            total_chunks += ep_steps[env_idx]
            total_bitrate += ep_bitrates[env_idx]
            total_stall += ep_stalls[env_idx]
            total_smooth += ep_smooths[env_idx]
            total_duration += ep_steps[env_idx] * env.segment_duration
            episodes.append({
                "trace": Path(trace_file).name,
                "qoe": ep_qoes[env_idx],
                "mean_bitrate": ep_bitrates[env_idx] / max(ep_steps[env_idx], 1),
                "stall_time": ep_stalls[env_idx],
                "smoothness_penalty": ep_smooths[env_idx],
                "quality_switch_count": ep_switches[env_idx],
                "chunks": ep_steps[env_idx],
                "action_counts": {
                    str(idx): int(ep_action_counts[env_idx][idx])
                    for idx in range(len(ep_action_counts[env_idx]))
                },
            })

        if progress_interval and len(episodes) % int(progress_interval) == 0:
            recent_qoes = [item["qoe"] for item in episodes]
            print(
                f"[TrainEval] {label} "
                f"episode={len(episodes)}/{len(trace_files)} "
                f"mean_qoe={np.mean(recent_qoes):.3f}"
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


@torch.no_grad()
def evaluate_model_policy(
    model,
    trace_files,
    trace_split,
    qoe_profile,
    target_return,
    device,
    trace_dir=None,
    jump_limit=1,
    progress_interval=0,
    label="model",
    stats_path=None,
    batch_size=1,
):
    if int(batch_size or 1) > 1 and hasattr(model, "predict_logits_batch"):
        return evaluate_model_policy_batched(
            model=model,
            trace_files=trace_files,
            trace_split=trace_split,
            trace_dir=trace_dir,
            qoe_profile=qoe_profile,
            target_return=target_return,
            device=device,
            jump_limit=jump_limit,
            progress_interval=progress_interval,
            label=label,
            stats_path=stats_path,
            batch_size=batch_size,
        )

    was_training = model.training
    model.eval()
    rtg = load_return_to_go_processor(
        target_return=target_return,
        stats_path=stats_path,
    )

    episodes = []
    total_chunks = 0
    total_bitrate = 0.0
    total_stall = 0.0
    total_smooth = 0.0
    total_duration = 0.0
    rebuffer_chunks = 0
    action_counts = np.zeros(ACTION_DIM, dtype=np.int64)

    for trace_file in trace_files:
        env = ABREnv(
            trace_split=trace_split,
            trace_dir=trace_dir,
            trace_file=trace_file,
            random_start=False,
            qoe_profile=qoe_profile,
        )
        state, _ = env.reset()
        history = model.new_history()
        done = False
        step = 0
        last_action = 0
        prev_action = None
        ep_qoe = 0.0
        ep_bitrate = 0.0
        ep_stall = 0.0
        ep_smooth = 0.0
        ep_switches = 0
        ep_steps = 0
        ep_action_counts = np.zeros(ACTION_DIM, dtype=np.int64)
        current_target_return = rtg.target_return

        while not done:
            state_tensor = torch.from_numpy(state).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ).unsqueeze(0).unsqueeze(0)
            logits, return_emb, state_emb, time_emb = model.predict_logits(
                state=state_tensor,
                target_return=current_target_return,
                timestep=step,
                history=history,
            )
            action = int(torch.argmax(logits).item())
            action = constrained_action(action, last_action, env.num_bitrates, jump_limit)
            action_counts[action] += 1
            ep_action_counts[action] += 1

            next_state, reward, done, _, info = env.step(action)
            current_target_return = rtg.update(current_target_return, reward)
            model.append_history(
                history=history,
                return_emb=return_emb,
                state_emb=state_emb,
                time_emb=time_emb,
                action=action,
                device=device,
            )

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
            last_action = action
            state = next_state
            step += 1

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
            recent_qoes = [item["qoe"] for item in episodes]
            print(
                f"[TrainEval] {label} "
                f"episode={len(episodes)}/{len(trace_files)} "
                f"mean_qoe={np.mean(recent_qoes):.3f}"
            )

    qoes = np.asarray([item["qoe"] for item in episodes], dtype=np.float32)
    stall_times = np.asarray([item["stall_time"] for item in episodes], dtype=np.float32)
    smoothness_penalties = np.asarray(
        [item["smoothness_penalty"] for item in episodes],
        dtype=np.float32,
    )
    switch_counts = np.asarray(
        [item["quality_switch_count"] for item in episodes],
        dtype=np.float32,
    )

    summary = {
        "policy": label,
        "qoe_profile": qoe_profile,
        "episodes": len(episodes),
        "chunks": total_chunks,
        "mean_qoe": float(np.mean(qoes)) if qoes.size else 0.0,
        "median_qoe": float(np.median(qoes)) if qoes.size else 0.0,
        "p10_qoe": float(np.percentile(qoes, 10)) if qoes.size else 0.0,
        "worst_qoe": float(np.min(qoes)) if qoes.size else 0.0,
        "best_qoe": float(np.max(qoes)) if qoes.size else 0.0,
        "mean_bitrate": total_bitrate / max(total_chunks, 1),
        "mean_stall_time": float(np.mean(stall_times)) if stall_times.size else 0.0,
        "mean_smoothness_penalty": (
            float(np.mean(smoothness_penalties)) if smoothness_penalties.size else 0.0
        ),
        "rebuffer_ratio": total_stall / max(total_duration, 1e-8),
        "stall_chunk_ratio": rebuffer_chunks / max(total_chunks, 1),
        "quality_switch_count": float(np.mean(switch_counts)) if switch_counts.size else 0.0,
        "action_counts": {
            str(idx): int(action_counts[idx])
            for idx in range(len(action_counts))
        },
        "action_distribution": {
            str(idx): float(action_counts[idx] / max(int(action_counts.sum()), 1))
            for idx in range(len(action_counts))
        },
    }

    if was_training:
        model.train()
    return summary, episodes
