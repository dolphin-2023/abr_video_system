import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from abr_env import ABREnv
from settings import SEGMENT_DURATION
from trace_utils import filter_high_bandwidth_trace_files


def infer_quality_index(state_matrix, bitrates_kbps):
    normalized = float(state_matrix[0, -1])
    if normalized <= 0:
        return 0
    bitrate = normalized * max(bitrates_kbps)
    return int(np.argmin(np.abs(np.asarray(bitrates_kbps, dtype=np.float32) - bitrate)))


def bola_expert_policy(state_matrix, available_bitrates_kbps):
    """BOLA expert with a throughput guard for safer behavior cloning data."""
    buffer_level = float(state_matrix[1, -1]) * 10.0
    min_bitrate = float(available_bitrates_kbps[0])
    max_bitrate = float(available_bitrates_kbps[-1])

    utilities = [math.log(float(bitrate) / min_bitrate) for bitrate in available_bitrates_kbps]
    current_idx = infer_quality_index(state_matrix, available_bitrates_kbps)

    v = 0.9
    gamma = 5.0
    best_score = -float("inf")
    decision = 0

    for idx, bitrate_kbps in enumerate(available_bitrates_kbps):
        bitrate_weight = float(bitrate_kbps) / max_bitrate
        score = v * utilities[idx] - gamma * bitrate_weight / (buffer_level + 0.1)
        if score > best_score:
            best_score = score
            decision = idx

    if abs(decision - current_idx) > 1:
        decision = current_idx + (1 if decision > current_idx else -1)

    throughput_history = np.asarray(state_matrix[2], dtype=np.float32) * 1000.0
    positive_history = throughput_history[throughput_history > 1.0]
    if positive_history.size == 0:
        return 0

    harmonic_kbps = positive_history.size / np.sum(1.0 / np.maximum(positive_history, 1.0))
    safe_kbps = float(harmonic_kbps) * 0.85
    safe_indices = [
        idx for idx, bitrate in enumerate(available_bitrates_kbps)
        if float(bitrate) <= safe_kbps
    ]
    safe_idx = max(safe_indices) if safe_indices else 0

    if buffer_level < SEGMENT_DURATION:
        safe_idx = min(safe_idx, current_idx)

    decision = min(decision, safe_idx)
    return max(0, min(decision, len(available_bitrates_kbps) - 1))


def harmonic_throughput_kbps(state_matrix):
    history = np.asarray(state_matrix[2], dtype=np.float32) * 1000.0
    history = history[history > 1.0]
    if history.size == 0:
        return 0.0
    return float(history.size / np.sum(1.0 / np.maximum(history, 1.0)))


def mpc_expert_policy(state_matrix, env, safety_factor=0.9, jump_limit=1):
    est_kbps = harmonic_throughput_kbps(state_matrix) * float(safety_factor)
    if est_kbps <= 1.0:
        return 0

    current_idx = infer_quality_index(state_matrix, env.bitrates_kbps)
    buffer_level = float(state_matrix[1, -1]) * 10.0
    last_bitrate = float(env.bitrates_kbps[current_idx])

    best_score = -float("inf")
    best_action = 0
    for action, bitrate in enumerate(env.bitrates_kbps):
        chunk_kbit = env.estimate_chunk_size_kbit(action, env.current_segment)
        download_time = chunk_kbit / max(10.0, est_kbps)
        stall_time = max(0.0, download_time - buffer_level)
        quality_score = env.compute_quality_score(float(bitrate))
        smooth_penalty = env.compute_smooth_penalty(float(bitrate), last_bitrate)
        score = quality_score - env.penalty_stall * stall_time - env.penalty_smooth * smooth_penalty
        if score > best_score:
            best_score = score
            best_action = action

    if jump_limit is not None and int(jump_limit) >= 0:
        jump_limit = int(jump_limit)
        lower = max(0, current_idx - jump_limit)
        upper = min(env.num_bitrates - 1, current_idx + jump_limit)
        best_action = max(lower, min(upper, best_action))
    return max(0, min(env.num_bitrates - 1, int(best_action)))


def high_bitrate_expert_policy(
    state_matrix,
    env,
    safety_factor=1.1,
    jump_limit=1,
    min_buffer_seconds=8.0,
    max_stall_seconds=0.25,
):
    """MPC variant that deliberately covers high bitrate actions when safe."""
    fallback = mpc_expert_policy(
        state_matrix,
        env,
        safety_factor=0.95,
        jump_limit=jump_limit,
    )
    est_kbps = harmonic_throughput_kbps(state_matrix) * float(safety_factor)
    if est_kbps <= 1.0:
        return fallback

    current_idx = infer_quality_index(state_matrix, env.bitrates_kbps)
    buffer_level = float(state_matrix[1, -1]) * 10.0
    safe_action = 0
    for action, bitrate in enumerate(env.bitrates_kbps):
        chunk_kbit = env.estimate_chunk_size_kbit(action, env.current_segment)
        download_time = chunk_kbit / max(10.0, est_kbps)
        stall_time = max(0.0, download_time - buffer_level)
        if float(bitrate) <= est_kbps and stall_time <= float(max_stall_seconds):
            safe_action = action

    if buffer_level >= float(min_buffer_seconds) and safe_action > fallback:
        decision = safe_action
    else:
        decision = fallback

    if jump_limit is not None and int(jump_limit) >= 0:
        jump_limit = int(jump_limit)
        lower = max(0, current_idx - jump_limit)
        upper = min(env.num_bitrates - 1, current_idx + jump_limit)
        decision = max(lower, min(upper, decision))
    return max(0, min(env.num_bitrates - 1, int(decision)))


def apply_trace_filter(
    env,
    trace_filter="all",
    min_mean_throughput_kbps=5000.0,
    high_bandwidth_percentile=70.0,
):
    mode = str(trace_filter or "all").strip().lower().replace("_", "-")
    if mode in {"all", "none"}:
        return 0
    if mode != "high-bandwidth":
        raise ValueError(f"Unknown trace_filter: {trace_filter}")
    if not env.trace_files:
        return 0

    before_count = len(env.trace_files)
    selected, threshold = filter_high_bandwidth_trace_files(
        env.trace_files,
        min_mean_throughput_kbps=min_mean_throughput_kbps,
        percentile=high_bandwidth_percentile,
    )
    env.trace_files = sorted(selected)
    env.trace_files_by_source = env._group_trace_files(env.trace_files)
    print(
        f"[DataCollector] trace_filter=high-bandwidth "
        f"kept={len(env.trace_files)}/{before_count} "
        f"threshold_mean_kbps={threshold:.1f}"
    )
    return before_count - len(env.trace_files)


def collect_expert_trajectories(
    num_episodes=100,
    save_path="expert_data.npz",
    gamma=1.0,
    trace_split=None,
    trace_dir=None,
    stats_path=None,
    qoe_profile=None,
    expert_policy="bola",
    mpc_ratio=0.35,
    high_bitrate_ratio=0.0,
    expert_seed=20260507,
    mpc_safety_factor=0.9,
    high_mpc_safety_factor=1.1,
    high_mpc_min_buffer=8.0,
    high_mpc_max_stall=0.25,
    high_mpc_jump_limit=1,
    trace_filter="all",
    min_mean_throughput_kbps=5000.0,
    high_bandwidth_percentile=70.0,
    exclude_trace_files=None,
    progress_interval=10,
):
    env = ABREnv(trace_split=trace_split, trace_dir=trace_dir, qoe_profile=qoe_profile)
    excluded_trace_names = {
        Path(trace_file).name
        for trace_file in (exclude_trace_files or [])
    }
    if excluded_trace_names:
        before_count = len(env.trace_files)
        env.trace_files = [
            trace_file
            for trace_file in env.trace_files
            if Path(trace_file).name not in excluded_trace_names
        ]
        env.trace_files_by_source = env._group_trace_files(env.trace_files)
        if not env.trace_files:
            raise RuntimeError("All traces were excluded from expert collection.")
        print(
            f"[DataCollector] excluded_validation_traces="
            f"{before_count - len(env.trace_files)}"
        )
    filtered_trace_count = apply_trace_filter(
        env,
        trace_filter=trace_filter,
        min_mean_throughput_kbps=min_mean_throughput_kbps,
        high_bandwidth_percentile=high_bandwidth_percentile,
    )
    bitrates = env.bitrates_kbps
    trajectories = []
    total_samples = 0
    rng = random.Random(expert_seed)
    expert_policy = str(expert_policy).strip().lower().replace("-", "_")
    expert_episode_counts = {"bola": 0, "mpc": 0, "high_mpc": 0}

    print(
        f"[DataCollector] collecting expert trajectories, episodes={num_episodes}, "
        f"policy={expert_policy}, mpc_ratio={mpc_ratio:.2f}"
    )
    for episode in range(num_episodes):
        episode_policy = expert_policy
        if expert_policy == "mixed":
            draw = rng.random()
            if draw < float(high_bitrate_ratio):
                episode_policy = "high_mpc"
            elif draw < float(high_bitrate_ratio) + float(mpc_ratio):
                episode_policy = "mpc"
            else:
                episode_policy = "bola"
        if episode_policy not in {"bola", "mpc", "high_mpc"}:
            raise ValueError(f"Unknown expert_policy: {expert_policy}")
        expert_episode_counts[episode_policy] += 1

        state, _ = env.reset()
        done = False
        ep_states = []
        ep_actions = []
        ep_rewards = []
        trace_name = getattr(env, "current_trace_name", "unknown")

        while not done:
            if episode_policy == "high_mpc":
                action = high_bitrate_expert_policy(
                    state,
                    env,
                    safety_factor=high_mpc_safety_factor,
                    jump_limit=high_mpc_jump_limit,
                    min_buffer_seconds=high_mpc_min_buffer,
                    max_stall_seconds=high_mpc_max_stall,
                )
            elif episode_policy == "mpc":
                action = mpc_expert_policy(
                    state,
                    env,
                    safety_factor=mpc_safety_factor,
                    jump_limit=1,
                )
            else:
                action = bola_expert_policy(state, bitrates)
            ep_states.append(state.copy())
            ep_actions.append(action)
            state, reward, done, _, _ = env.step(action)
            ep_rewards.append(reward)

        if len(ep_actions) < 2:
            continue

        returns = np.zeros(len(ep_rewards), dtype=np.float32)
        running = 0.0
        for idx in range(len(ep_rewards) - 1, -1, -1):
            running = float(ep_rewards[idx]) + gamma * running
            returns[idx] = running

        trajectories.append({
            "states": np.asarray(ep_states, dtype=np.float32),
            "actions": np.asarray(ep_actions, dtype=np.int64),
            "rewards": np.asarray(ep_rewards, dtype=np.float32),
            "returns": returns,
            "timesteps": np.arange(len(ep_actions), dtype=np.int64),
            "trace_name": trace_name,
            "expert_policy": episode_policy,
        })
        total_samples += len(ep_actions)

        if progress_interval and (episode + 1) % int(progress_interval) == 0:
            ep_return = float(np.sum(ep_rewards))
            print(
                f"[DataCollector] {episode + 1}/{num_episodes} "
                f"episode_return={ep_return:.2f} "
                f"steps={len(ep_actions)}"
            )

    if not trajectories:
        raise RuntimeError("No trajectories were collected.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "expert_policy": expert_policy,
        "expert_episode_counts": expert_episode_counts,
        "mpc_ratio": float(mpc_ratio),
        "high_bitrate_ratio": float(high_bitrate_ratio),
        "mpc_safety_factor": float(mpc_safety_factor),
        "high_mpc_safety_factor": float(high_mpc_safety_factor),
        "high_mpc_min_buffer": float(high_mpc_min_buffer),
        "high_mpc_max_stall": float(high_mpc_max_stall),
        "high_mpc_jump_limit": int(high_mpc_jump_limit),
        "trace_filter": str(trace_filter),
        "filtered_trace_count": int(filtered_trace_count),
        "min_mean_throughput_kbps": float(min_mean_throughput_kbps),
        "high_bandwidth_percentile": float(high_bandwidth_percentile),
        "excluded_trace_count": len(excluded_trace_names),
        "excluded_trace_names": sorted(excluded_trace_names),
        "download_model": "time_integrated_trace_payload_0.95_rtt_0.08",
        "trace_split": trace_split or os.environ.get("ABR_TRACE_SPLIT", "train"),
        "trace_dir": env.trace_dir,
        "qoe_profile": env.qoe_profile,
        "num_episodes": int(num_episodes),
        "gamma": float(gamma),
    }
    np.savez_compressed(
        save_path,
        trajectories=np.asarray(trajectories, dtype=object),
        metadata=np.asarray(metadata, dtype=object),
    )

    init_returns = [float(traj["returns"][0]) for traj in trajectories]
    actions = np.concatenate([traj["actions"] for traj in trajectories])
    rewards = np.concatenate([traj["rewards"] for traj in trajectories])
    action_counts = np.bincount(actions, minlength=env.num_bitrates)
    stats = {
        **metadata,
        "save_path": str(save_path),
        "num_trajectories": len(trajectories),
        "num_samples": int(total_samples),
        "action_counts": {
            str(idx): int(action_counts[idx])
            for idx in range(env.num_bitrates)
        },
        "reward_min": float(rewards.min()),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "return_min": float(min(init_returns)),
        "return_mean": float(np.mean(init_returns)),
        "return_max": float(max(init_returns)),
    }
    if stats_path is None:
        stats_path = save_path.with_suffix(".stats.json")
    Path(stats_path).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[DataCollector] saved {len(trajectories)} trajectories, "
        f"{total_samples} samples -> {save_path}"
    )
    print(
        f"[DataCollector] raw return stats: "
        f"max={max(init_returns):.2f}, "
        f"mean={np.mean(init_returns):.2f}, "
        f"min={min(init_returns):.2f}"
    )
    print(f"[DataCollector] saved stats -> {stats_path}")


def read_trace_list(path):
    if not path:
        return []
    trace_list_path = Path(path)
    if not trace_list_path.exists():
        raise FileNotFoundError(f"Trace exclude list not found: {trace_list_path}")
    return [
        line.strip()
        for line in trace_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save-path", default="expert_data.npz")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--trace-split", default=None)
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--qoe-profile", default=None)
    parser.add_argument("--expert-policy", default="bola", choices=["bola", "mpc", "mixed", "high-mpc"])
    parser.add_argument("--mpc-ratio", type=float, default=0.35)
    parser.add_argument("--high-bitrate-ratio", type=float, default=0.0)
    parser.add_argument("--expert-seed", type=int, default=20260507)
    parser.add_argument("--mpc-safety-factor", type=float, default=0.9)
    parser.add_argument("--high-mpc-safety-factor", type=float, default=1.1)
    parser.add_argument("--high-mpc-min-buffer", type=float, default=8.0)
    parser.add_argument("--high-mpc-max-stall", type=float, default=0.25)
    parser.add_argument("--high-mpc-jump-limit", type=int, default=1)
    parser.add_argument("--trace-filter", default="all", choices=["all", "high-bandwidth"])
    parser.add_argument("--min-mean-throughput-kbps", type=float, default=5000.0)
    parser.add_argument("--high-bandwidth-percentile", type=float, default=70.0)
    parser.add_argument("--exclude-trace-file", action="append", default=[])
    parser.add_argument("--exclude-trace-list", default=None)
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()
    exclude_trace_files = list(args.exclude_trace_file or [])
    exclude_trace_files.extend(read_trace_list(args.exclude_trace_list))
    collect_expert_trajectories(
        num_episodes=args.episodes,
        save_path=args.save_path,
        gamma=args.gamma,
        trace_split=args.trace_split,
        trace_dir=args.trace_dir,
        stats_path=args.stats_path,
        qoe_profile=args.qoe_profile,
        expert_policy=args.expert_policy,
        mpc_ratio=args.mpc_ratio,
        high_bitrate_ratio=args.high_bitrate_ratio,
        expert_seed=args.expert_seed,
        mpc_safety_factor=args.mpc_safety_factor,
        high_mpc_safety_factor=args.high_mpc_safety_factor,
        high_mpc_min_buffer=args.high_mpc_min_buffer,
        high_mpc_max_stall=args.high_mpc_max_stall,
        high_mpc_jump_limit=args.high_mpc_jump_limit,
        trace_filter=args.trace_filter,
        min_mean_throughput_kbps=args.min_mean_throughput_kbps,
        high_bandwidth_percentile=args.high_bandwidth_percentile,
        exclude_trace_files=exclude_trace_files,
        progress_interval=args.progress_interval,
    )
