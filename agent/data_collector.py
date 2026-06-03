import argparse
import json
import math
import multiprocessing
import os
import random
import shutil
import sys
import tempfile
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


def bba_expert_policy(state_matrix, available_bitrates_kbps, reservoir=5.0, cushion=15.0):
    """Classic buffer-based expert used to diversify high-bandwidth actions."""
    buffer_level = float(state_matrix[1, -1]) * 10.0
    bitrates = np.asarray(available_bitrates_kbps, dtype=np.float32)
    min_bitrate = float(bitrates[0])
    max_bitrate = float(bitrates[-1])

    if buffer_level <= float(reservoir):
        target_kbps = min_bitrate
    elif buffer_level >= float(cushion):
        target_kbps = max_bitrate
    else:
        span = max(float(cushion) - float(reservoir), 1e-6)
        ratio = (buffer_level - float(reservoir)) / span
        target_kbps = min_bitrate + ratio * (max_bitrate - min_bitrate)

    action = int(np.argmin(np.abs(bitrates - target_kbps)))
    return max(0, min(len(available_bitrates_kbps) - 1, action))


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


EXPERT_POLICIES = ("bola", "mpc", "bba", "high_mpc")


def normalize_expert_policy(policy):
    normalized = str(policy or "").strip().lower().replace("-", "_")
    if normalized == "highmpc":
        normalized = "high_mpc"
    return normalized


def make_expert_counts():
    return {name: 0 for name in EXPERT_POLICIES}


def choose_episode_policy(expert_policy, rng, config):
    expert_policy = normalize_expert_policy(expert_policy)
    if expert_policy != "mixed":
        return expert_policy

    bba_ratio = max(0.0, float(config.get("bba_ratio", 0.0)))
    high_ratio = max(0.0, float(config.get("high_bitrate_ratio", 0.0)))
    mpc_ratio = max(0.0, float(config.get("mpc_ratio", 0.35)))

    draw = rng.random()
    if draw < bba_ratio:
        return "bba"
    if draw < bba_ratio + high_ratio:
        return "high_mpc"
    if draw < bba_ratio + high_ratio + mpc_ratio:
        return "mpc"
    return "bola"


def expert_action(episode_policy, state, env, bitrates, config):
    episode_policy = normalize_expert_policy(episode_policy)
    if episode_policy == "bba":
        return bba_expert_policy(
            state,
            bitrates,
            reservoir=config.get("bba_reservoir", 5.0),
            cushion=config.get("bba_cushion", 15.0),
        )
    if episode_policy == "high_mpc":
        return high_bitrate_expert_policy(
            state, env,
            safety_factor=config.get("high_mpc_safety_factor", 1.1),
            jump_limit=config.get("high_mpc_jump_limit", 1),
            min_buffer_seconds=config.get("high_mpc_min_buffer", 8.0),
            max_stall_seconds=config.get("high_mpc_max_stall", 0.25),
        )
    if episode_policy == "mpc":
        return mpc_expert_policy(
            state, env,
            safety_factor=config.get("mpc_safety_factor", 0.9),
            jump_limit=1,
        )
    return bola_expert_policy(state, bitrates)


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


def _worker_collect_episodes(worker_id, num_episodes, temp_dir, config):
    """Collect a subset of episodes inside a single worker process.

    Saves trajectories to a temporary npz under *temp_dir* so that the
    parent process can merge them without pickling large lists across
    process boundaries.

    Parameters in *config* are the same keyword arguments accepted by
    :func:`collect_expert_trajectories`.
    """
    seed = config["expert_seed"] + worker_id
    rng = random.Random(seed)
    gamma = config.get("gamma", 1.0)
    expert_policy = config["expert_policy"]
    exclude_trace_files = config.get("exclude_trace_files") or []

    env = ABREnv(
        trace_split=config.get("trace_split"),
        trace_dir=config.get("trace_dir"),
        qoe_profile=config.get("qoe_profile"),
    )

    excluded_trace_names = {Path(t).name for t in exclude_trace_files}
    if excluded_trace_names:
        env.trace_files = [
            tf for tf in env.trace_files
            if Path(tf).name not in excluded_trace_names
        ]
        env.trace_files_by_source = env._group_trace_files(env.trace_files)

    apply_trace_filter(
        env,
        trace_filter=config.get("trace_filter", "all"),
        min_mean_throughput_kbps=config.get("min_mean_throughput_kbps", 5000.0),
        high_bandwidth_percentile=config.get("high_bandwidth_percentile", 70.0),
    )

    bitrates = env.bitrates_kbps
    trajectories = []
    expert_episode_counts = make_expert_counts()

    for _ in range(num_episodes):
        episode_policy = choose_episode_policy(expert_policy, rng, config)
        if episode_policy not in EXPERT_POLICIES:
            raise ValueError(f"Unknown expert_policy: {expert_policy}")

        expert_episode_counts[episode_policy] += 1

        state, _ = env.reset()
        done = False
        ep_states, ep_actions, ep_rewards = [], [], []
        trace_name = getattr(env, "current_trace_name", "unknown")

        while not done:
            action = expert_action(episode_policy, state, env, bitrates, config)

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

    temp_path = Path(temp_dir) / f"temp_worker_{worker_id:04d}.npz"
    np.savez_compressed(
        temp_path,
        trajectories=np.asarray(trajectories, dtype=object),
        metadata=np.asarray(
            {"expert_episode_counts": expert_episode_counts,
             "worker_id": worker_id,
             "num_trajectories": len(trajectories)},
            dtype=object,
        ),
    )
    return str(temp_path)


def _collect_parallel(
    num_episodes,
    save_path,
    gamma,
    trace_split,
    trace_dir,
    stats_path,
    qoe_profile,
    expert_policy,
    expert_seed,
    exclude_trace_files,
    progress_interval,
    workers,
    **extra_config,
):
    """Multiprocess collection: each worker saves a temp npz, then merge."""
    episodes_per_worker = [num_episodes // workers] * workers
    for i in range(num_episodes % workers):
        episodes_per_worker[i] += 1

    active_workers = sum(1 for n in episodes_per_worker if n > 0)
    config = {
        "gamma": gamma,
        "trace_split": trace_split,
        "trace_dir": trace_dir,
        "qoe_profile": qoe_profile,
        "expert_policy": expert_policy,
        "expert_seed": expert_seed,
        "exclude_trace_files": exclude_trace_files or [],
    }
    config.update(extra_config)

    temp_dir = Path(tempfile.mkdtemp(prefix="abr_collect_"))
    print(
        f"[DataCollector] workers={active_workers} "
        f"episodes_per_worker={episodes_per_worker[:active_workers]} "
        f"temp_dir={temp_dir}"
    )

    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=active_workers) as pool:
            args_list = [
                (worker_id, episodes_per_worker[worker_id], str(temp_dir), config)
                for worker_id in range(active_workers)
            ]
            temp_paths = pool.starmap(_worker_collect_episodes, args_list)

        temp_paths = [Path(p) for p in temp_paths if p is not None]
        if not temp_paths:
            raise RuntimeError("No trajectories collected by any worker.")

        # Merge temp npz files
        all_trajectories = []
        merged_counts = make_expert_counts()
        for tp in temp_paths:
            raw = np.load(tp, allow_pickle=True)
            all_trajectories.extend(list(raw["trajectories"]))
            meta = dict(raw["metadata"].item())
            for key in EXPERT_POLICIES:
                merged_counts[key] += int(meta.get("expert_episode_counts", {}).get(key, 0))

        if not all_trajectories:
            raise RuntimeError("No trajectories collected by any worker.")

        total_samples = sum(len(t["actions"]) for t in all_trajectories)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "expert_policy": expert_policy,
            "expert_episode_counts": merged_counts,
            "bba_ratio": float(config.get("bba_ratio", 0.0)),
            "bba_reservoir": float(config.get("bba_reservoir", 5.0)),
            "bba_cushion": float(config.get("bba_cushion", 15.0)),
            "mpc_ratio": float(config.get("mpc_ratio", 0.35)),
            "high_bitrate_ratio": float(config.get("high_bitrate_ratio", 0.0)),
            "mpc_safety_factor": float(config.get("mpc_safety_factor", 0.9)),
            "high_mpc_safety_factor": float(config.get("high_mpc_safety_factor", 1.1)),
            "high_mpc_min_buffer": float(config.get("high_mpc_min_buffer", 8.0)),
            "high_mpc_max_stall": float(config.get("high_mpc_max_stall", 0.25)),
            "high_mpc_jump_limit": int(config.get("high_mpc_jump_limit", 1)),
            "trace_filter": str(config.get("trace_filter", "all")),
            "filtered_trace_count": 0,
            "min_mean_throughput_kbps": float(config.get("min_mean_throughput_kbps", 5000.0)),
            "high_bandwidth_percentile": float(config.get("high_bandwidth_percentile", 70.0)),
            "excluded_trace_count": len(config.get("exclude_trace_files", []) or []),
            "excluded_trace_names": sorted(
                Path(t).name for t in (config.get("exclude_trace_files") or [])
            ),
            "download_model": "time_integrated_trace_payload_0.95_rtt_0.08",
            "trace_split": trace_split or os.environ.get("ABR_TRACE_SPLIT", "train"),
            "trace_dir": "parallel",
            "qoe_profile": qoe_profile,
            "num_episodes": int(num_episodes),
            "gamma": float(gamma),
            "workers": int(workers),
        }
        np.savez_compressed(
            save_path,
            trajectories=np.asarray(all_trajectories, dtype=object),
            metadata=np.asarray(metadata, dtype=object),
        )

        init_returns = [float(t["returns"][0]) for t in all_trajectories]
        actions = np.concatenate([t["actions"] for t in all_trajectories])
        rewards = np.concatenate([t["rewards"] for t in all_trajectories])
        num_bitrates = max(int(actions.max()) + 1, 6) if actions.size else 6
        action_counts = np.bincount(actions, minlength=num_bitrates)
        stats = {
            **metadata,
            "save_path": str(save_path),
            "num_trajectories": len(all_trajectories),
            "num_samples": int(total_samples),
            "action_counts": {
                str(idx): int(action_counts[idx])
                for idx in range(len(action_counts))
            },
            "reward_min": float(rewards.min()),
            "reward_mean": float(rewards.mean()),
            "reward_max": float(rewards.max()),
            "return_min": float(min(init_returns)),
            "return_mean": float(np.mean(init_returns)),
            "return_max": float(max(init_returns)),
        }
        stats_path = Path(stats_path) if stats_path else save_path.with_suffix(".stats.json")
        stats_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"[DataCollector] saved {len(all_trajectories)} trajectories, "
            f"{total_samples} samples -> {save_path}"
        )
        print(
            f"[DataCollector] raw return stats: "
            f"max={max(init_returns):.2f}, "
            f"mean={np.mean(init_returns):.2f}, "
            f"min={min(init_returns):.2f}"
        )
        print(f"[DataCollector] saved stats -> {stats_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def collect_expert_trajectories(
    num_episodes=100,
    save_path="expert_data.npz",
    gamma=1.0,
    trace_split=None,
    trace_dir=None,
    stats_path=None,
    qoe_profile=None,
    expert_policy="bola",
    bba_ratio=0.0,
    bba_reservoir=5.0,
    bba_cushion=15.0,
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
    workers=1,
):
    expert_policy = str(expert_policy).strip().lower().replace("-", "_")

    if int(workers) > 1:
        return _collect_parallel(
            num_episodes=int(num_episodes),
            save_path=Path(save_path),
            gamma=float(gamma),
            trace_split=trace_split,
            trace_dir=trace_dir,
            stats_path=stats_path,
            qoe_profile=qoe_profile,
            expert_policy=expert_policy,
            expert_seed=int(expert_seed),
            exclude_trace_files=exclude_trace_files,
            progress_interval=int(progress_interval),
            workers=int(workers),
            # Pass all config through so workers see the same settings.
            bba_ratio=float(bba_ratio),
            bba_reservoir=float(bba_reservoir),
            bba_cushion=float(bba_cushion),
            mpc_ratio=float(mpc_ratio),
            high_bitrate_ratio=float(high_bitrate_ratio),
            mpc_safety_factor=float(mpc_safety_factor),
            high_mpc_safety_factor=float(high_mpc_safety_factor),
            high_mpc_min_buffer=float(high_mpc_min_buffer),
            high_mpc_max_stall=float(high_mpc_max_stall),
            high_mpc_jump_limit=int(high_mpc_jump_limit),
            trace_filter=trace_filter,
            min_mean_throughput_kbps=float(min_mean_throughput_kbps),
            high_bandwidth_percentile=float(high_bandwidth_percentile),
        )

    # --- serial path (workers == 1) ---
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
    expert_episode_counts = make_expert_counts()

    print(
        f"[DataCollector] collecting expert trajectories, episodes={num_episodes}, "
        f"policy={expert_policy}, bba_ratio={bba_ratio:.2f}, mpc_ratio={mpc_ratio:.2f}"
    )
    for episode in range(num_episodes):
        episode_policy = choose_episode_policy(
            expert_policy,
            rng,
            {
                "bba_ratio": bba_ratio,
                "high_bitrate_ratio": high_bitrate_ratio,
                "mpc_ratio": mpc_ratio,
            },
        )
        if episode_policy not in EXPERT_POLICIES:
            raise ValueError(f"Unknown expert_policy: {expert_policy}")
        expert_episode_counts[episode_policy] += 1

        state, _ = env.reset()
        done = False
        ep_states = []
        ep_actions = []
        ep_rewards = []
        trace_name = getattr(env, "current_trace_name", "unknown")

        while not done:
            action = expert_action(
                episode_policy,
                state,
                env,
                bitrates,
                {
                    "bba_reservoir": bba_reservoir,
                    "bba_cushion": bba_cushion,
                    "high_mpc_safety_factor": high_mpc_safety_factor,
                    "high_mpc_jump_limit": high_mpc_jump_limit,
                    "high_mpc_min_buffer": high_mpc_min_buffer,
                    "high_mpc_max_stall": high_mpc_max_stall,
                    "mpc_safety_factor": mpc_safety_factor,
                },
            )
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
        "bba_ratio": float(bba_ratio),
        "bba_reservoir": float(bba_reservoir),
        "bba_cushion": float(bba_cushion),
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


def collect_selected_expert_trajectories(
    trace_files,
    save_path,
    gamma=1.0,
    trace_split=None,
    trace_dir=None,
    qoe_profile="pensieve",
    expert_policies=None,
    bba_reservoir=5.0,
    bba_cushion=15.0,
    mpc_safety_factor=0.9,
    selection_unit="window",
    selection_window=20,
    selection_stride=5,
    selection_min_return=None,
    negative_return_weight=1.0,
    positive_return_weight=1.0,
    progress_interval=10,
    seed=20260507,
):
    """Build a selected expert pool from multiple baseline policies on shared traces.

    ``selection_unit="trace"`` keeps one best full trajectory per trace.
    ``selection_unit="window"`` performs sliding-window best-of-experts
    selection: each fixed-size window chooses the baseline with the highest
    local QoE on the same trace. ``selection_window`` is the QoE comparison
    horizon and is independent of the history width encoded in each state.
    """
    if expert_policies is None:
        expert_policies = ["bola", "mpc", "bba"]
    expert_policies = [normalize_expert_policy(p) for p in expert_policies]
    for p in expert_policies:
        if p not in EXPERT_POLICIES:
            raise ValueError(f"Unknown expert policy: {p}")

    selection_unit = str(selection_unit or "window").strip().lower().replace("-", "_")
    if selection_unit in {"segment", "local"}:
        selection_unit = "window"
    if selection_unit not in {"trace", "window"}:
        raise ValueError("selection_unit must be 'trace' or 'window'")
    selection_window = max(2, int(selection_window))
    selection_stride = max(1, int(selection_stride))
    selection_min_return = (
        -float("inf")
        if selection_min_return is None
        else float(selection_min_return)
    )
    negative_return_weight = max(0.0, float(negative_return_weight))
    positive_return_weight = max(0.0, float(positive_return_weight))

    trace_files = [Path(t) for t in trace_files]
    if not trace_files:
        raise RuntimeError("No trace files provided for selected expert collection.")

    trajectories = []
    policy_win_counts = {p: 0 for p in expert_policies}
    candidate_return_sums = {p: [] for p in expert_policies}
    total_samples = 0
    total_windows = 0
    attempted_windows = 0
    skipped_windows = 0

    print(
        f"[SelectedExpert] traces={len(trace_files)} "
        f"policies={expert_policies} selection={selection_unit} "
        f"window={selection_window} stride={selection_stride} "
        f"min_return={selection_min_return:g} "
        f"weights(neg,pos)=({negative_return_weight:g},{positive_return_weight:g}) "
        f"qoe_profile={qoe_profile}"
    )

    def discount_rewards(rewards):
        rewards = np.asarray(rewards, dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = float(rewards[t]) + float(gamma) * running
            returns[t] = running
        return returns

    def weight_for_return(raw_return):
        if float(raw_return) < 0.0:
            return negative_return_weight
        return positive_return_weight

    def rollout_policy(trace_file, policy_name):
        env = ABREnv(
            trace_split=trace_split,
            trace_dir=trace_dir,
            trace_file=str(trace_file),
            random_start=False,
            qoe_profile=qoe_profile,
        )
        state, _ = env.reset()
        done = False
        ep_states, ep_actions, ep_rewards = [], [], []

        while not done:
            action = expert_action(
                policy_name, state, env, env.bitrates_kbps,
                {
                    "bba_reservoir": bba_reservoir,
                    "bba_cushion": bba_cushion,
                    "mpc_safety_factor": mpc_safety_factor,
                },
            )
            ep_states.append(state.copy())
            ep_actions.append(action)
            state, reward, done, _, _ = env.step(action)
            ep_rewards.append(reward)

        if len(ep_actions) < 2:
            return None
        rewards = np.asarray(ep_rewards, dtype=np.float32)
        return {
            "states": np.asarray(ep_states, dtype=np.float32),
            "actions": np.asarray(ep_actions, dtype=np.int64),
            "rewards": rewards,
            "returns": discount_rewards(rewards),
            "timesteps": np.arange(len(ep_actions), dtype=np.int64),
            "trace_name": Path(trace_file).name,
            "expert_policy": policy_name,
        }

    for idx, trace_file in enumerate(trace_files):
        candidates = {}
        for policy_name in expert_policies:
            trajectory = rollout_policy(trace_file, policy_name)
            if trajectory is not None:
                candidates[policy_name] = trajectory

        if not candidates:
            continue

        if selection_unit == "trace":
            candidate_returns = {
                policy_name: float(np.sum(candidate["rewards"]))
                for policy_name, candidate in candidates.items()
            }
            selected_policy = max(candidate_returns, key=candidate_returns.get)
            selected_return = float(candidate_returns[selected_policy])
            selected = dict(candidates[selected_policy])
            selected["candidate_returns"] = dict(candidate_returns)
            selected["selection_unit"] = "trace"
            selected["local_return"] = selected_return
            selected["sample_weight"] = weight_for_return(selected_return)

            for policy_name, value in candidate_returns.items():
                candidate_return_sums[policy_name].append(float(value))
            attempted_windows += 1
            if selected_return < selection_min_return:
                skipped_windows += 1
                continue
            policy_win_counts[selected_policy] += 1
            trajectories.append(selected)
            total_samples += len(selected["actions"])
            total_windows += 1
        else:
            episode_len = min(len(candidate["actions"]) for candidate in candidates.values())
            max_start = max(0, episode_len - selection_window)
            starts = list(range(0, max_start + 1, selection_stride))
            if starts[-1] != max_start:
                starts.append(max_start)

            for start in starts:
                end = min(start + selection_window, episode_len)
                if end - start < 2:
                    continue
                candidate_returns = {
                    policy_name: float(np.sum(candidate["rewards"][start:end]))
                    for policy_name, candidate in candidates.items()
                }
                selected_policy = max(candidate_returns, key=candidate_returns.get)
                selected_return = float(candidate_returns[selected_policy])

                for policy_name, value in candidate_returns.items():
                    candidate_return_sums[policy_name].append(float(value))
                attempted_windows += 1
                if selected_return < selection_min_return:
                    skipped_windows += 1
                    continue

                selected_full = candidates[selected_policy]
                segment_rewards = selected_full["rewards"][start:end].copy()
                selected = {
                    "states": selected_full["states"][start:end].copy(),
                    "actions": selected_full["actions"][start:end].copy(),
                    "rewards": segment_rewards,
                    "returns": discount_rewards(segment_rewards),
                    "timesteps": selected_full["timesteps"][start:end].copy(),
                    "trace_name": selected_full["trace_name"],
                    "expert_policy": selected_policy,
                    "selection_unit": "window",
                    "window_start": int(start),
                    "window_end": int(end),
                    "local_return": selected_return,
                    "sample_weight": weight_for_return(selected_return),
                    "candidate_returns": dict(candidate_returns),
                }

                policy_win_counts[selected_policy] += 1
                trajectories.append(selected)
                total_samples += len(selected["actions"])
                total_windows += 1

        if progress_interval and (idx + 1) % int(progress_interval) == 0:
            print(
                f"[SelectedExpert] {idx + 1}/{len(trace_files)} traces, "
                f"selected_windows={total_windows}"
            )

    if not trajectories:
        raise RuntimeError("No trajectories were collected.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "collection_mode": "selected_expert_pool",
        "expert_policies": expert_policies,
        "policy_win_counts": policy_win_counts,
        "bba_reservoir": float(bba_reservoir),
        "bba_cushion": float(bba_cushion),
        "mpc_safety_factor": float(mpc_safety_factor),
        "selection_unit": selection_unit,
        "selection_window": int(selection_window),
        "selection_stride": int(selection_stride),
        "selection_min_return": (
            None
            if selection_min_return == -float("inf")
            else float(selection_min_return)
        ),
        "negative_return_weight": float(negative_return_weight),
        "positive_return_weight": float(positive_return_weight),
        "trace_split": trace_split or "",
        "trace_dir": str(trace_dir) if trace_dir else "",
        "qoe_profile": qoe_profile,
        "num_trajectories": len(trajectories),
        "num_traces_attempted": len(trace_files),
        "num_windows_attempted": int(attempted_windows),
        "num_windows_kept": int(total_windows),
        "num_windows_skipped": int(skipped_windows),
        "gamma": float(gamma),
        "seed": int(seed),
    }
    np.savez_compressed(
        save_path,
        trajectories=np.asarray(trajectories, dtype=object),
        metadata=np.asarray(metadata, dtype=object),
    )

    actions = np.concatenate([t["actions"] for t in trajectories])
    rewards = np.concatenate([t["rewards"] for t in trajectories])
    num_bitrates = max(int(actions.max()) + 1, 6) if actions.size else 6
    action_counts = np.bincount(actions, minlength=num_bitrates)
    init_returns = [float(t["returns"][0]) for t in trajectories]
    sample_weights = np.asarray(
        [float(t.get("sample_weight", 1.0)) for t in trajectories],
        dtype=np.float32,
    )
    candidate_return_summary = {
        p: {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for p, values in candidate_return_sums.items()
        if values
    }

    stats = {
        **metadata,
        "save_path": str(save_path),
        "num_samples": int(total_samples),
        "action_counts": {str(i): int(action_counts[i]) for i in range(len(action_counts))},
        "reward_min": float(rewards.min()),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "return_min": float(min(init_returns)),
        "return_mean": float(np.mean(init_returns)),
        "return_max": float(max(init_returns)),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_mean": float(sample_weights.mean()),
        "sample_weight_max": float(sample_weights.max()),
        "candidate_return_summary": candidate_return_summary,
    }
    stats_path = save_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[SelectedExpert] saved {len(trajectories)} trajectories ({total_samples} samples) -> {save_path}"
    )
    print(f"[SelectedExpert] policy wins: {policy_win_counts}")
    print(f"[SelectedExpert] action distribution: {action_counts.tolist()}")
    return trajectories


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] not in {"collect", "selected", "-h", "--help"}:
        sys.argv.insert(1, "collect")

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", help="Collection mode")

    # --- original expert collection ---
    orig = sub.add_parser("collect", help="Original mixed/single-policy collection")
    orig.add_argument("--episodes", type=int, default=100)
    orig.add_argument("--save-path", default="expert_data.npz")
    orig.add_argument("--gamma", type=float, default=1.0)
    orig.add_argument("--trace-split", default=None)
    orig.add_argument("--trace-dir", default=None)
    orig.add_argument("--stats-path", default=None)
    orig.add_argument("--qoe-profile", default=None)
    orig.add_argument("--expert-policy", default="bola", choices=["bola", "mpc", "bba", "mixed", "high-mpc", "high_mpc"])
    orig.add_argument("--bba-ratio", type=float, default=0.0)
    orig.add_argument("--bba-reservoir", type=float, default=5.0)
    orig.add_argument("--bba-cushion", type=float, default=15.0)
    orig.add_argument("--mpc-ratio", type=float, default=0.35)
    orig.add_argument("--high-bitrate-ratio", type=float, default=0.0)
    orig.add_argument("--expert-seed", type=int, default=20260507)
    orig.add_argument("--mpc-safety-factor", type=float, default=0.9)
    orig.add_argument("--high-mpc-safety-factor", type=float, default=1.1)
    orig.add_argument("--high-mpc-min-buffer", type=float, default=8.0)
    orig.add_argument("--high-mpc-max-stall", type=float, default=0.25)
    orig.add_argument("--high-mpc-jump-limit", type=int, default=1)
    orig.add_argument("--trace-filter", default="all", choices=["all", "high-bandwidth"])
    orig.add_argument("--min-mean-throughput-kbps", type=float, default=5000.0)
    orig.add_argument("--high-bandwidth-percentile", type=float, default=70.0)
    orig.add_argument("--exclude-trace-file", action="append", default=[])
    orig.add_argument("--exclude-trace-list", default=None)
    orig.add_argument("--progress-interval", type=int, default=10)
    orig.add_argument("--workers", type=int, default=1)

    # --- selected expert pool collection ---
    sel = sub.add_parser("selected", help="Per-trace best-of-N expert pool collection")
    sel.add_argument("--save-path", required=True)
    sel.add_argument("--gamma", type=float, default=1.0)
    sel.add_argument("--trace-split", default="train")
    sel.add_argument("--trace-dir", default=None)
    sel.add_argument("--qoe-profile", default="pensieve")
    sel.add_argument("--max-traces", type=int, default=0, help="Limit traces for small-scale runs (0=all)")
    sel.add_argument("--expert-policies", default="bola,mpc,bba", help="Comma-separated policy names")
    sel.add_argument("--bba-reservoir", type=float, default=5.0)
    sel.add_argument("--bba-cushion", type=float, default=15.0)
    sel.add_argument("--mpc-safety-factor", type=float, default=0.9)
    sel.add_argument("--selection-unit", default="window", choices=["trace", "window"])
    sel.add_argument("--selection-window", type=int, default=20)
    sel.add_argument("--selection-stride", type=int, default=5)
    sel.add_argument(
        "--selection-min-return",
        type=float,
        default=None,
        help="Drop selected trace/window samples whose local raw QoE return is below this threshold.",
    )
    sel.add_argument(
        "--negative-return-weight",
        type=float,
        default=1.0,
        help="Training weight assigned to kept selected windows with negative local raw QoE.",
    )
    sel.add_argument(
        "--positive-return-weight",
        type=float,
        default=1.0,
        help="Training weight assigned to kept selected windows with non-negative local raw QoE.",
    )
    sel.add_argument("--seed", type=int, default=20260507)
    sel.add_argument("--progress-interval", type=int, default=5)

    args = parser.parse_args()

    if args.mode == "selected":
        from trace_utils import sample_trace_files

        probe_env = ABREnv(
            trace_split=args.trace_split,
            trace_dir=args.trace_dir,
            random_start=False,
            qoe_profile=args.qoe_profile,
        )
        all_traces = sorted(probe_env.trace_files)
        max_traces = int(args.max_traces)
        if max_traces > 0 and len(all_traces) > max_traces:
            all_traces = sample_trace_files(
                all_traces, episodes=max_traces, mode="stratified", seed=int(args.seed),
            )
        policies = [p.strip() for p in str(args.expert_policies).split(",") if p.strip()]
        collect_selected_expert_trajectories(
            trace_files=all_traces,
            save_path=args.save_path,
            gamma=args.gamma,
            trace_split=args.trace_split,
            trace_dir=args.trace_dir,
            qoe_profile=args.qoe_profile,
            expert_policies=policies,
            bba_reservoir=args.bba_reservoir,
            bba_cushion=args.bba_cushion,
            mpc_safety_factor=args.mpc_safety_factor,
            selection_unit=args.selection_unit,
            selection_window=args.selection_window,
            selection_stride=args.selection_stride,
            selection_min_return=args.selection_min_return,
            negative_return_weight=args.negative_return_weight,
            positive_return_weight=args.positive_return_weight,
            progress_interval=args.progress_interval,
            seed=args.seed,
        )
    else:
        # Original collection mode (default)
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
            bba_ratio=args.bba_ratio,
            bba_reservoir=args.bba_reservoir,
            bba_cushion=args.bba_cushion,
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
            workers=args.workers,
        )
