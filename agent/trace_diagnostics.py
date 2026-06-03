import argparse
import json
import os
from pathlib import Path

import numpy as np

from abr_env import ABREnv
from evaluate import make_policy
from settings import (
    BASE_MODEL_PATH,
    OFFLINE_RL_MODEL_PATH,
    PENSIEVE_MODEL_PATH,
    RL_MODEL_PATH,
    SFT_MODEL_PATH,
)


def load_eval_trace_names(path, indices):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    trace_names = payload.get("trace_files", [])
    selected = []
    for idx in indices:
        if idx < 0 or idx >= len(trace_names):
            raise IndexError(f"trace index out of range: {idx}")
        selected.append(trace_names[idx])
    return selected


def map_trace_paths(trace_split, trace_dir, qoe_profile):
    probe = ABREnv(
        trace_split=trace_split,
        trace_dir=trace_dir,
        random_start=False,
        qoe_profile=qoe_profile,
    )
    return {Path(path).name: path for path in probe.trace_files}


def run_episode(policy, trace_file, trace_split, trace_dir, qoe_profile):
    env = ABREnv(
        trace_split=trace_split,
        trace_dir=trace_dir,
        trace_file=trace_file,
        random_start=False,
        qoe_profile=qoe_profile,
    )
    state, _ = env.reset()
    policy.reset(env)

    done = False
    step = 0
    prev_action = None
    total_qoe = 0.0
    total_stall = 0.0
    total_smooth = 0.0
    total_bitrate = 0.0
    switches = 0
    rows = []

    while not done:
        buffer_before = float(env.current_buffer)
        target_return_before = getattr(policy, "current_target_return", None)
        action = int(policy.act(state, env))
        next_state, reward, done, _, info = env.step(action)
        if hasattr(policy, "observe_reward"):
            policy.observe_reward(reward, info)
        target_return_after = getattr(policy, "current_target_return", None)

        if prev_action is not None and action != prev_action:
            switches += 1

        total_qoe += float(reward)
        total_stall += float(info["stall_time"])
        total_smooth += float(info["smoothness_penalty"])
        total_bitrate += float(info["bitrate_kbps"])

        rows.append({
            "step": step,
            "action": action,
            "bitrate_kbps": float(info["bitrate_kbps"]),
            "buffer_before": buffer_before,
            "buffer_after": float(env.current_buffer),
            "stall_time": float(info["stall_time"]),
            "reward": float(reward),
            "quality_score": float(info["quality_score"]),
            "smoothness_penalty": float(info["smoothness_penalty"]),
            "throughput_kbps": float(info["throughput_kbps"]),
            "measured_throughput_kbps": float(info["measured_throughput_kbps"]),
            "download_time": float(info["download_time"]),
            "target_return_before": (
                float(target_return_before) if target_return_before is not None else None
            ),
            "target_return_after": (
                float(target_return_after) if target_return_after is not None else None
            ),
        })

        prev_action = action
        state = next_state
        step += 1

    return {
        "trace": Path(trace_file).name,
        "qoe": total_qoe,
        "mean_bitrate": total_bitrate / max(step, 1),
        "stall_time": total_stall,
        "smoothness_penalty": total_smooth,
        "quality_switch_count": switches,
        "chunks": step,
        "action_counts": {
            str(idx): int(count)
            for idx, count in enumerate(np.bincount(
                [row["action"] for row in rows],
                minlength=env.num_bitrates,
            ))
        },
        "steps": rows,
    }


def worst_stall_steps(episode, limit=6):
    rows = sorted(
        episode["steps"],
        key=lambda item: (item["stall_time"], -item["reward"]),
        reverse=True,
    )
    return rows[:limit]


def main():
    parser = argparse.ArgumentParser(
        description="Rerun selected traces and save per-chunk ABR diagnostics."
    )
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--trace-indices", nargs="+", type=int, required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["bola", "mpc", "netllm-sft", "netllm-offline-rl"],
    )
    parser.add_argument("--trace-split", default="test")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--chunk-size-path", default=None)
    parser.add_argument("--qoe-profile", default="pensieve")
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--target-return", type=float, default=None)
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--sft-model-path", default=SFT_MODEL_PATH)
    parser.add_argument("--offline-rl-model-path", default=OFFLINE_RL_MODEL_PATH)
    parser.add_argument("--rl-model-path", default=RL_MODEL_PATH)
    parser.add_argument("--pensieve-model-path", default=PENSIEVE_MODEL_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("agent") / "runs" / "diagnostics" / "trace_diagnostics.json",
    )
    args = parser.parse_args()

    if args.video_id:
        os.environ["ABR_VIDEO_ID"] = str(args.video_id)
    if args.chunk_size_path:
        os.environ["ABR_CHUNK_SIZE_PATH"] = str(Path(args.chunk_size_path).resolve())

    trace_names = load_eval_trace_names(args.evaluation_json, args.trace_indices)
    trace_paths = map_trace_paths(args.trace_split, args.trace_dir, args.qoe_profile)
    missing = [name for name in trace_names if name not in trace_paths]
    if missing:
        raise FileNotFoundError(f"trace files not found in split: {missing}")

    payload = {
        "evaluation_json": str(args.evaluation_json),
        "trace_indices": args.trace_indices,
        "trace_files": trace_names,
        "policies": {},
    }

    for policy_idx, policy_name in enumerate(args.policies):
        print(f"[TraceDiag] policy={policy_name}")
        policy = make_policy(
            policy_name,
            seed=args.seed + policy_idx,
            device=args.device,
            sft_model_path=args.sft_model_path,
            offline_rl_model_path=args.offline_rl_model_path,
            rl_model_path=args.rl_model_path,
            pensieve_model_path=args.pensieve_model_path,
            stats_path=args.stats_path,
            base_model_path=args.base_model_path,
        )
        if args.target_return is not None and hasattr(policy, "target_return"):
            policy.target_return = float(args.target_return)
        episodes = []
        for trace_name in trace_names:
            episode = run_episode(
                policy=policy,
                trace_file=trace_paths[trace_name],
                trace_split=args.trace_split,
                trace_dir=args.trace_dir,
                qoe_profile=args.qoe_profile,
            )
            episodes.append(episode)
            worst = worst_stall_steps(episode, limit=3)
            worst_text = ", ".join(
                f"t{row['step']}:a{row['action']} stall={row['stall_time']:.2f}"
                for row in worst
            )
            print(
                f"  idx={args.trace_indices[len(episodes)-1]:02d} "
                f"qoe={episode['qoe']:.2f} stall={episode['stall_time']:.2f} "
                f"br={episode['mean_bitrate']:.0f} switches={episode['quality_switch_count']} "
                f"worst=({worst_text})"
            )
        payload["policies"][policy_name] = episodes

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[TraceDiag] saved -> {args.output}")


if __name__ == "__main__":
    main()
