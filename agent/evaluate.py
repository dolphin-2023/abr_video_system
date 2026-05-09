import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from abr_env import ABREnv
from data_collector import bola_expert_policy
from checkpoint_utils import load_netllm_checkpoint, resolve_checkpoint_path
from return_utils import load_return_to_go_processor
from settings import (
    ACTION_DIM,
    BASE_MODEL_PATH,
    M_IN_K,
    RL_MODEL_PATH,
    SFT_MODEL_PATH,
)
from trace_utils import sample_trace_files


def harmonic_throughput_kbps(state_matrix):
    history = np.asarray(state_matrix[2], dtype=np.float32) * M_IN_K
    history = history[history > 1.0]
    if history.size == 0:
        return 0.0
    return float(history.size / np.sum(1.0 / np.maximum(history, 1.0)))


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reset(self, env):
        pass

    def act(self, state, env):
        return self.rng.randrange(env.num_bitrates)


class BolaPolicy:
    name = "bola"

    def reset(self, env):
        pass

    def act(self, state, env):
        return bola_expert_policy(state, env.bitrates_kbps)


class MpcPolicy:
    name = "mpc"

    def __init__(self, safety_factor=0.9):
        self.safety_factor = float(safety_factor)
        self.last_action = 0

    def reset(self, env):
        self.last_action = 0

    def act(self, state, env):
        # 用历史吞吐估计下一块下载时间，再枚举动作选择最高 QoE。
        est_kbps = harmonic_throughput_kbps(state) * self.safety_factor
        if est_kbps <= 1.0:
            return 0

        buffer_level = float(state[1, -1]) * 10.0
        best_score = -float("inf")
        best_action = 0
        for action, bitrate in enumerate(env.bitrates_kbps):
            chunk_kbit = env.estimate_chunk_size_kbit(action, env.current_segment)
            download_time = chunk_kbit / max(10.0, est_kbps)
            stall_time = max(0.0, download_time - buffer_level)
            quality_score = env.compute_quality_score(float(bitrate))
            smooth_penalty = env.compute_smooth_penalty(
                float(bitrate),
                env.bitrates_kbps[self.last_action],
            )
            score = quality_score - env.penalty_stall * stall_time - env.penalty_smooth * smooth_penalty
            if score > best_score:
                best_score = score
                best_action = action

        if abs(best_action - self.last_action) > 1:
            best_action = self.last_action + (1 if best_action > self.last_action else -1)
        self.last_action = max(0, min(env.num_bitrates - 1, best_action))
        return self.last_action


class NetLLMSFTPolicy:
    name = "netllm-sft"

    def __init__(
        self,
        model_path=SFT_MODEL_PATH,
        device=None,
        policy_name="netllm-sft",
        stats_path=None,
        base_model_path=BASE_MODEL_PATH,
    ):
        import torch
        from network import NetLLMABR

        self.name = policy_name
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rtg = load_return_to_go_processor(stats_path=stats_path)
        self.target_return = self.rtg.target_return
        self.model = NetLLMABR(
            model_name_or_path=str(base_model_path),
            action_dim=ACTION_DIM,
            lora_rank=128,
        )
        model_path = Path(model_path)
        resolved_model_path = resolve_checkpoint_path(model_path)
        if not resolved_model_path.exists():
            raise FileNotFoundError(f"SFT model not found: {model_path}")
        loaded_path = load_netllm_checkpoint(
            self.model,
            resolved_model_path,
            map_location=self.device,
            strict=False,
        )
        print(f"[Evaluate] loaded {self.name} weights: {loaded_path}")
        self.model.to(self.device)
        self.model.eval()
        self.history = None
        self.step = 0
        self.last_action = 0
        self.current_target_return = self.target_return

    def reset(self, env):
        self.history = self.model.new_history()
        self.step = 0
        self.last_action = 0
        self.current_target_return = self.target_return

    def act(self, state, env):
        state_tensor = self.torch.from_numpy(state).to(
            device=self.device,
            dtype=self.torch.float32,
            non_blocking=True,
        ).unsqueeze(0).unsqueeze(0)

        with self.torch.no_grad():
            logits, return_emb, state_emb, time_emb = self.model.predict_logits(
                state=state_tensor,
                target_return=self.current_target_return,
                timestep=self.step,
                history=self.history,
            )
            action, _ = self.model.select_action(logits, deterministic=True)

        if abs(action - self.last_action) > 1:
            action = self.last_action + (1 if action > self.last_action else -1)
        action = max(0, min(env.num_bitrates - 1, int(action)))
        self.model.append_history(
            history=self.history,
            return_emb=return_emb,
            state_emb=state_emb,
            time_emb=time_emb,
            action=action,
            device=self.device,
        )
        self.last_action = action
        self.step += 1
        return action

    def observe_reward(self, reward, info=None):
        self.current_target_return = self.rtg.update(
            self.current_target_return,
            reward,
        )


def make_policy(
    name,
    seed=0,
    device=None,
    sft_model_path=SFT_MODEL_PATH,
    rl_model_path=RL_MODEL_PATH,
    stats_path=None,
    base_model_path=BASE_MODEL_PATH,
):
    normalized = name.lower()
    if normalized == "random":
        return RandomPolicy(seed=seed)
    if normalized == "bola":
        return BolaPolicy()
    if normalized == "mpc":
        return MpcPolicy()
    if normalized in {"netllm-sft", "sft"}:
        return NetLLMSFTPolicy(
            model_path=sft_model_path,
            device=device,
            policy_name="netllm-sft",
            stats_path=stats_path,
            base_model_path=base_model_path,
        )
    if normalized in {"netllm-rl", "rl"}:
        return NetLLMSFTPolicy(
            model_path=rl_model_path,
            device=device,
            policy_name="netllm-rl",
            stats_path=stats_path,
            base_model_path=base_model_path,
        )
    raise ValueError(f"Unknown policy: {name}")


def evaluate_policy(
    policy,
    trace_files,
    trace_split,
    qoe_profile,
    progress_interval=10,
    trace_dir=None,
    env_verbose=False,
):
    episodes = []
    total_chunks = 0
    total_bitrate = 0.0
    total_stall = 0.0
    total_smooth = 0.0
    total_duration = 0.0
    rebuffer_chunks = 0

    for trace_file in trace_files:
        env = ABREnv(
            trace_split=trace_split,
            trace_dir=trace_dir,
            trace_file=trace_file,
            random_start=False,
            qoe_profile=qoe_profile,
            verbose=env_verbose,
        )
        state, _ = env.reset()
        policy.reset(env)

        done = False
        prev_action = None
        ep_qoe = 0.0
        ep_bitrate = 0.0
        ep_stall = 0.0
        ep_smooth = 0.0
        ep_switches = 0
        ep_steps = 0

        while not done:
            action = int(policy.act(state, env))
            next_state, reward, done, _, info = env.step(action)
            if hasattr(policy, "observe_reward"):
                policy.observe_reward(reward, info)

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
        })
        if progress_interval and len(episodes) % int(progress_interval) == 0:
            recent_qoes = [item["qoe"] for item in episodes]
            print(
                f"[Evaluate] policy={policy.name} "
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
        "policy": policy.name,
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
    }
    return summary, episodes


def print_table(results):
    header = (
        "policy",
        "mean_qoe",
        "median_qoe",
        "p10_qoe",
        "worst_qoe",
        "best_qoe",
        "mean_bitrate",
        "mean_stall_time",
        "mean_smoothness_penalty",
        "rebuffer_ratio",
        "quality_switch_count",
    )
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for item in results:
        print(
            f"{item['policy']} | "
            f"{item['mean_qoe']:.3f} | "
            f"{item['median_qoe']:.3f} | "
            f"{item['p10_qoe']:.3f} | "
            f"{item['worst_qoe']:.3f} | "
            f"{item['best_qoe']:.3f} | "
            f"{item['mean_bitrate']:.1f} | "
            f"{item['mean_stall_time']:.3f} | "
            f"{item['mean_smoothness_penalty']:.3f} | "
            f"{item['rebuffer_ratio']:.4f} | "
            f"{item['quality_switch_count']:.2f}"
        )


def run_evaluation(
    trace_split="test",
    qoe_profile="pensieve",
    episodes=20,
    sample_mode="stratified",
    policies=None,
    seed=20260507,
    device=None,
    progress_interval=10,
    trace_dir=None,
    video_id=None,
    chunk_size_path=None,
    output=Path("models") / "evaluation_summary.json",
    sft_model_path=SFT_MODEL_PATH,
    rl_model_path=RL_MODEL_PATH,
    stats_path=None,
    base_model_path=BASE_MODEL_PATH,
    env_verbose=False,
):
    if video_id:
        os.environ["ABR_VIDEO_ID"] = str(video_id)
    if chunk_size_path:
        os.environ["ABR_CHUNK_SIZE_PATH"] = str(chunk_size_path)

    probe_env = ABREnv(
        trace_split=trace_split,
        trace_dir=trace_dir,
        random_start=False,
        qoe_profile=qoe_profile,
        verbose=env_verbose,
    )
    trace_files = sorted(probe_env.trace_files)
    trace_files = sample_trace_files(
        trace_files,
        episodes=episodes,
        mode=sample_mode,
        seed=seed,
    )
    if not trace_files:
        raise RuntimeError(f"No trace files found for split={trace_split}")

    policies = list(policies or ["bola", "mpc", "random"])
    if "all" in {item.lower() for item in policies}:
        policies = ["bola", "mpc", "random", "netllm-sft", "netllm-rl"]

    summaries = []
    details = {}
    for idx, policy_name in enumerate(policies):
        print(
            f"[Evaluate] policy={policy_name} traces={len(trace_files)} "
            f"split={trace_split} qoe={qoe_profile} sample={sample_mode}"
        )
        policy = make_policy(
            policy_name,
            seed=seed + idx,
            device=device,
            sft_model_path=sft_model_path,
            rl_model_path=rl_model_path,
            stats_path=stats_path,
            base_model_path=base_model_path,
        )
        summary, episodes_detail = evaluate_policy(
            policy,
            trace_files,
            trace_split,
            qoe_profile,
            progress_interval=progress_interval,
            trace_dir=trace_dir,
            env_verbose=env_verbose,
        )
        summaries.append(summary)
        details[summary["policy"]] = episodes_detail

    payload = {
        "trace_split": trace_split,
        "trace_dir": trace_dir,
        "qoe_profile": qoe_profile,
        "sample_mode": sample_mode,
        "seed": seed,
        "trace_count": len(trace_files),
        "policies": summaries,
        "episodes": details,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_table(summaries)
    print(f"[Evaluate] saved -> {output}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ABR policies on a fixed trace split.")
    parser.add_argument("--trace-split", default="test")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument(
        "--qoe-profile",
        default="pensieve",
        choices=["pensieve", "log", "linear3"],
        help="Main evaluation reward. Use log as auxiliary sensitivity analysis.",
    )
    parser.add_argument("--episodes", type=int, default=20, help="0 means all traces in the split.")
    parser.add_argument(
        "--sample-mode",
        default="stratified",
        choices=["first", "random", "stratified"],
        help="How to choose traces when --episodes is smaller than the split.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["bola", "mpc", "random"],
        help="Available: bola mpc random netllm-sft netllm-rl all",
    )
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print per-policy progress every N episodes. Use 0 to disable.",
    )
    parser.add_argument("--video-id", default=None, help="Optional DASH video id for real chunk sizes.")
    parser.add_argument("--chunk-size-path", default=None, help="Optional precomputed chunk_sizes.json.")
    parser.add_argument("--sft-model-path", default=SFT_MODEL_PATH)
    parser.add_argument("--rl-model-path", default=RL_MODEL_PATH)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--env-verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("models") / "evaluation_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    run_evaluation(
        trace_split=args.trace_split,
        qoe_profile=args.qoe_profile,
        episodes=args.episodes,
        sample_mode=args.sample_mode,
        policies=args.policies,
        seed=args.seed,
        device=args.device,
        progress_interval=args.progress_interval,
        trace_dir=args.trace_dir,
        video_id=args.video_id,
        chunk_size_path=args.chunk_size_path,
        output=args.output,
        sft_model_path=args.sft_model_path,
        rl_model_path=args.rl_model_path,
        stats_path=args.stats_path,
        base_model_path=args.base_model_path,
        env_verbose=args.env_verbose,
    )


if __name__ == "__main__":
    main()
