import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from abr_env import ABREnv
from checkpoint_utils import load_netllm_checkpoint
from network import NetLLMABR
from return_utils import load_return_to_go_processor
from settings import (
    ACTION_DIM,
    BASE_MODEL_PATH,
    OFFLINE_RL_MODEL_PATH,
    RL_MODEL_PATH,
    SFT_MODEL_PATH,
    TRAINING_STATS_PATH,
)
from trace_utils import sample_trace_files
from training_eval import constrained_action


CHECKPOINT_ALIASES = {
    "sft": SFT_MODEL_PATH,
    "netllm-sft": SFT_MODEL_PATH,
    "offline_rl": OFFLINE_RL_MODEL_PATH,
    "offline-rl": OFFLINE_RL_MODEL_PATH,
    "netllm-offline-rl": OFFLINE_RL_MODEL_PATH,
    "rl": RL_MODEL_PATH,
    "netllm-rl": RL_MODEL_PATH,
}


def parse_csv_floats(value):
    items = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw:
            items.append(float(raw))
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return items


def resolve_checkpoint(value):
    key = str(value).strip().lower()
    if key in CHECKPOINT_ALIASES:
        return CHECKPOINT_ALIASES[key]
    return Path(value)


def resolve_device(requested, allow_cpu):
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available, and loading the local LLM on CPU can be very slow. "
            "Pass --allow-cpu if you intentionally want to run this diagnostic on CPU."
        )
    return device


def entropy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))


def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(q, 1e-12)))))


def expected_action(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return float(np.sum(probs * np.arange(probs.size, dtype=np.float64)))


def action_distribution(actions):
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=ACTION_DIM)
    total = max(int(counts.sum()), 1)
    return {str(idx): float(counts[idx] / total) for idx in range(len(counts))}


def build_trace_files(trace_split, trace_dir, episodes, sample_mode, seed, qoe_profile):
    probe_env = ABREnv(
        trace_split=trace_split,
        trace_dir=trace_dir,
        random_start=False,
        qoe_profile=qoe_profile,
    )
    trace_files = sorted(probe_env.trace_files)
    return sample_trace_files(
        trace_files,
        episodes=int(episodes),
        mode=sample_mode,
        seed=int(seed),
    )


def load_model(checkpoint_path, base_model_path, device):
    model = NetLLMABR(
        model_name_or_path=str(base_model_path),
        action_dim=ACTION_DIM,
        lora_rank=128,
    )
    loaded_path = load_netllm_checkpoint(
        model,
        checkpoint_path,
        map_location=device,
        strict=False,
    )
    if loaded_path is None:
        raise FileNotFoundError(f"NetLLM checkpoint not found: {checkpoint_path}")
    model.to(device)
    model.eval()
    return model, loaded_path


@torch.no_grad()
def probe_state(model, state, history, timestep, return_values, device):
    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).unsqueeze(0)

    outputs = []
    for target_return in return_values:
        logits, return_emb, state_emb, time_emb = model.predict_logits(
            state=state_tensor,
            target_return=float(target_return),
            timestep=int(timestep),
            history=history,
        )
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits_np = logits.detach().cpu().numpy()
        outputs.append({
            "target_return": float(target_return),
            "logits": logits_np.tolist(),
            "probabilities": probs.tolist(),
            "action": int(np.argmax(probs)),
            "expected_action": expected_action(probs),
            "entropy": entropy(probs),
            "_return_emb": return_emb,
            "_state_emb": state_emb,
            "_time_emb": time_emb,
        })
    return outputs


def strip_tensors(items):
    clean = []
    for item in items:
        clean.append({
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        })
    return clean


def summarize_records(records, base_index):
    if not records:
        return {}

    by_multiplier = defaultdict(list)
    base_actions = []
    unique_action_counts = []
    for record in records:
        probes = record["probes"]
        base = probes[base_index]
        base_probs = np.asarray(base["probabilities"], dtype=np.float64)
        base_logits = np.asarray(base["logits"], dtype=np.float64)
        base_actions.append(int(base["action"]))
        unique_action_counts.append(len({int(item["action"]) for item in probes}))

        for idx, item in enumerate(probes):
            probs = np.asarray(item["probabilities"], dtype=np.float64)
            logits = np.asarray(item["logits"], dtype=np.float64)
            by_multiplier[idx].append({
                "target_return": float(item["target_return"]),
                "action": int(item["action"]),
                "expected_action": float(item["expected_action"]),
                "entropy": float(item["entropy"]),
                "changed": int(item["action"] != base["action"]),
                "kl_to_base": kl_divergence(probs, base_probs),
                "max_abs_logit_delta": float(np.max(np.abs(logits - base_logits))),
            })

    per_return = {}
    for idx, values in by_multiplier.items():
        per_return[str(idx)] = {
            "mean_target_return": float(np.mean([item["target_return"] for item in values])),
            "action_change_rate": float(np.mean([item["changed"] for item in values])),
            "mean_expected_action": float(np.mean([item["expected_action"] for item in values])),
            "mean_entropy": float(np.mean([item["entropy"] for item in values])),
            "mean_kl_to_base": float(np.mean([item["kl_to_base"] for item in values])),
            "mean_max_abs_logit_delta": float(
                np.mean([item["max_abs_logit_delta"] for item in values])
            ),
        }

    return {
        "states_probed": len(records),
        "base_action_distribution": action_distribution(base_actions),
        "distinct_action_rate": float(np.mean([count > 1 for count in unique_action_counts])),
        "mean_unique_actions_per_state": float(np.mean(unique_action_counts)),
        "per_return_index": per_return,
    }


def run_diagnostic(args):
    device = resolve_device(args.device, args.allow_cpu)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    stats_path = Path(args.stats_path)
    rtg = load_return_to_go_processor(stats_path=stats_path)
    trace_files = build_trace_files(
        trace_split=args.trace_split,
        trace_dir=args.trace_dir,
        episodes=args.episodes,
        sample_mode=args.sample_mode,
        seed=args.seed,
        qoe_profile=args.qoe_profile,
    )
    if not trace_files:
        raise RuntimeError(f"No trace files found for split={args.trace_split}")

    model, loaded_path = load_model(
        checkpoint_path=checkpoint_path,
        base_model_path=args.base_model_path,
        device=device,
    )

    multipliers = list(args.return_multipliers)
    base_index = min(range(len(multipliers)), key=lambda idx: abs(multipliers[idx] - 1.0))
    records = []

    for trace_file in trace_files:
        env = ABREnv(
            trace_split=args.trace_split,
            trace_dir=args.trace_dir,
            trace_file=trace_file,
            random_start=False,
            qoe_profile=args.qoe_profile,
        )
        state, _ = env.reset()
        history = model.new_history()
        current_target_return = float(rtg.target_return)
        last_action = 0
        done = False

        for timestep in range(int(args.steps_per_trace)):
            if done:
                break

            center_return = float(current_target_return)
            return_values = [center_return * float(multiplier) for multiplier in multipliers]
            probes = probe_state(
                model=model,
                state=state,
                history=history,
                timestep=timestep,
                return_values=return_values,
                device=device,
            )
            base_probe = probes[base_index]
            rollout_action = constrained_action(
                base_probe["action"],
                last_action,
                env.num_bitrates,
                jump_limit=args.jump_limit,
            )

            records.append({
                "trace": Path(trace_file).name,
                "timestep": int(timestep),
                "center_target_return": center_return,
                "rollout_action": int(rollout_action),
                "probes": strip_tensors(probes),
            })

            next_state, reward, done, _, _ = env.step(rollout_action)
            current_target_return = rtg.update(current_target_return, reward)
            model.append_history(
                history=history,
                return_emb=base_probe["_return_emb"],
                state_emb=base_probe["_state_emb"],
                time_emb=base_probe["_time_emb"],
                action=rollout_action,
                device=device,
            )
            last_action = int(rollout_action)
            state = next_state

    output = Path(args.output)
    payload = {
        "checkpoint": str(checkpoint_path),
        "loaded_checkpoint": str(loaded_path),
        "base_model_path": str(args.base_model_path),
        "device": str(device),
        "stats_path": str(stats_path),
        "output": str(output),
        "rtg": rtg.describe(),
        "trace_split": args.trace_split,
        "trace_dir": args.trace_dir,
        "qoe_profile": args.qoe_profile,
        "episodes": int(args.episodes),
        "steps_per_trace": int(args.steps_per_trace),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "return_multipliers": multipliers,
        "base_return_index": int(base_index),
        "summary": summarize_records(records, base_index=base_index),
        "records": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def print_summary(payload):
    summary = payload.get("summary", {})
    print(f"[RTG] checkpoint={payload['loaded_checkpoint']}")
    print(f"[RTG] {payload['rtg']}")
    print(
        f"[RTG] probed={summary.get('states_probed', 0)} "
        f"distinct_action_rate={summary.get('distinct_action_rate', 0.0):.3f} "
        f"mean_unique_actions={summary.get('mean_unique_actions_per_state', 0.0):.3f}"
    )
    print("[RTG] base_action_distribution=" + json.dumps(
        summary.get("base_action_distribution", {}),
        ensure_ascii=False,
    ))
    for idx, item in summary.get("per_return_index", {}).items():
        print(
            f"[RTG] return_index={idx} "
            f"mean_target={item['mean_target_return']:.6f} "
            f"change_rate={item['action_change_rate']:.3f} "
            f"expected_action={item['mean_expected_action']:.3f} "
            f"kl={item['mean_kl_to_base']:.6f} "
            f"logit_delta={item['mean_max_abs_logit_delta']:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe whether a NetLLM ABR checkpoint changes logits/actions when RTG changes."
    )
    parser.add_argument(
        "--checkpoint",
        default="sft",
        help="Checkpoint alias/path: sft, offline_rl, rl, or a checkpoint directory.",
    )
    parser.add_argument("--base-model-path", type=Path, default=BASE_MODEL_PATH)
    parser.add_argument("--stats-path", type=Path, default=TRAINING_STATS_PATH)
    parser.add_argument("--trace-split", default="test")
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--qoe-profile", default="pensieve")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps-per-trace", type=int, default=8)
    parser.add_argument("--sample-mode", default="stratified", choices=["first", "random", "stratified"])
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--jump-limit", type=int, default=1)
    parser.add_argument(
        "--return-multipliers",
        type=parse_csv_floats,
        default=parse_csv_floats("0.25,0.5,1.0,1.5,2.0"),
        help="Comma-separated multipliers around the current remaining target return.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow loading the LLM on CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("agent") / "runs" / "diagnostics" / "rtg_sensitivity.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run_diagnostic(args)
    print_summary(payload)
    print(f"[RTG] saved -> {args.output}")
