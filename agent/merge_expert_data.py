import argparse
import json
from pathlib import Path

import numpy as np


def load_metadata(raw):
    if "metadata" not in raw.files:
        return {}
    metadata_obj = raw["metadata"]
    if metadata_obj.shape == ():
        return dict(metadata_obj.item())
    return {}


def merge_expert_datasets(input_paths, output_path, stats_path=None):
    trajectories = []
    sources = []
    metadata_by_source = {}

    for input_path in input_paths:
        input_path = Path(input_path)
        raw = np.load(input_path, allow_pickle=True)
        source_trajectories = list(raw["trajectories"])
        metadata = load_metadata(raw)
        source_name = input_path.name
        metadata_by_source[source_name] = metadata

        for trajectory in source_trajectories:
            item = dict(trajectory)
            item["source_dataset"] = source_name
            trajectories.append(item)
        sources.append({
            "path": str(input_path),
            "num_trajectories": len(source_trajectories),
            "metadata": metadata,
        })

    if not trajectories:
        raise RuntimeError("No trajectories found in input datasets.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_metadata = {
        "expert_policy": "merged",
        "sources": sources,
        "num_trajectories": len(trajectories),
    }
    np.savez_compressed(
        output_path,
        trajectories=np.asarray(trajectories, dtype=object),
        metadata=np.asarray(merged_metadata, dtype=object),
    )

    actions = np.concatenate([
        np.asarray(trajectory["actions"], dtype=np.int64)
        for trajectory in trajectories
    ])
    rewards = np.concatenate([
        np.asarray(trajectory["rewards"], dtype=np.float32)
        for trajectory in trajectories
    ])
    init_returns = [
        float(np.asarray(trajectory["returns"], dtype=np.float32)[0])
        for trajectory in trajectories
        if len(trajectory["returns"]) > 0
    ]
    action_dim = int(actions.max()) + 1 if actions.size else 0
    action_counts = np.bincount(actions, minlength=max(action_dim, 6))
    stats = {
        **merged_metadata,
        "save_path": str(output_path),
        "num_samples": int(actions.size),
        "action_counts": {
            str(idx): int(action_counts[idx])
            for idx in range(len(action_counts))
        },
        "reward_min": float(rewards.min()),
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max()),
        "return_min": float(min(init_returns)) if init_returns else 0.0,
        "return_mean": float(np.mean(init_returns)) if init_returns else 0.0,
        "return_max": float(max(init_returns)) if init_returns else 0.0,
    }

    stats_path = Path(stats_path) if stats_path else output_path.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[MergeExpertData] merged {len(input_paths)} files, "
        f"{len(trajectories)} trajectories, {actions.size} samples -> {output_path}"
    )
    print(
        "[MergeExpertData] action_counts="
        + ", ".join(f"{idx}:{int(count)}" for idx, count in enumerate(action_counts))
    )
    if len(action_counts) >= 6 and int(action_counts[5]) == 0:
        print(
            "[MergeExpertData] warning: action 5 has zero expert samples. "
            "Increase high-bandwidth BBA/MPC collection before a final run."
        )
    print(f"[MergeExpertData] saved stats -> {stats_path}")
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Merge expert trajectory NPZ files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    merge_expert_datasets(
        input_paths=args.inputs,
        output_path=args.output,
        stats_path=args.stats_path,
    )


if __name__ == "__main__":
    main()
