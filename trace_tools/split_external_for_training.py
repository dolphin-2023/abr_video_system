#!/usr/bin/env python3
"""Build deterministic external_mix train, validation, and test splits.

The generated directories live under simulator/traces so ABREnv can resolve
them directly by trace_split name.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


DEFAULT_COUNTS = {
    "train": {"puffer": 240, "weak": 150},
    "valid": {"puffer": 45, "weak": 35},
}


def mean_kbps(path: Path) -> float:
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                values.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def stratified_order(files: list[Path], seed: int) -> list[Path]:
    rng = random.Random(seed)
    ranked = sorted(files, key=lambda item: (mean_kbps(item), item.name))
    bins: list[list[Path]] = [[] for _ in range(10)]
    for idx, item in enumerate(ranked):
        bucket = min(9, idx * 10 // max(1, len(ranked)))
        bins[bucket].append(item)
    for bucket in bins:
        rng.shuffle(bucket)

    ordered: list[Path] = []
    while any(bins):
        for bucket in bins:
            if bucket:
                ordered.append(bucket.pop())
    return ordered


def take(source: list[Path], count: int, label: str) -> tuple[list[Path], list[Path]]:
    if count > len(source):
        raise ValueError(f"not enough {label} traces: need {count}, have {len(source)}")
    return source[:count], source[count:]


def prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(files: list[Path], output_dir: Path) -> list[str]:
    copied = []
    for src in files:
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied.append(src.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic external_mix trace splits.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--train-puffer", type=int, default=DEFAULT_COUNTS["train"]["puffer"])
    parser.add_argument("--train-weak", type=int, default=DEFAULT_COUNTS["train"]["weak"])
    parser.add_argument("--valid-puffer", type=int, default=DEFAULT_COUNTS["valid"]["puffer"])
    parser.add_argument("--valid-weak", type=int, default=DEFAULT_COUNTS["valid"]["weak"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    trace_root = args.project_root / "simulator" / "traces"
    puffer_dir = trace_root / "external_puffer_recent"
    weak_dir = trace_root / "external_weak_mobile"

    puffer = stratified_order(sorted(puffer_dir.glob("*.txt")), args.seed)
    weak = stratified_order(sorted(weak_dir.glob("*.txt")), args.seed + 1)

    train_puffer, puffer = take(puffer, args.train_puffer, "puffer train")
    valid_puffer, puffer = take(puffer, args.valid_puffer, "puffer valid")
    train_weak, weak = take(weak, args.train_weak, "weak train")
    valid_weak, weak = take(weak, args.valid_weak, "weak valid")

    splits = {
        "external_mix_train": train_puffer + train_weak,
        "external_mix_valid": valid_puffer + valid_weak,
        "external_mix_test": puffer + weak,
    }

    manifest = {
        "seed": args.seed,
        "source_counts": {
            "external_puffer_recent": len(train_puffer) + len(valid_puffer) + len(puffer),
            "external_weak_mobile": len(train_weak) + len(valid_weak) + len(weak),
        },
        "splits": {},
    }

    for split_name, files in splits.items():
        output_dir = trace_root / split_name
        prepare_dir(output_dir, args.overwrite)
        copied = copy_split(files, output_dir)
        puffer_count = sum(1 for item in files if item.parent == puffer_dir)
        weak_count = sum(1 for item in files if item.parent == weak_dir)
        manifest["splits"][split_name] = {
            "count": len(files),
            "puffer_count": puffer_count,
            "weak_mobile_count": weak_count,
            "puffer_ratio": round(puffer_count / max(1, len(files)), 4),
            "files": copied,
        }

    manifest_path = trace_root / "external_mix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for split_name, info in manifest["splits"].items():
        print(
            f"{split_name}: {info['count']} traces "
            f"(puffer={info['puffer_count']}, weak={info['weak_mobile_count']}, "
            f"puffer_ratio={info['puffer_ratio']})"
        )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
