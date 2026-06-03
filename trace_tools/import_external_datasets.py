import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_PUFFER_DIR = DEFAULT_SOURCE_ROOT / "Puffer"
DEFAULT_PERFORM_CSV = (
    DEFAULT_SOURCE_ROOT
    / "A Large-Scale Dataset of 4G, NB-IoT, and 5G Non-Standalone Network Measurements"
    / "Throughput Tests - Speedtest - Active Measurements.csv"
)
DEFAULT_LTE_ZIP = (
    DEFAULT_SOURCE_ROOT
    / "Beyond Throughput a 4G LTE Dataset with Channel and Context Metrics"
    / "LTE_Dataset.zip"
)
DEFAULT_PUFFER_OUTPUT = PROJECT_ROOT / "simulator" / "traces" / "external_puffer_recent"
DEFAULT_WEAK_OUTPUT = PROJECT_ROOT / "simulator" / "traces" / "external_weak_mobile"


def manifest_path(path):
    """Return a portable path for generated manifests when possible."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def safe_float(value):
    text = str(value or "").strip().replace(",", "")
    if not text or text == "?":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def trace_digest(values):
    text = "\n".join(f"{float(value):.2f}" for value in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clamp_kbps(values):
    return [
        float(min(150_000.0, max(10.0, value)))
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def normalize_kbps(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return []
    non_zero = [value for value in values if value > 0.0]
    reference = statistics.median(non_zero or values)
    if reference > 100_000.0:
        values = [value / 1000.0 for value in values]
    return clamp_kbps(values)


def resample_to_1hz(points, unit="kbps"):
    if not points:
        return []
    points = sorted((float(time_value), float(value)) for time_value, value in points if value is not None)
    if not points:
        return []
    base = points[0][0]
    points = [(time_value - base, value) for time_value, value in points]
    duration = max(1, int(math.ceil(points[-1][0])))
    buckets = [[] for _ in range(duration + 1)]
    for time_value, value in points:
        idx = int(max(0, min(duration, math.floor(time_value))))
        buckets[idx].append(value)

    output = []
    last_value = points[0][1]
    for bucket in buckets:
        if bucket:
            last_value = float(np.mean(bucket))
        output.append(last_value)

    if unit == "bytes_per_second":
        return clamp_kbps([value * 8.0 / 1000.0 for value in output])
    if unit == "bits_per_second":
        return clamp_kbps([value / 1000.0 for value in output])
    return normalize_kbps(output)


def write_trace(path, values, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {header}\n" + "\n".join(f"{float(value):.2f}" for value in values) + "\n",
        encoding="utf-8",
    )


def clear_trace_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for trace_file in path.glob("*.txt"):
        trace_file.unlink()
    manifest = path / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def summarize(values):
    array = np.asarray(values, dtype=np.float32)
    return {
        "seconds": int(len(values)),
        "mean_kbps": float(np.mean(array)),
        "median_kbps": float(np.median(array)),
        "p10_kbps": float(np.percentile(array, 10)),
        "p90_kbps": float(np.percentile(array, 90)),
        "digest": trace_digest(values),
    }


def parse_puffer_sent_csv(sent_csv):
    streams = defaultdict(list)
    with Path(sent_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_id = row.get("session_id")
            index = row.get("index")
            channel = row.get("channel")
            time_ns = safe_float(row.get("time (ns GMT)"))
            delivery_rate = safe_float(row.get("delivery_rate"))
            if (
                not session_id
                or index is None
                or not channel
                or time_ns is None
                or delivery_rate is None
                or delivery_rate <= 0.0
            ):
                continue
            streams[(session_id, str(index), channel)].append((time_ns / 1e9, delivery_rate))
    return {
        stream_key: resample_to_1hz(points, unit="bytes_per_second")
        for stream_key, points in streams.items()
    }


def stratified_sample(candidates, max_traces, seed, bins):
    if len(candidates) <= max_traces:
        return list(candidates)

    rng = random.Random(seed)
    buckets = {idx: [] for idx in range(len(bins) - 1)}
    overflow = []
    for candidate in candidates:
        median = candidate[3]["median_kbps"]
        placed = False
        for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            if low <= median < high or (idx == len(bins) - 2 and median <= high):
                buckets[idx].append(candidate)
                placed = True
                break
        if not placed:
            overflow.append(candidate)

    quota = max(1, max_traces // max(1, len(buckets)))
    selected = []
    remainders = []
    for bucket_items in buckets.values():
        rng.shuffle(bucket_items)
        selected.extend(bucket_items[:quota])
        remainders.extend(bucket_items[quota:])
    rng.shuffle(overflow)
    remainders.extend(overflow)
    rng.shuffle(remainders)
    selected.extend(remainders[: max(0, max_traces - len(selected))])
    return selected[:max_traces]


def build_puffer_recent(
    source_dir=DEFAULT_PUFFER_DIR,
    output_dir=DEFAULT_PUFFER_OUTPUT,
    max_traces=300,
    min_seconds=90,
    min_median_kbps=3000.0,
    max_median_kbps=25000.0,
    seed=20260509,
    clear=False,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    sent_files = sorted(source_dir.glob("video_sent_*.csv"))
    if not sent_files:
        raise FileNotFoundError(f"No video_sent_*.csv files found in {source_dir}")
    if clear:
        clear_trace_dir(output_dir)

    candidates = []
    for sent_csv in sent_files:
        for stream_key, values in parse_puffer_sent_csv(sent_csv).items():
            if len(values) < min_seconds:
                continue
            stats = summarize(values)
            if stats["median_kbps"] < min_median_kbps or stats["median_kbps"] > max_median_kbps:
                continue
            candidates.append((sent_csv, stream_key, values, stats))

    bins = [min_median_kbps, 6000.0, 9000.0, 12000.0, 16000.0, max_median_kbps]
    candidates = stratified_sample(candidates, max_traces=max_traces * 3, seed=seed, bins=bins)

    selected = []
    seen = set()
    for sent_csv, stream_key, values, stats in candidates:
        if stats["digest"] in seen:
            continue
        stream_text = "__".join(str(part) for part in stream_key)
        stream_sha1 = hashlib.sha1(stream_text.encode("utf-8")).hexdigest()
        target = output_dir / f"puffer_recent__{stream_sha1[:10]}__{stats['digest'][:10]}.txt"
        write_trace(
            target,
            values,
            "Puffer video_sent delivery_rate converted from bytes/s to Kbit/s",
        )
        seen.add(stats["digest"])
        selected.append({
            "source": manifest_path(sent_csv),
            "stream_key": {
                "session_sha1": hashlib.sha1(str(stream_key[0]).encode("utf-8")).hexdigest(),
                "index": stream_key[1],
                "channel": stream_key[2],
            },
            "target": manifest_path(target),
            **stats,
        })
        if len(selected) >= max_traces:
            break

    manifest = {
        "name": "external_puffer_recent",
        "purpose": "external_source_pool",
        "source_dir": manifest_path(source_dir),
        "output_dir": manifest_path(output_dir),
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "min_seconds": int(min_seconds),
        "min_median_kbps": float(min_median_kbps),
        "max_median_kbps": float(max_median_kbps),
        "sampling": "stratified_by_median_kbps",
        "median_bins_kbps": bins,
        "unit_strategy": "Puffer video_sent.delivery_rate is bytes/s; converted via bytes/s * 8 / 1000",
        "seed": int(seed),
        "traces": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ExternalImport] puffer_recent selected={len(selected)} candidates={len(candidates)} output={output_dir}")
    return manifest


def parse_lte_zip(lte_zip):
    traces = []
    with zipfile.ZipFile(lte_zip) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with archive.open(name) as raw:
                text = (line.decode("utf-8", errors="replace") for line in raw)
                reader = csv.DictReader(text)
                current = []
                for row in reader:
                    state = str(row.get("State") or "").strip().upper()
                    value = safe_float(row.get("DL_bitrate"))
                    if state == "D" and value is not None and value > 0.0:
                        current.append(value)
                    elif current:
                        values = normalize_kbps(current)
                        if values:
                            traces.append((name, values))
                        current = []
                if current:
                    values = normalize_kbps(current)
                    if values:
                        traces.append((name, values))
    return traces


def parse_perform_time(row):
    date_text = str(row.get("Date") or "").strip()
    time_text = str(row.get("Time") or "").strip()
    if not date_text or not time_text:
        return None
    for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_text} {time_text}", fmt).timestamp()
        except ValueError:
            continue
    return None


def perform_downlink_value(row):
    # The PERFORM active-measurement CSV reports these downlink fields in Kbps.
    for key in ("Current Netw. DL", "Mean Netw. DL", "5G PDSCH Throughput", "LTE PDSCH Throughput"):
        value = safe_float(row.get(key))
        if value is not None and value > 0.0:
            return value
    five_g = safe_float(row.get("5G PDSCH Throughput")) or 0.0
    lte = safe_float(row.get("LTE PDSCH Throughput")) or 0.0
    total = five_g + lte
    return total if total > 0.0 else None


def split_values(values, window_seconds, key):
    if len(values) <= window_seconds:
        return [(key, values)]
    traces = []
    stride = max(window_seconds // 2, 1)
    for start in range(0, len(values) - window_seconds + 1, stride):
        traces.append((key, values[start:start + window_seconds]))
    return traces


def parse_perform_csv(perform_csv, window_seconds=180, max_gap_seconds=5.0):
    groups = defaultdict(list)
    with Path(perform_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = parse_perform_time(row)
            value = perform_downlink_value(row)
            if timestamp is None or value is None:
                continue
            key = (
                row.get("Campaign") or "campaign",
                row.get("Operator") or "operator",
                row.get("Scenario") or "scenario",
                row.get("RAT Info") or "rat",
            )
            groups[key].append((timestamp, value))

    traces = []
    for key, points in groups.items():
        points = sorted(points)
        current = []
        last_time = None
        for timestamp, value in points:
            if last_time is not None and timestamp - last_time > max_gap_seconds and current:
                values = resample_to_1hz(current)
                if values:
                    traces.extend(split_values(values, int(window_seconds), key))
                current = []
            current.append((timestamp, value))
            last_time = timestamp
        values = resample_to_1hz(current)
        if values:
            traces.extend(split_values(values, int(window_seconds), key))
    return traces


def build_weak_mobile(
    perform_csv=DEFAULT_PERFORM_CSV,
    lte_zip=DEFAULT_LTE_ZIP,
    output_dir=DEFAULT_WEAK_OUTPUT,
    max_traces=300,
    min_seconds=90,
    max_median_kbps=3000.0,
    seed=20260509,
    clear=False,
):
    output_dir = Path(output_dir)
    if clear:
        clear_trace_dir(output_dir)

    candidates = []
    if Path(lte_zip).exists():
        for name, values in parse_lte_zip(lte_zip):
            if len(values) >= min_seconds:
                stats = summarize(values)
                if stats["median_kbps"] <= max_median_kbps:
                    candidates.append(("lte_dataset", name, values, stats))
    if Path(perform_csv).exists():
        for key, values in parse_perform_csv(perform_csv):
            if len(values) >= min_seconds:
                stats = summarize(values)
                if stats["median_kbps"] <= max_median_kbps:
                    candidates.append(("perform_active", "__".join(str(part) for part in key), values, stats))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: (item[3]["median_kbps"], item[3]["p90_kbps"]))

    selected = []
    seen = set()
    for source_name, source_item, values, stats in candidates:
        if stats["digest"] in seen:
            continue
        source_hash = hashlib.sha1(str(source_item).encode("utf-8")).hexdigest()[:10]
        target = output_dir / f"weak_mobile__{source_name}__{source_hash}__{stats['digest'][:10]}.txt"
        write_trace(target, values, "External weak/mobile throughput trace in Kbit/s")
        seen.add(stats["digest"])
        selected.append({
            "source": source_name,
            "source_item": str(source_item),
            "target": manifest_path(target),
            **stats,
        })
        if len(selected) >= max_traces:
            break

    manifest = {
        "name": "external_weak_mobile",
        "purpose": "external_source_pool",
        "perform_csv": manifest_path(perform_csv),
        "lte_zip": manifest_path(lte_zip),
        "output_dir": manifest_path(output_dir),
        "selected_count": len(selected),
        "candidate_count": len(candidates),
        "min_seconds": int(min_seconds),
        "max_median_kbps": float(max_median_kbps),
        "unit_strategy": {
            "lte_dataset": "DL_bitrate Kbps, State == D download rows only, continuous D segments split",
            "perform_active": "Current/Mean Netw. DL and PDSCH throughput fields treated as Kbps",
        },
        "seed": int(seed),
        "traces": selected,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ExternalImport] weak_mobile selected={len(selected)} candidates={len(candidates)} output={output_dir}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Import downloaded external ABR trace datasets.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--puffer-dir", type=Path, default=None)
    parser.add_argument("--perform-csv", type=Path, default=None)
    parser.add_argument("--lte-zip", type=Path, default=None)
    parser.add_argument("--puffer-output-dir", type=Path, default=DEFAULT_PUFFER_OUTPUT)
    parser.add_argument("--weak-output-dir", type=Path, default=DEFAULT_WEAK_OUTPUT)
    parser.add_argument("--max-traces", type=int, default=300)
    parser.add_argument("--min-seconds", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--skip-puffer", action="store_true")
    parser.add_argument("--skip-weak", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    puffer_dir = args.puffer_dir or source_root / "Puffer"
    perform_csv = args.perform_csv or (
        source_root
        / "A Large-Scale Dataset of 4G, NB-IoT, and 5G Non-Standalone Network Measurements"
        / "Throughput Tests - Speedtest - Active Measurements.csv"
    )
    lte_zip = args.lte_zip or (
        source_root
        / "Beyond Throughput a 4G LTE Dataset with Channel and Context Metrics"
        / "LTE_Dataset.zip"
    )

    if not args.skip_puffer:
        build_puffer_recent(
            source_dir=puffer_dir,
            output_dir=args.puffer_output_dir,
            max_traces=args.max_traces,
            min_seconds=args.min_seconds,
            seed=args.seed,
            clear=args.clear,
        )
    if not args.skip_weak:
        build_weak_mobile(
            perform_csv=perform_csv,
            lte_zip=lte_zip,
            output_dir=args.weak_output_dir,
            max_traces=args.max_traces,
            min_seconds=args.min_seconds,
            seed=args.seed,
            clear=args.clear,
        )


if __name__ == "__main__":
    main()
