import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def infer_trace_source(path):
    name = Path(path).name
    if "__" in name:
        return name.split("__", 1)[0]
    for prefix in ("puffer", "hsdpa", "cooked", "oboe", "fcc", "mixed"):
        if name.startswith(prefix + "_"):
            return prefix
    return "other"


def group_trace_files(trace_files):
    grouped = defaultdict(list)
    for trace_file in trace_files:
        grouped[infer_trace_source(trace_file)].append(trace_file)
    return dict(grouped)


def sample_trace_files(trace_files, episodes, mode="stratified", seed=0):
    files = list(trace_files)
    if episodes <= 0 or episodes >= len(files):
        return files

    rng = random.Random(seed)
    mode = str(mode).strip().lower()

    if mode == "random":
        rng.shuffle(files)
        return files[:episodes]

    if mode == "first" or mode != "stratified":
        return files[:episodes]

    grouped = defaultdict(list)
    for trace_file in files:
        grouped[infer_trace_source(trace_file)].append(trace_file)

    shuffled_groups = {}
    for source, group_files in sorted(grouped.items()):
        group = list(group_files)
        rng.shuffle(group)
        shuffled_groups[source] = group

    total = len(files)
    quotas = {
        source: episodes * len(group_files) / total
        for source, group_files in shuffled_groups.items()
    }
    counts = {
        source: min(len(shuffled_groups[source]), int(quotas[source]))
        for source in shuffled_groups
    }

    selected = sum(counts.values())
    for source in sorted(shuffled_groups, key=lambda item: quotas[item], reverse=True):
        if selected >= episodes:
            break
        if counts[source] == 0:
            counts[source] = 1
            selected += 1

    while selected < episodes:
        candidates = [
            source
            for source, group_files in shuffled_groups.items()
            if counts[source] < len(group_files)
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda source: (
                quotas[source] - int(quotas[source]),
                len(shuffled_groups[source]),
            ),
            reverse=True,
        )
        for source in candidates:
            if selected >= episodes:
                break
            counts[source] += 1
            selected += 1

    sampled = []
    for source, group_files in shuffled_groups.items():
        sampled.extend(group_files[:counts[source]])
    rng.shuffle(sampled)
    return sampled[:episodes]


def read_trace_values(trace_file):
    values = []
    try:
        with open(trace_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    values.append(float(line))
    except (OSError, ValueError):
        return []
    return values


def trace_mean_throughput_kbps(trace_file):
    values = read_trace_values(trace_file)
    if not values:
        return 0.0
    return float(np.mean(values))


def filter_high_bandwidth_trace_files(
    trace_files,
    min_mean_throughput_kbps=5000.0,
    percentile=70.0,
):
    files = list(trace_files)
    if not files:
        return [], 0.0

    scored = [(trace_mean_throughput_kbps(trace_file), trace_file) for trace_file in files]
    scores = np.asarray([score for score, _ in scored], dtype=np.float32)
    threshold = max(
        float(min_mean_throughput_kbps),
        float(np.percentile(scores, float(percentile))),
    )
    selected = [trace_file for score, trace_file in scored if score >= threshold]
    if not selected:
        scored.sort(reverse=True)
        keep_count = max(1, int(math.ceil(len(scored) * 0.25)))
        selected = [trace_file for _, trace_file in scored[:keep_count]]
    return sorted(selected), threshold
