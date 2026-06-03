import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from settings import AGENT_DIR, BITRATES_KBPS, M_IN_K, PROJECT_ROOT, SEGMENT_DURATION


CHUNK_RE = re.compile(r"^chunk_(?P<rep>[^_]+)_(?P<num>\d+)\.m4s$")
DEFAULT_DASH_ROOT = PROJECT_ROOT / "processor" / "dash_output"


def cbr_fallback_sizes_mb(bitrates_kbps=None):
    """没有真实 DASH 文件时，回退到 CBR 估计值。"""
    bitrates = list(BITRATES_KBPS if bitrates_kbps is None else bitrates_kbps)
    return np.asarray(
        [(br * SEGMENT_DURATION) / 8.0 / M_IN_K for br in bitrates],
        dtype=np.float32,
    )


def find_video_dir(video_id, dash_root=None):
    dash_root = Path(dash_root or DEFAULT_DASH_ROOT)
    video_dir = dash_root / str(video_id)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"DASH video directory not found: {video_dir}")
    return video_dir


def parse_video_representations(video_dir):
    """从 MPD 中读取 video representation 与码率的映射。"""
    video_dir = Path(video_dir)
    mpd_files = sorted(video_dir.glob("*.mpd"))
    if not mpd_files:
        raise FileNotFoundError(f"No MPD file found in {video_dir}")

    root = ET.parse(mpd_files[0]).getroot()
    namespace = {"mpd": "urn:mpeg:dash:schema:mpd:2011"}
    representations = []

    for adaptation in root.findall(".//mpd:AdaptationSet", namespace):
        if adaptation.attrib.get("contentType") != "video":
            continue
        for rep in adaptation.findall("mpd:Representation", namespace):
            rep_id = rep.attrib.get("id")
            bandwidth = rep.attrib.get("bandwidth")
            if rep_id is None or bandwidth is None:
                continue
            representations.append({
                "representation_id": rep_id,
                "bitrate_kbps": float(bandwidth) / 1000.0,
                "mpd_path": str(mpd_files[0]),
            })

    if not representations:
        raise ValueError(f"No video representations found in {mpd_files[0]}")
    return representations


def build_chunk_size_table(video_id=None, video_dir=None, dash_root=None, bitrates_kbps=None):
    """统计每个码率档位、每个 chunk 的真实大小，单位为 MB。"""
    bitrates = np.asarray(
        list(BITRATES_KBPS if bitrates_kbps is None else bitrates_kbps),
        dtype=np.float32,
    )
    if video_dir is None:
        video_dir = find_video_dir(video_id, dash_root=dash_root)
    video_dir = Path(video_dir)

    reps = parse_video_representations(video_dir)
    rep_to_action = {}
    rep_metadata = []
    for rep in reps:
        action_idx = int(np.argmin(np.abs(bitrates - rep["bitrate_kbps"])))
        rep_to_action[str(rep["representation_id"])] = action_idx
        rep_metadata.append({
            "representation_id": str(rep["representation_id"]),
            "bitrate_kbps": rep["bitrate_kbps"],
            "action_index": action_idx,
        })

    sizes_by_action = {idx: {} for idx in range(len(bitrates))}
    for chunk_path in video_dir.glob("chunk_*_*.m4s"):
        match = CHUNK_RE.match(chunk_path.name)
        if not match:
            continue
        rep_id = match.group("rep")
        if rep_id not in rep_to_action:
            continue
        action_idx = rep_to_action[rep_id]
        segment_num = int(match.group("num"))
        sizes_by_action[action_idx][segment_num] = chunk_path.stat().st_size / 1_000_000.0

    max_segment = max(
        (max(items) for items in sizes_by_action.values() if items),
        default=0,
    )
    if max_segment <= 0:
        raise ValueError(f"No video chunks found in {video_dir}")

    fallback = cbr_fallback_sizes_mb(bitrates)
    table = np.zeros((len(bitrates), max_segment), dtype=np.float32)
    for action_idx in range(len(bitrates)):
        for segment_num in range(1, max_segment + 1):
            table[action_idx, segment_num - 1] = sizes_by_action[action_idx].get(
                segment_num,
                float(fallback[action_idx]),
            )

    metadata = {
        "video_id": str(video_id or video_dir.name),
        "video_dir": str(video_dir),
        "bitrates_kbps": [float(item) for item in bitrates],
        "num_segments": int(max_segment),
        "representations": rep_metadata,
    }
    return table, metadata


def save_chunk_size_index(video_id, output_path, dash_root=None, bitrates_kbps=None):
    table, metadata = build_chunk_size_table(
        video_id=video_id,
        dash_root=dash_root,
        bitrates_kbps=bitrates_kbps,
    )
    payload = dict(metadata)
    payload["sizes_mb"] = table.tolist()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def load_chunk_size_json(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = np.asarray(payload["sizes_mb"], dtype=np.float32)
    metadata = {key: value for key, value in payload.items() if key != "sizes_mb"}
    metadata["chunk_size_path"] = str(path)
    return table, metadata


def load_chunk_sizes_from_env(bitrates_kbps=None):
    """ABREnv 调用入口：优先读 JSON，其次按 ABR_VIDEO_ID 扫描 DASH 目录。"""
    import os

    json_path = os.environ.get("ABR_CHUNK_SIZE_PATH")
    if json_path:
        return load_chunk_size_json(json_path)

    video_id = os.environ.get("ABR_VIDEO_ID")
    if video_id:
        dash_root = os.environ.get("ABR_DASH_ROOT")
        return build_chunk_size_table(
            video_id=video_id,
            dash_root=dash_root,
            bitrates_kbps=bitrates_kbps,
        )

    return None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect real DASH chunk sizes for ABREnv."
    )
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--dash-root", type=Path, default=DEFAULT_DASH_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=AGENT_DIR / "assets" / "chunk_sizes.local.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = save_chunk_size_index(
        video_id=args.video_id,
        output_path=args.output,
        dash_root=args.dash_root,
    )
    print(
        f"[ChunkSizes] video_id={payload['video_id']} "
        f"segments={payload['num_segments']} output={args.output}"
    )


if __name__ == "__main__":
    main()
