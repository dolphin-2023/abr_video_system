import glob
import os
import random
from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from settings import (
    BITRATES_KBPS,
    BUFFER_NORM_FACTOR,
    CHUNK_TIL_VIDEO_END_CAP,
    M_IN_K,
    PAST_K,
    SEGMENT_DURATION,
    TOTAL_SEGMENTS,
)
from chunk_sizes import load_chunk_sizes_from_env


PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT_SEC = 0.08
MIN_THROUGHPUT_KBPS = 10.0


class ABREnv(gym.Env):
    """Pensieve/NetLLM-style ABR simulation environment.

    Observation shape is (6, 6):
    row 0: last selected bitrate, normalized by max bitrate
    row 1: buffer level in seconds, normalized by 10
    row 2: throughput history in Kbps, normalized by 1000
    row 3: download time history in ms, normalized by 1000 and 10
    row 4: next chunk sizes in MB for each bitrate
    row 5: remaining chunks ratio, capped at 48 chunks
    """

    metadata = {"render_modes": []}
    _reported_chunk_size_keys = set()
    _trace_cache = {}

    def __init__(
        self,
        trace_split=None,
        trace_dir=None,
        trace_file=None,
        random_start=True,
        chunk_sizes_mb=None,
        chunk_size_metadata=None,
        qoe_profile=None,
        verbose=None,
        exclude_trace_files=None,
    ):
        super().__init__()

        self.bitrates_kbps = list(BITRATES_KBPS)
        self.num_bitrates = len(self.bitrates_kbps)
        self.past_k = PAST_K

        self.action_space = spaces.Discrete(self.num_bitrates)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(6, self.past_k),
            dtype=np.float32,
        )

        self.max_buffer_capacity = 30.0
        self.segment_duration = SEGMENT_DURATION
        self.total_segments = TOTAL_SEGMENTS
        self.random_start = bool(random_start)

        self.penalty_stall = 4.3
        self.penalty_smooth = 1.0
        self.qoe_profile = self._resolve_qoe_profile(qoe_profile)
        self.weight_quality = 3.0 if self.qoe_profile == "linear3" else 1.0

        self.configured_trace_split = trace_split
        self.configured_trace_dir = trace_dir
        self.fixed_trace_file = os.path.abspath(trace_file) if trace_file else None
        self.exclude_trace_files = self._normalize_excluded_trace_files(exclude_trace_files)
        self.verbose = self._resolve_verbose(verbose)
        self.trace_dir = self._resolve_trace_dir()
        self.trace_files = sorted(glob.glob(os.path.join(self.trace_dir, "*.txt")))
        if self.fixed_trace_file:
            self.trace_files = [self.fixed_trace_file]
        elif self.exclude_trace_files:
            before_count = len(self.trace_files)
            self.trace_files = [
                trace_file
                for trace_file in self.trace_files
                if not self._is_excluded_trace_file(trace_file)
            ]
            excluded_count = before_count - len(self.trace_files)
            if excluded_count > 0 and self.verbose:
                print(f"[ABREnv] excluded_traces={excluded_count}")
            if before_count > 0 and not self.trace_files:
                raise RuntimeError("All traces were excluded from ABREnv.")
        self.trace_files_by_source = self._group_trace_files(self.trace_files)
        if self.trace_files and self.verbose:
            group_summary = ", ".join(
                f"{source}:{len(files)}"
                for source, files in sorted(self.trace_files_by_source.items())
            )
            print(
                f"[ABREnv] traces={len(self.trace_files)} "
                f"dir={self.trace_dir} groups=({group_summary})"
            )

        if chunk_sizes_mb is None:
            chunk_sizes_mb, chunk_size_metadata = load_chunk_sizes_from_env(self.bitrates_kbps)
        self.chunk_sizes_mb = (
            np.asarray(chunk_sizes_mb, dtype=np.float32)
            if chunk_sizes_mb is not None
            else None
        )
        self.chunk_size_metadata = chunk_size_metadata or {}
        if self.chunk_sizes_mb is not None:
            if self.chunk_sizes_mb.shape[0] != self.num_bitrates:
                raise ValueError(
                    "chunk size table bitrate dimension does not match ACTION_DIM"
                )
            # 有真实 DASH chunk 时，以真实视频长度作为 episode 长度。
            self.total_segments = int(self.chunk_sizes_mb.shape[1])
            chunk_key = (
                self.chunk_size_metadata.get("chunk_size_path"),
                self.chunk_size_metadata.get("video_id"),
                self.total_segments,
            )
            if self.verbose and chunk_key not in ABREnv._reported_chunk_size_keys:
                ABREnv._reported_chunk_size_keys.add(chunk_key)
                print(
                    f"[ABREnv] real_chunk_sizes=on "
                    f"segments={self.total_segments} "
                    f"video={self.chunk_size_metadata.get('video_id', 'unknown')}"
                )

        self.state_matrix = np.zeros((6, self.past_k), dtype=np.float32)
        self.throughput_history = deque([0.0] * self.past_k, maxlen=self.past_k)
        self.download_time_history = deque([0.0] * self.past_k, maxlen=self.past_k)
        self.current_trace = []
        self.current_trace_times = []
        self.trace_ptr = 1
        self.last_trace_time = 0.0
        self.current_trace_name = "synthetic"
        self.current_buffer = 0.0
        self.last_quality_idx = 0
        self.current_segment = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state_matrix = np.zeros((6, self.past_k), dtype=np.float32)
        self.throughput_history = deque([0.0] * self.past_k, maxlen=self.past_k)
        self.download_time_history = deque([0.0] * self.past_k, maxlen=self.past_k)
        self.current_buffer = 0.0
        self.last_quality_idx = 0
        self.current_segment = 0

        self._load_random_trace()
        self._refresh_initial_state()
        return self.state_matrix.copy(), {}

    def step(self, action):
        action = int(action)
        current_throughput_kbps = float(self.current_trace[self.trace_ptr])
        chosen_bitrate_kbps = float(self.bitrates_kbps[action])
        chunk_size_mb = self._get_chunk_size_mb(action, self.current_segment)
        chunk_size_kbit = chunk_size_mb * 8.0 * M_IN_K

        buffer_before_download = self.current_buffer

        download_time = self._download_chunk_time(chunk_size_kbit)
        stall_time = max(0.0, download_time - buffer_before_download)
        self.current_buffer = (
            max(0.0, buffer_before_download - download_time) + self.segment_duration
        )
        player_sleep_time = max(0.0, self.current_buffer - self.max_buffer_capacity)
        if player_sleep_time > 0.0:
            self.current_buffer -= player_sleep_time
            self._advance_trace_time(player_sleep_time)

        quality_score = self.compute_quality_score(chosen_bitrate_kbps)
        smooth_penalty = self.compute_smooth_penalty(
            chosen_bitrate_kbps,
            self.bitrates_kbps[self.last_quality_idx],
        )
        reward = quality_score - self.penalty_stall * stall_time - self.penalty_smooth * smooth_penalty

        measured_throughput_kbps = chunk_size_kbit / max(download_time, 1e-6)
        self.throughput_history.append(measured_throughput_kbps)
        self.download_time_history.append(download_time * M_IN_K)

        self._update_state(action, chosen_bitrate_kbps)

        self.last_quality_idx = action
        self.current_segment += 1
        done = self.current_segment >= self.total_segments

        info = {
            "trace_name": self.current_trace_name,
            "quality_idx": action,
            "qoe_profile": self.qoe_profile,
            "bitrate_kbps": chosen_bitrate_kbps,
            "chunk_size_mb": chunk_size_mb,
            "chunk_size_kbit": chunk_size_kbit,
            "stall_time": stall_time,
            "qoe": reward,
            "quality_score": quality_score,
            "smoothness_penalty": smooth_penalty,
            "throughput_kbps": current_throughput_kbps,
            "measured_throughput_kbps": measured_throughput_kbps,
            "download_time": download_time,
            "sleep_time": player_sleep_time,
        }
        return self.state_matrix.copy(), reward, bool(done), False, info

    def _load_random_trace(self):
        if not self.trace_files:
            trace_len = max(self.total_segments + 1, 2)
            self.current_trace = np.random.uniform(500, 5000, size=trace_len).tolist()
            self.current_trace_name = "synthetic"
            self._reset_trace_clock()
            return

        trace_file = self._choose_trace_file()
        self.current_trace_name = os.path.basename(trace_file)
        trace_data = self._read_trace_file(trace_file)

        if not trace_data:
            trace_len = max(self.total_segments + 1, 2)
            self.current_trace = np.random.uniform(500, 5000, size=trace_len).tolist()
        elif len(trace_data) < 2:
            self.current_trace = [trace_data[0], trace_data[0]]
        else:
            self.current_trace = trace_data
        self._reset_trace_clock()

    @classmethod
    def _read_trace_file(cls, trace_file):
        trace_path = os.path.abspath(str(trace_file))
        cached = cls._trace_cache.get(trace_path)
        if cached is not None:
            return cached

        trace_data = []
        try:
            with open(trace_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        trace_data.append(float(line))
        except (OSError, ValueError):
            trace_data = []

        cls._trace_cache[trace_path] = trace_data
        return trace_data

    def _reset_trace_clock(self):
        self.current_trace = list(self.current_trace)
        if len(self.current_trace) < 2:
            fallback = float(self.current_trace[0]) if self.current_trace else 1000.0
            self.current_trace = [fallback, fallback]
        self.current_trace_times = [float(idx) for idx in range(len(self.current_trace))]
        self.trace_ptr = (
            random.randint(1, len(self.current_trace) - 1)
            if self.random_start
            else 1
        )
        self.last_trace_time = self.current_trace_times[self.trace_ptr - 1]

    def _download_chunk_time(self, chunk_size_kbit):
        sent_kbit = 0.0
        delay_sec = 0.0

        while True:
            throughput_kbps = max(MIN_THROUGHPUT_KBPS, float(self.current_trace[self.trace_ptr]))
            duration = self.current_trace_times[self.trace_ptr] - self.last_trace_time
            if duration <= 1e-12:
                self._move_to_next_trace_interval()
                continue

            payload_kbit = throughput_kbps * duration * PACKET_PAYLOAD_PORTION
            if sent_kbit + payload_kbit >= chunk_size_kbit:
                fractional_time = (
                    (chunk_size_kbit - sent_kbit)
                    / (throughput_kbps * PACKET_PAYLOAD_PORTION)
                )
                delay_sec += fractional_time
                self.last_trace_time += fractional_time
                break

            sent_kbit += payload_kbit
            delay_sec += duration
            self.last_trace_time = self.current_trace_times[self.trace_ptr]
            self._move_to_next_trace_interval()

        return delay_sec + LINK_RTT_SEC

    def _advance_trace_time(self, sleep_time):
        remaining = max(0.0, float(sleep_time))
        while remaining > 1e-12:
            duration = self.current_trace_times[self.trace_ptr] - self.last_trace_time
            if duration <= 1e-12:
                self._move_to_next_trace_interval()
                continue
            if duration > remaining:
                self.last_trace_time += remaining
                break
            remaining -= duration
            self.last_trace_time = self.current_trace_times[self.trace_ptr]
            self._move_to_next_trace_interval()

    def _move_to_next_trace_interval(self):
        self.trace_ptr += 1
        if self.trace_ptr >= len(self.current_trace):
            self.trace_ptr = 1
            self.last_trace_time = self.current_trace_times[0]

    def _choose_trace_file(self):
        if not self.trace_files_by_source:
            return random.choice(self.trace_files)
        source = random.choice(list(self.trace_files_by_source.keys()))
        return random.choice(self.trace_files_by_source[source])

    def _resolve_trace_dir(self):
        if self.configured_trace_dir:
            return os.path.abspath(self.configured_trace_dir)

        configured_dir = os.environ.get("ABR_TRACE_DIR")
        if configured_dir:
            return os.path.abspath(configured_dir)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        split_name = (
            self.configured_trace_split
            or os.environ.get("ABR_TRACE_SPLIT", "train")
        )
        split_name = str(split_name).strip().lower() or "train"
        direct_split_dir = os.path.join(project_root, "simulator", "traces", split_name)
        if os.path.isdir(direct_split_dir):
            return os.path.abspath(direct_split_dir)

        split_dir = os.path.join(project_root, "simulator", "traces", "real_world_split", split_name)
        if os.path.isdir(split_dir):
            return os.path.abspath(split_dir)

        legacy_dir = os.path.join(project_root, "simulator", "traces", "real_world")
        return os.path.abspath(legacy_dir)

    def _resolve_qoe_profile(self, qoe_profile):
        configured = qoe_profile or os.environ.get("ABR_QOE_PROFILE", "pensieve")
        profile = str(configured).strip().lower()
        aliases = {
            "default": "pensieve",
            "linear": "pensieve",
            "pensieve": "pensieve",
            "linear3": "linear3",
            "old": "linear3",
            "log": "log",
            "logqoe": "log",
        }
        return aliases.get(profile, "pensieve")

    def _resolve_verbose(self, verbose):
        if verbose is not None:
            return bool(verbose)
        configured = os.environ.get("ABR_ENV_VERBOSE", "")
        return str(configured).strip().lower() in {"1", "true", "yes", "on"}

    def compute_quality_score(self, bitrate_kbps):
        """QoE 的质量项：训练默认使用 Pensieve 的 Mbps 线性收益。"""
        bitrate_kbps = float(bitrate_kbps)
        if self.qoe_profile == "log":
            min_bitrate = max(float(min(self.bitrates_kbps)), 1.0)
            return float(np.log(max(bitrate_kbps, 1.0) / min_bitrate))
        return (bitrate_kbps / M_IN_K) * self.weight_quality

    def compute_smooth_penalty(self, bitrate_kbps, last_bitrate_kbps):
        """QoE 的平滑项，Log QoE 下使用 log utility 的变化量。"""
        bitrate_kbps = float(bitrate_kbps)
        last_bitrate_kbps = float(last_bitrate_kbps)
        if self.qoe_profile == "log":
            min_bitrate = max(float(min(self.bitrates_kbps)), 1.0)
            now_u = np.log(max(bitrate_kbps, 1.0) / min_bitrate)
            last_u = np.log(max(last_bitrate_kbps, 1.0) / min_bitrate)
            return float(abs(now_u - last_u))
        return abs(bitrate_kbps - last_bitrate_kbps) / M_IN_K

    def _group_trace_files(self, trace_files):
        grouped = defaultdict(list)
        for trace_file in trace_files:
            name = os.path.basename(trace_file)
            source = self._infer_trace_source(name)
            grouped[source].append(trace_file)
        return dict(grouped)

    def _normalize_excluded_trace_files(self, trace_files):
        excluded = {"names": set(), "paths": set()}
        for trace_file in trace_files or []:
            trace_path = os.path.abspath(str(trace_file))
            excluded["paths"].add(trace_path)
            excluded["names"].add(os.path.basename(trace_path))
        return excluded

    def _is_excluded_trace_file(self, trace_file):
        trace_path = os.path.abspath(str(trace_file))
        return (
            trace_path in self.exclude_trace_files["paths"]
            or os.path.basename(trace_path) in self.exclude_trace_files["names"]
        )

    def _infer_trace_source(self, filename):
        if "__" in filename:
            return filename.split("__", 1)[0]
        for prefix in ("puffer", "hsdpa", "cooked", "oboe", "fcc", "mixed"):
            if filename.startswith(prefix + "_"):
                return prefix
        return "other"

    def estimate_chunk_size_kbit(self, action, segment_index=None):
        """供 MPC/evaluate 使用，和环境 step 保持同一套 chunk size。"""
        return self._get_chunk_size_mb(action, segment_index) * 8.0 * M_IN_K

    def _get_chunk_size_mb(self, action, segment_index=None):
        if self.chunk_sizes_mb is not None:
            idx = self.current_segment if segment_index is None else int(segment_index)
            idx = max(0, min(idx, self.chunk_sizes_mb.shape[1] - 1))
            return float(self.chunk_sizes_mb[int(action), idx])
        return (self.bitrates_kbps[int(action)] * self.segment_duration) / 8.0 / M_IN_K

    def _compute_next_chunk_sizes(self, segment_index=None):
        if self.chunk_sizes_mb is not None:
            idx = self.current_segment + 1 if segment_index is None else int(segment_index)
            idx = max(0, min(idx, self.chunk_sizes_mb.shape[1] - 1))
            return [float(self.chunk_sizes_mb[action_idx, idx]) for action_idx in range(self.num_bitrates)]

        return [
            (br_kbps * self.segment_duration) / 8.0 / M_IN_K
            for br_kbps in self.bitrates_kbps
        ]

    def _refresh_initial_state(self):
        # 初始状态也写入下一块大小，避免第一步完全看不到视频侧信息。
        for idx, size_mb in enumerate(self._compute_next_chunk_sizes(segment_index=0)):
            self.state_matrix[4, idx] = size_mb
        self.state_matrix[5, -1] = (
            min(self.total_segments, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
        )

    def _update_state(self, action, chosen_bitrate_kbps):
        self.state_matrix = np.roll(self.state_matrix, -1, axis=1)
        self.state_matrix[0, -1] = chosen_bitrate_kbps / float(max(self.bitrates_kbps))
        self.state_matrix[1, -1] = self.current_buffer / BUFFER_NORM_FACTOR

        for idx, throughput in enumerate(self.throughput_history):
            self.state_matrix[2, idx] = throughput / M_IN_K

        for idx, download_ms in enumerate(self.download_time_history):
            self.state_matrix[3, idx] = download_ms / M_IN_K / BUFFER_NORM_FACTOR

        for idx, size_mb in enumerate(self._compute_next_chunk_sizes()):
            self.state_matrix[4, idx] = size_mb

        remaining = max(0, self.total_segments - (self.current_segment + 1))
        self.state_matrix[5, -1] = (
            min(remaining, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
        )
