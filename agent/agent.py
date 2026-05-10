from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chunk_sizes import build_chunk_size_table
from checkpoint_utils import load_netllm_checkpoint
from network import NetLLMABR
from return_utils import load_return_to_go_processor
from settings import (
    ACTION_DIM,
    BASE_MODEL_PATH,
    BUFFER_NORM_FACTOR,
    CHUNK_TIL_VIDEO_END_CAP,
    M_IN_K,
    OFFLINE_RL_MODEL_PATH,
    PAST_K,
    RL_MODEL_PATH,
    SFT_MODEL_PATH,
    normalize_remaining_chunks,
)


app = FastAPI(title="NetLLM ABR Decision Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NetworkState(BaseModel):
    video_id: str
    buffer_level: float
    throughput_kbps: float
    available_bitrates: List[int]
    current_quality_index: int
    segment_duration: Optional[float] = 4.0
    last_reward: Optional[float] = None
    segment_qoe: Optional[float] = None


class ResetSessionRequest(BaseModel):
    video_id: str


class SessionState:
    """Per-playback state used to build NetLLM's (6, 6) observation."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.state_matrix = np.zeros((6, PAST_K), dtype=np.float32)
        self.throughput_history = [0.0] * PAST_K
        self.download_time_history = [0.0] * PAST_K
        self.last_quality_idx = 0
        self.step_counter = 0
        self.llm_history = None
        self.remaining_target_return = None
        self.has_model_decision = False
        self.chunk_sizes_mb = None
        self.chunk_size_attempted = False

    def load_chunk_sizes_once(self, bitrates_kbps):
        if self.chunk_size_attempted:
            return
        self.chunk_size_attempted = True
        try:
            table, metadata = build_chunk_size_table(
                video_id=self.video_id,
                bitrates_kbps=bitrates_kbps,
            )
            self.chunk_sizes_mb = table
            print(
                f"[Agent] loaded real chunk sizes: "
                f"video={metadata['video_id']} segments={metadata['num_segments']}"
            )
        except Exception:
            # 前端可直接使用任意 video_id；找不到 DASH 文件时继续用 CBR 估计。
            self.chunk_sizes_mb = None

    def update(
        self,
        throughput_kbps: float,
        buffer_level: float,
        quality_idx: int,
        download_time_ms: float,
        available_bitrates_bps: List[int],
        segment_duration: float,
    ):
        bitrates_kbps = [bps / 1000.0 for bps in available_bitrates_bps]
        max_br_kbps = max(bitrates_kbps) if bitrates_kbps else 9000.0
        self.load_chunk_sizes_once(bitrates_kbps)

        self.throughput_history = self.throughput_history[1:] + [float(throughput_kbps)]
        self.download_time_history = self.download_time_history[1:] + [float(download_time_ms)]

        self.state_matrix = np.roll(self.state_matrix, -1, axis=1)

        safe_quality_idx = min(max(int(quality_idx), 0), max(len(bitrates_kbps) - 1, 0))
        br_kbps = bitrates_kbps[safe_quality_idx] if bitrates_kbps else 0.0
        self.state_matrix[0, -1] = br_kbps / max_br_kbps if max_br_kbps > 0 else 0.0
        self.state_matrix[1, -1] = float(buffer_level) / BUFFER_NORM_FACTOR

        for idx, throughput in enumerate(self.throughput_history):
            self.state_matrix[2, idx] = throughput / M_IN_K

        for idx, download_ms in enumerate(self.download_time_history):
            self.state_matrix[3, idx] = download_ms / M_IN_K / BUFFER_NORM_FACTOR

        for idx in range(ACTION_DIM):
            if idx < len(bitrates_kbps):
                if self.chunk_sizes_mb is not None and idx < self.chunk_sizes_mb.shape[0]:
                    segment_idx = min(self.step_counter, self.chunk_sizes_mb.shape[1] - 1)
                    self.state_matrix[4, idx] = float(self.chunk_sizes_mb[idx, segment_idx])
                else:
                    self.state_matrix[4, idx] = (
                        bitrates_kbps[idx] * float(segment_duration) / 8.0 / M_IN_K
                    )
            else:
                self.state_matrix[4, idx] = 0.0

        total_segments = (
            int(self.chunk_sizes_mb.shape[1])
            if self.chunk_sizes_mb is not None
            else 60
        )
        remaining = max(0, total_segments - self.step_counter)
        self.state_matrix[5, -1] = normalize_remaining_chunks(remaining)
        self.step_counter += 1
        self.last_quality_idx = safe_quality_idx
        return self.state_matrix


session_states: Dict[str, SessionState] = {}


def get_session(video_id: str) -> SessionState:
    if video_id not in session_states:
        session_states[video_id] = SessionState(video_id)
    return session_states[video_id]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RTG_PROCESSOR = load_return_to_go_processor()
TARGET_RETURN = RTG_PROCESSOR.target_return

print(f"[System] building NetLLM agent on {DEVICE}")
print(f"[System] base model: {BASE_MODEL_PATH}")
netllm_agent = NetLLMABR(
    model_name_or_path=str(BASE_MODEL_PATH),
    action_dim=ACTION_DIM,
    lora_rank=128,
)

ACTIVE_POLICY_NAME = "random-init"
ACTIVE_POLICY_PATH = None
for policy_name, checkpoint_path in (
    ("netllm-rl", RL_MODEL_PATH),
    ("netllm-offline-rl", OFFLINE_RL_MODEL_PATH),
    ("netllm-sft", SFT_MODEL_PATH),
):
    loaded_path = load_netllm_checkpoint(
        netllm_agent,
        checkpoint_path,
        map_location=DEVICE,
        strict=False,
    )
    if loaded_path is not None:
        ACTIVE_POLICY_NAME = policy_name
        ACTIVE_POLICY_PATH = loaded_path
        print(f"[System] loaded {policy_name} weights: {loaded_path}")
        break

if ACTIVE_POLICY_PATH is None:
    print("[System] RL/OfflineRL/SFT weights not found; using randomly initialized adapters and head.")

netllm_agent.to(DEVICE)
netllm_agent.eval()


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "base_model": str(BASE_MODEL_PATH),
        "active_policy": ACTIVE_POLICY_NAME,
        "active_policy_path": str(ACTIVE_POLICY_PATH) if ACTIVE_POLICY_PATH else None,
        "rl_model": str(RL_MODEL_PATH),
        "offline_rl_model": str(OFFLINE_RL_MODEL_PATH),
        "sft_model": str(SFT_MODEL_PATH),
        "target_return": TARGET_RETURN,
        "rtg": RTG_PROCESSOR.describe(),
    }


@app.post("/abr_decision")
async def make_abr_decision(state: NetworkState):
    available_bitrates_bps = state.available_bitrates
    available_count = len(available_bitrates_bps)
    if available_count == 0:
        return {"quality_index": 0, "llm_reasoning": "No available bitrate levels."}

    session = get_session(state.video_id)
    if session.llm_history is None:
        session.llm_history = netllm_agent.new_history()
    if session.remaining_target_return is None:
        session.remaining_target_return = TARGET_RETURN

    bitrates_kbps = [bps / 1000.0 for bps in available_bitrates_bps]
    current_idx = min(max(int(state.current_quality_index), 0), available_count - 1)
    chosen_br_kbps = bitrates_kbps[current_idx]
    segment_duration = float(state.segment_duration or 4.0)
    chunk_size_kbit = chosen_br_kbps * segment_duration
    estimated_download_ms = chunk_size_kbit / max(10.0, state.throughput_kbps) * M_IN_K

    state_matrix = session.update(
        throughput_kbps=state.throughput_kbps,
        buffer_level=state.buffer_level,
        quality_idx=current_idx,
        download_time_ms=estimated_download_ms,
        available_bitrates_bps=available_bitrates_bps,
        segment_duration=segment_duration,
    )

    observed_reward = (
        state.last_reward
        if state.last_reward is not None
        else state.segment_qoe
    )
    if session.has_model_decision and observed_reward is not None:
        session.remaining_target_return = RTG_PROCESSOR.update(
            session.remaining_target_return,
            observed_reward,
        )
    target_return = session.remaining_target_return
    state_tensor = torch.tensor(
        state_matrix,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        with autocast_context():
            logits, return_emb, state_emb, time_emb = netllm_agent.predict_logits(
                state=state_tensor,
                target_return=target_return,
                timestep=session.step_counter,
                history=session.llm_history,
            )
            decision, _ = netllm_agent.select_action(logits, deterministic=True)

    decision = min(int(decision), available_count - 1)
    if abs(decision - current_idx) > 1:
        decision = current_idx + (1 if decision > current_idx else -1)
    decision = max(0, min(available_count - 1, decision))

    netllm_agent.append_history(
        history=session.llm_history,
        return_emb=return_emb,
        state_emb=state_emb,
        time_emb=time_emb,
        action=decision,
        device=DEVICE,
    )
    session.has_model_decision = True

    print(
        f"[Agent] buffer={state.buffer_level:.1f}s "
        f"throughput={state.throughput_kbps:.0f}Kbps "
        f"quality={current_idx}->{decision} "
        f"step={session.step_counter} "
        f"target_return={target_return:.4f}"
    )

    return {
        "quality_index": decision,
        "llm_reasoning": f"NetLLM target_return={target_return:.4f}",
    }


@app.post("/reset_session")
async def reset_session(
    request: Optional[ResetSessionRequest] = Body(default=None),
    video_id: Optional[str] = None,
):
    target_video_id = video_id or (request.video_id if request else None)
    if target_video_id is None:
        return {"status": "error", "message": "video_id is required"}
    session_states.pop(target_video_id, None)
    return {"status": "ok", "video_id": target_video_id}


if __name__ == "__main__":
    uvicorn.run("agent:app", host="127.0.0.1", port=8081, reload=True)
