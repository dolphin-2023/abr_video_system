import os
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent

# Use ABR_LLM_PATH for machine-specific model locations.
BASE_MODEL_PATH = Path(
    os.environ.get("ABR_LLM_PATH", PROJECT_ROOT / "models" / "qwen3.5-4b-base")
)

MODEL_DIR = AGENT_DIR / "models"
SFT_MODEL_PATH = MODEL_DIR / "netllm_sft"
OFFLINE_RL_MODEL_PATH = MODEL_DIR / "netllm_offline_rl"
RL_MODEL_PATH = MODEL_DIR / "netllm_rl"
PENSIEVE_MODEL_PATH = MODEL_DIR / "pensieve_torch"
LEGACY_SFT_MODEL_PATH = MODEL_DIR / "netllm_sft.pth"
LEGACY_RL_MODEL_PATH = MODEL_DIR / "netllm_rl.pth"
TRAINING_STATS_PATH = MODEL_DIR / "training_stats.json"

ACTION_DIM = 6
PAST_K = 6
SEGMENT_DURATION = 4.0
TOTAL_SEGMENTS = 60

M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0

BITRATES_KBPS = [300, 600, 900, 2000, 4500, 9000]

# Fallback used before a dataset-specific value is available. Existing
# pre-optimization checkpoints were trained with raw returns around this scale.
DEFAULT_TARGET_RETURN = 800.0


def normalize_remaining_chunks(remaining_chunks):
    """Normalize the remaining chunk count for row 5 of the ABR state matrix.

    Both ABREnv._update_state and SessionState.update use this so that
    training and inference stay consistent on clamping behavior.
    """
    return min(max(0, remaining_chunks), CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
