import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from settings import DEFAULT_TARGET_RETURN, TRAINING_STATS_PATH


def _as_optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(mapping, keys, default=None):
    for key in keys:
        if key in mapping:
            value = _as_optional_float(mapping.get(key))
            if value is not None:
                return value
    return default


@dataclass
class ReturnToGoProcessor:
    """统一管理 Decision Transformer 的 return-to-go。

    SFT 阶段没有直接使用原始 QoE，而是把 reward 按训练集范围归一化，
    再除以 return_scale。评估、在线推理和 RL 都要使用同一套尺度扣减，
    否则模型看到的 target_return 会和训练时不一致。
    """

    target_return: float
    min_reward: Optional[float] = None
    max_reward: Optional[float] = None
    return_scale: float = 1.0
    clip_reward: bool = True
    source: str = "default"

    @property
    def uses_normalized_reward(self):
        return (
            self.min_reward is not None
            and self.max_reward is not None
            and self.max_reward > self.min_reward
            and self.return_scale > 0.0
        )

    def process_reward(self, reward):
        """把环境返回的原始 QoE 转成 SFT 训练时使用的 reward 尺度。"""
        reward = float(reward)
        if not self.uses_normalized_reward:
            return reward

        min_reward = float(self.min_reward)
        max_reward = float(self.max_reward)
        if self.clip_reward:
            reward = min(max_reward, max(min_reward, reward))
        return (reward - min_reward) / (max_reward - min_reward) / float(self.return_scale)

    def update(self, current_target_return, reward):
        """NetLLM 式更新：剩余目标回报 = 当前目标回报 - 本步收益。"""
        return float(current_target_return) - self.process_reward(reward)

    def describe(self):
        if self.uses_normalized_reward:
            return (
                f"target_return={self.target_return:.6f}, "
                f"reward=[{self.min_reward:.3f},{self.max_reward:.3f}], "
                f"return_scale={self.return_scale:g}, source={self.source}"
            )
        return f"target_return={self.target_return:.6f}, raw_reward_update, source={self.source}"


def load_return_to_go_processor(
    target_return=None,
    stats_path=TRAINING_STATS_PATH,
    clip_reward=True,
):
    """从 training_stats.json 读取 RTG 参数；读不到时保守回退。"""
    stats_path = Path(stats_path)
    stats = {}
    source = "default"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            source = str(stats_path)
        except (OSError, ValueError, TypeError):
            stats = {}

    resolved_target_return = (
        float(target_return)
        if target_return is not None
        else _first_float(
            stats,
            ("target_return", "max_return", "raw_max_return"),
            float(DEFAULT_TARGET_RETURN),
        )
    )
    min_reward = _first_float(stats, ("min_reward", "reward_min"))
    max_reward = _first_float(stats, ("max_reward", "reward_max"))
    return_scale = _first_float(stats, ("return_scale", "scale"), 1.0)

    return ReturnToGoProcessor(
        target_return=float(resolved_target_return),
        min_reward=min_reward,
        max_reward=max_reward,
        return_scale=max(float(return_scale), 1e-8),
        clip_reward=bool(clip_reward),
        source=source,
    )
