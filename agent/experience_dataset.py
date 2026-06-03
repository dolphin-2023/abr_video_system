import json
from pathlib import Path

import numpy as np


class ABRExperienceDataset:
    """Sliding-window dataset for NetLLM-style ABR behavior cloning."""

    def __init__(
        self,
        data_path="expert_data.npz",
        max_length=20,
        sample_step=None,
        gamma=1.0,
        return_scale=1000.0,
        reward_normalization="minmax",
        reward_percentile_low=0.0,
        reward_percentile_high=100.0,
        clip_normalized_reward=True,
    ):
        self.data_path = Path(data_path)
        self.max_length = int(max_length)
        self.sample_step = int(sample_step or max_length)
        self.gamma = float(gamma)
        self.return_scale = float(return_scale)
        self.reward_normalization = str(reward_normalization or "minmax").strip().lower()
        if self.reward_normalization == "robust":
            self.reward_normalization = "percentile"
        if self.reward_normalization not in {"minmax", "percentile"}:
            raise ValueError("reward_normalization must be 'minmax', 'robust', or 'percentile'")
        self.reward_percentile_low = float(reward_percentile_low)
        self.reward_percentile_high = float(reward_percentile_high)
        if not (0.0 <= self.reward_percentile_low < self.reward_percentile_high <= 100.0):
            raise ValueError("reward percentiles must satisfy 0 <= low < high <= 100")
        if self.reward_normalization == "minmax":
            self.reward_percentile_low = 0.0
            self.reward_percentile_high = 100.0
        self.clip_normalized_reward = bool(clip_normalized_reward)

        raw = np.load(self.data_path, allow_pickle=True)
        self.trajectories = list(raw["trajectories"])
        self.metadata = {}
        if "metadata" in raw.files:
            metadata_obj = raw["metadata"]
            if metadata_obj.shape == ():
                self.metadata = dict(metadata_obj.item())
        if not self.trajectories:
            raise ValueError(f"No trajectories found in {self.data_path}")

        all_rewards = np.concatenate(
            [np.asarray(t["rewards"], dtype=np.float32) for t in self.trajectories]
        )
        self.raw_reward_min = float(all_rewards.min())
        self.raw_reward_max = float(all_rewards.max())
        if self.reward_normalization == "percentile":
            self.reward_min = float(np.percentile(all_rewards, self.reward_percentile_low))
            self.reward_max = float(np.percentile(all_rewards, self.reward_percentile_high))
        else:
            self.reward_min = self.raw_reward_min
            self.reward_max = self.raw_reward_max
        self.reward_span = max(self.reward_max - self.reward_min, 1e-8)

        self.samples = []
        self.episode_returns = []
        self._build_samples()

        returns = np.concatenate([sample["returns"] for sample in self.samples])
        raw_return_arrays = [
            np.asarray(t.get("returns", []), dtype=np.float32)
            for t in self.trajectories
            if "returns" in t
        ]
        raw_returns = (
            np.concatenate(raw_return_arrays)
            if raw_return_arrays
            else np.asarray([], dtype=np.float32)
        )
        self.stats = {
            "data_path": str(self.data_path),
            "expert_policy": self.metadata.get("expert_policy"),
            "trace_split": self.metadata.get("trace_split"),
            "trace_dir": self.metadata.get("trace_dir"),
            "qoe_profile": self.metadata.get("qoe_profile"),
            "num_trajectories": len(self.trajectories),
            "num_samples": len(self.samples),
            "max_length": self.max_length,
            "sample_step": self.sample_step,
            "gamma": self.gamma,
            "return_scale": self.return_scale,
            "reward_normalization": self.reward_normalization,
            "reward_percentile_low": self.reward_percentile_low,
            "reward_percentile_high": self.reward_percentile_high,
            "clip_normalized_reward": self.clip_normalized_reward,
            "raw_min_reward": self.raw_reward_min,
            "raw_max_reward": self.raw_reward_max,
            "min_reward": self.reward_min,
            "max_reward": self.reward_max,
            "min_return": float(returns.min()),
            "mean_return": float(returns.mean()),
            "max_return": float(returns.max()),
            "mean_episode_return": float(np.mean(self.episode_returns)),
            "max_episode_return": float(np.max(self.episode_returns)),
        }
        sample_weights = np.asarray(
            [float(sample.get("sample_weight", 1.0)) for sample in self.samples],
            dtype=np.float32,
        )
        self.stats.update({
            "sample_weight_min": float(sample_weights.min()),
            "sample_weight_mean": float(sample_weights.mean()),
            "sample_weight_max": float(sample_weights.max()),
        })
        if raw_returns.size > 0:
            self.stats.update({
                "raw_min_return": float(raw_returns.min()),
                "raw_mean_return": float(raw_returns.mean()),
                "raw_max_return": float(raw_returns.max()),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def save_stats(self, path, target_return_scale=1.0):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = dict(self.stats)
        stats["target_return_scale"] = float(target_return_scale)
        stats["target_return"] = stats["max_return"] * float(target_return_scale)
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats

    def _build_samples(self):
        for trajectory in self.trajectories:
            states = np.asarray(trajectory["states"], dtype=np.float32)
            actions = np.asarray(trajectory["actions"], dtype=np.int64)
            rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
            timesteps = np.asarray(
                trajectory.get("timesteps", np.arange(len(actions))),
                dtype=np.int64,
            )
            sample_weight = float(trajectory.get("sample_weight", 1.0))

            if len(actions) < 2:
                continue

            normalized_rewards = self._normalize_rewards(rewards)
            returns = self._discount_returns(normalized_rewards)
            self.episode_returns.append(float(returns[0]))

            max_start = max(0, len(actions) - self.max_length)
            starts = list(range(0, max_start + 1, self.sample_step))
            if starts[-1] != max_start:
                starts.append(max_start)

            for start in starts:
                end = min(start + self.max_length, len(actions))
                if end - start < 2:
                    continue
                self.samples.append({
                    "states": states[start:end],
                    "actions": actions[start:end],
                    "returns": returns[start:end],
                    "timesteps": timesteps[start:end],
                    "sample_weight": sample_weight,
                })

        if not self.samples:
            raise ValueError(
                "No training windows were created. Reduce max_length or collect more data."
            )

    def _discount_returns(self, rewards):
        returns = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            running = float(rewards[idx]) + self.gamma * running
            returns[idx] = running / self.return_scale
        return returns

    def _normalize_rewards(self, rewards):
        normalized = (rewards - self.reward_min) / self.reward_span
        if self.clip_normalized_reward:
            normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32)


def collate_abr_samples(batch):
    """Collate a variable-length ABR sample list into a right-padded batch.

    Returns tensors ready for NetLLMABR.forward():
        states  [B, T_max, 6, 6]
        actions [B, T_max]
        returns [B, T_max]
        timesteps [B, T_max]
        mask    [B, T_max]  1.0 for real tokens, 0.0 for padding
    """
    max_len = max(s["states"].shape[0] for s in batch)
    B = len(batch)

    states = np.zeros((B, max_len, 6, 6), dtype=np.float32)
    actions = np.zeros((B, max_len), dtype=np.int64)
    returns = np.zeros((B, max_len), dtype=np.float32)
    timesteps = np.zeros((B, max_len), dtype=np.int64)
    mask = np.zeros((B, max_len), dtype=np.float32)

    for i, s in enumerate(batch):
        T = s["states"].shape[0]
        states[i, :T] = s["states"]
        actions[i, :T] = s["actions"]
        returns[i, :T] = s["returns"]
        timesteps[i, :T] = s["timesteps"]
        mask[i, :T] = 1.0

    return states, actions, returns, timesteps, mask


def collate_abr_samples_with_weights(batch):
    """Collate ABR samples and include one scalar training weight per sample."""
    states, actions, returns, timesteps, mask = collate_abr_samples(batch)
    sample_weights = np.asarray(
        [float(s.get("sample_weight", 1.0)) for s in batch],
        dtype=np.float32,
    )
    return states, actions, returns, timesteps, mask, sample_weights
