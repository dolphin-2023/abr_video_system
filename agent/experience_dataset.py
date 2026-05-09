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
    ):
        self.data_path = Path(data_path)
        self.max_length = int(max_length)
        self.sample_step = int(sample_step or max_length)
        self.gamma = float(gamma)
        self.return_scale = float(return_scale)

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
        self.reward_min = float(all_rewards.min())
        self.reward_max = float(all_rewards.max())
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
            "min_reward": self.reward_min,
            "max_reward": self.reward_max,
            "min_return": float(returns.min()),
            "mean_return": float(returns.mean()),
            "max_return": float(returns.max()),
            "mean_episode_return": float(np.mean(self.episode_returns)),
            "max_episode_return": float(np.max(self.episode_returns)),
        }
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

            if len(actions) < 2:
                continue

            normalized_rewards = (rewards - self.reward_min) / self.reward_span
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
