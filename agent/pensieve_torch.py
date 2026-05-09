import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from settings import ACTION_DIM, PAST_K, PENSIEVE_MODEL_PATH


CHECKPOINT_NAME = "pensieve_actor_critic.pt"
META_NAME = "checkpoint_meta.json"


class PensieveActorCritic(nn.Module):
    """PyTorch Pensieve-style actor-critic baseline for the local ABR environment."""

    def __init__(self, action_dim=ACTION_DIM, past_k=PAST_K, hidden_dim=128):
        super().__init__()
        self.action_dim = int(action_dim)
        self.past_k = int(past_k)
        self.hidden_dim = int(hidden_dim)

        self.quality_fc = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.buffer_fc = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.throughput_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.delay_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.next_size_conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=action_dim),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.remaining_fc = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())

        conv_width = past_k - 4 + 1
        merged_dim = hidden_dim * 4 + hidden_dim * conv_width * 2
        self.shared = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        if state.ndim == 2:
            state = state.unsqueeze(0)
        state = state.float()

        quality = self.quality_fc(state[:, 0, -1:].reshape(-1, 1))
        buffer = self.buffer_fc(state[:, 1, -1:].reshape(-1, 1))
        throughput = self.throughput_conv(state[:, 2:3, :])
        delay = self.delay_conv(state[:, 3:4, :])
        next_size = self.next_size_conv(state[:, 4:5, : self.action_dim])
        remaining = self.remaining_fc(state[:, 5, -1:].reshape(-1, 1))

        merged = torch.cat(
            (quality, buffer, throughput, delay, next_size, remaining),
            dim=1,
        )
        features = self.shared(merged)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def action_distribution(self, state):
        logits, value = self.forward(state)
        return torch.distributions.Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, state, deterministic=True):
        logits, value = self.forward(state)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(1, action.reshape(-1, 1)).squeeze(1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action[0].item()), float(log_prob[0].item()), float(value[0].item())


def _checkpoint_dir(path):
    path = Path(path)
    if path.suffix:
        return path.parent
    return path


def save_pensieve_checkpoint(model, path=PENSIEVE_MODEL_PATH, metadata=None):
    checkpoint_dir = _checkpoint_dir(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "action_dim": getattr(model, "action_dim", ACTION_DIM),
            "past_k": getattr(model, "past_k", PAST_K),
        },
        checkpoint_dir / CHECKPOINT_NAME,
    )
    payload = {"format": "pensieve_torch", "contains": ["actor_critic"]}
    if metadata:
        payload.update(metadata)
    (checkpoint_dir / META_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return checkpoint_dir


def load_pensieve_checkpoint(model, path=PENSIEVE_MODEL_PATH, map_location=None, strict=True):
    path = Path(path)
    checkpoint_path = path if path.is_file() else path / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        return None
    payload = torch.load(checkpoint_path, map_location=map_location)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=strict)
    return checkpoint_path


class PensieveTorchPolicy:
    name = "pensieve"

    def __init__(self, model_path=PENSIEVE_MODEL_PATH, device=None, deterministic=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PensieveActorCritic(action_dim=ACTION_DIM, past_k=PAST_K)
        loaded = load_pensieve_checkpoint(
            self.model,
            model_path,
            map_location=self.device,
            strict=False,
        )
        if loaded is None:
            raise FileNotFoundError(f"Pensieve checkpoint not found: {model_path}")
        print(f"[Evaluate] loaded pensieve weights: {loaded}")
        self.model.to(self.device)
        self.model.eval()
        self.deterministic = bool(deterministic)

    def reset(self, env):
        pass

    def act(self, state, env):
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0)
        action, _, _ = self.model.act(state_tensor, deterministic=self.deterministic)
        return max(0, min(env.num_bitrates - 1, int(action)))
