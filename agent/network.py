import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, BitsAndBytesConfig


class EncoderNetwork(nn.Module):
    """Pensieve/Genet state encoder used by NetLLM."""

    def __init__(self, conv_size=4, num_bitrates=6, embed_dim=128):
        super().__init__()
        self.conv_size = conv_size
        self.num_bitrates = num_bitrates
        self.embed_dim = embed_dim

        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, embed_dim, conv_size),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(1, embed_dim, conv_size),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(1, embed_dim, num_bitrates),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.fc6 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())

    def forward(self, state):
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, 6, 6)

        f1 = self.fc1(state[:, 0:1, -1:]).reshape(batch_size, seq_len, -1)
        f2 = self.fc2(state[:, 1:2, -1:]).reshape(batch_size, seq_len, -1)
        f3 = self.conv3(state[:, 2:3, :]).reshape(batch_size, seq_len, -1)
        f4 = self.conv4(state[:, 3:4, :]).reshape(batch_size, seq_len, -1)
        f5 = self.conv5(state[:, 4:5, :self.num_bitrates]).reshape(batch_size, seq_len, -1)
        f6 = self.fc6(state[:, 5:6, -1:]).reshape(batch_size, seq_len, -1)
        return f1, f2, f3, f4, f5, f6


class NetLLMABR(nn.Module):
    """NetLLM-style Decision Transformer policy for ABR."""

    def __init__(
        self,
        model_name_or_path="Qwen/Qwen2.5-1.5B",
        past_k=6,
        action_dim=6,
        state_embed_dim=128,
        conv_size=4,
        lora_rank=128,
        lora_alpha=32,
        max_ep_len=100,
        load_in_4bit=True,
        gradient_checkpointing=None,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.past_k = past_k
        self.state_embed_dim = state_embed_dim

        print(f"[NetLLM] loading base model: {model_name_or_path}")
        model_kwargs = {
            "trust_remote_code": True,
        }
        if load_in_4bit and torch.cuda.is_available():
            model_kwargs.update({
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                "device_map": "auto",
            })
        else:
            model_kwargs.update({
                "device_map": "cpu",
                "torch_dtype": torch.float32,
            })

        base_llm = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        if hasattr(base_llm.config, "use_cache"):
            base_llm.config.use_cache = False
        d_model = base_llm.get_input_embeddings().weight.shape[1]
        self.plm_embed_size = d_model
        print(f"[NetLLM] hidden size: {d_model}")

        if lora_rank and lora_rank > 0:
            if gradient_checkpointing is None:
                gradient_checkpointing = os.environ.get(
                    "ABR_GRADIENT_CHECKPOINTING",
                    "0",
                ).strip().lower() in {"1", "true", "yes", "on"}
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.plm = get_peft_model(base_llm, lora_config)
            if gradient_checkpointing and hasattr(self.plm, "gradient_checkpointing_enable"):
                self.plm.gradient_checkpointing_enable()
            if hasattr(self.plm, "enable_input_require_grads"):
                self.plm.enable_input_require_grads()
        else:
            self.plm = base_llm

        self.state_encoder = EncoderNetwork(
            conv_size=conv_size,
            num_bitrates=action_dim,
            embed_dim=state_embed_dim,
        )

        conv_out_dim = state_embed_dim * (past_k - conv_size + 1)
        self.embed_state1 = nn.Linear(state_embed_dim, d_model)
        self.embed_state2 = nn.Linear(state_embed_dim, d_model)
        self.embed_state3 = nn.Linear(conv_out_dim, d_model)
        self.embed_state4 = nn.Linear(conv_out_dim, d_model)
        self.embed_state5 = nn.Linear(state_embed_dim, d_model)
        self.embed_state6 = nn.Linear(state_embed_dim, d_model)

        self.embed_return = nn.Linear(1, d_model)
        self.embed_action = nn.Linear(1, d_model)
        self.embed_timestep = nn.Embedding(max_ep_len + 1, d_model)
        self.embed_ln = nn.LayerNorm(d_model)
        self.action_head = nn.Linear(d_model, action_dim)

        self.states_dq, self.returns_dq, self.actions_dq = self.new_history_tuple()
        self.modules_except_plm = nn.ModuleList([
            self.state_encoder,
            self.embed_state1,
            self.embed_state2,
            self.embed_state3,
            self.embed_state4,
            self.embed_state5,
            self.embed_state6,
            self.embed_return,
            self.embed_action,
            self.embed_timestep,
            self.embed_ln,
            self.action_head,
        ])

    def forward(self, states, actions, returns, timesteps, attention_mask=None):
        batch_size, seq_len = states.shape[0], states.shape[1]
        device = states.device

        actions = actions.to(device)
        returns = returns.to(device)
        timesteps = timesteps.to(device)

        action_normalized = (actions.float() + 1.0) / self.action_dim
        time_emb = self.embed_timestep(timesteps)
        return_emb = self.embed_return(returns) + time_emb
        action_emb = self.embed_action(action_normalized) + time_emb

        f1, f2, f3, f4, f5, f6 = self.state_encoder(states)
        state_embs = [
            self.embed_state1(f1) + time_emb,
            self.embed_state2(f2) + time_emb,
            self.embed_state3(f3) + time_emb,
            self.embed_state4(f4) + time_emb,
            self.embed_state5(f5) + time_emb,
            self.embed_state6(f6) + time_emb,
        ]

        stacked_inputs = []
        action_positions = np.zeros(seq_len, dtype=np.int64)
        for idx in range(seq_len):
            step_tokens = torch.cat(
                (
                    return_emb[0, idx:idx + 1],
                    state_embs[0][0, idx:idx + 1],
                    state_embs[1][0, idx:idx + 1],
                    state_embs[2][0, idx:idx + 1],
                    state_embs[3][0, idx:idx + 1],
                    state_embs[4][0, idx:idx + 1],
                    state_embs[5][0, idx:idx + 1],
                    action_emb[0, idx:idx + 1],
                ),
                dim=0,
            )
            stacked_inputs.append(step_tokens)
            action_positions[idx] = (idx + 1) * 8 - 2

        stacked_inputs = torch.cat(stacked_inputs, dim=0).unsqueeze(0)
        stacked_inputs = stacked_inputs[:, -self.plm_embed_size:, :]
        stacked_inputs = self.embed_ln(stacked_inputs)

        if attention_mask is None:
            attention_mask = torch.ones(
                stacked_inputs.shape[0],
                stacked_inputs.shape[1],
                dtype=torch.long,
                device=device,
            )

        transformer_outputs = self.plm(
            inputs_embeds=stacked_inputs.to(self._plm_dtype()),
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = transformer_outputs.last_hidden_state
        action_hidden = hidden[:, action_positions, :].to(torch.float32)
        return self.action_head(action_hidden)

    def sample(
        self,
        state,
        target_return,
        timestep,
        history=None,
        deterministic=False,
        update_history=True,
        action_override=None,
    ):
        logits, return_emb, state_emb, time_emb = self.predict_logits(
            state=state,
            target_return=target_return,
            timestep=timestep,
            history=history,
        )

        if action_override is None:
            action, _ = self._sample_from_logits(logits, deterministic=deterministic)
        else:
            action = int(action_override)

        if update_history:
            self.append_history(
                history=history,
                return_emb=return_emb,
                state_emb=state_emb,
                time_emb=time_emb,
                action=action,
                device=state.device,
            )

        return int(action)

    def predict_logits(self, state, target_return, timestep, history=None):
        logits, return_emb, state_emb, time_emb = self.predict_logits_batch(
            states=state,
            target_returns=[target_return],
            timesteps=[timestep],
            histories=[history],
        )
        return logits[0], return_emb, state_emb, time_emb

    def predict_logits_batch(self, states, target_returns, timesteps, histories=None):
        device = states.device
        batch_size = states.shape[0]
        if histories is None:
            histories = [None] * batch_size
        if len(histories) != batch_size:
            raise ValueError("histories length must match states batch size")

        prev_embeds = []
        expected_tokens = None
        for history in histories:
            states_dq, returns_dq, actions_dq = self._history_tuple(history)
            prev_parts = []
            for idx in range(len(states_dq)):
                prev_parts.append(torch.cat(
                    (
                        returns_dq[idx].to(device),
                        states_dq[idx].to(device),
                        actions_dq[idx].to(device),
                    ),
                    dim=1,
                ))

            history_embeds = (
                torch.cat(prev_parts, dim=1)
                if any(part.shape[1] > 0 for part in prev_parts)
                else torch.zeros(1, 0, self.plm_embed_size, device=device)
            )
            if expected_tokens is None:
                expected_tokens = history_embeds.shape[1]
            elif history_embeds.shape[1] != expected_tokens:
                raise ValueError("batched histories must have the same token length")
            prev_embeds.append(history_embeds)

        prev_embeds = torch.cat(prev_embeds, dim=0)
        target_return = torch.as_tensor(
            target_returns,
            dtype=torch.float32,
            device=device,
        ).reshape(batch_size, 1, 1)
        timestep_t = torch.as_tensor(
            timesteps,
            dtype=torch.long,
            device=device,
        ).reshape(batch_size, 1)
        time_emb = self.embed_timestep(timestep_t)
        return_emb = self.embed_return(target_return) + time_emb

        f1, f2, f3, f4, f5, f6 = self.state_encoder(states)
        state_emb = torch.cat(
            [
                self.embed_state1(f1) + time_emb,
                self.embed_state2(f2) + time_emb,
                self.embed_state3(f3) + time_emb,
                self.embed_state4(f4) + time_emb,
                self.embed_state5(f5) + time_emb,
                self.embed_state6(f6) + time_emb,
            ],
            dim=1,
        )

        stacked = torch.cat((prev_embeds, return_emb, state_emb), dim=1)
        stacked = stacked[:, -self.plm_embed_size:, :]
        stacked = self.embed_ln(stacked)
        attention_mask = torch.ones(
            stacked.shape[0],
            stacked.shape[1],
            dtype=torch.long,
            device=device,
        )

        transformer_outputs = self.plm(
            inputs_embeds=stacked.to(self._plm_dtype()),
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = transformer_outputs.last_hidden_state
        logits = self.action_head(hidden[:, -1:, :].to(torch.float32)).squeeze(1)
        return logits, return_emb, state_emb, time_emb

    def new_history(self):
        states_dq, returns_dq, actions_dq = self.new_history_tuple()
        return {
            "states": states_dq,
            "returns": returns_dq,
            "actions": actions_dq,
        }

    def append_history(
        self,
        history,
        return_emb,
        state_emb,
        time_emb,
        action,
        device=None,
        detach_action=True,
    ):
        states_dq, returns_dq, actions_dq = self._history_tuple(history)
        if device is None:
            device = return_emb.device
        action_tensor = torch.tensor(
            [[[(int(action) + 1.0) / self.action_dim]]],
            dtype=torch.float32,
            device=device,
        )
        if detach_action:
            with torch.no_grad():
                action_emb = self.embed_action(action_tensor) + time_emb.detach()
            action_emb = action_emb.detach()
        else:
            # Keep gradients into embed_action during online RL. Return/state
            # history stays detached to avoid retaining the full episode graph.
            action_emb = self.embed_action(action_tensor) + time_emb.detach()

        returns_dq.append(return_emb.detach())
        states_dq.append(state_emb.detach())
        actions_dq.append(action_emb)

    def select_action(self, logits, deterministic=False):
        return self._sample_from_logits(logits, deterministic=deterministic)

    def new_history_tuple(self):
        return (
            deque([torch.zeros(1, 0, self.plm_embed_size)], maxlen=self.past_k),
            deque([torch.zeros(1, 0, self.plm_embed_size)], maxlen=self.past_k),
            deque([torch.zeros(1, 0, self.plm_embed_size)], maxlen=self.past_k),
        )

    def clear_dq(self, history=None):
        states_dq, returns_dq, actions_dq = self._history_tuple(history)
        states_dq.clear()
        returns_dq.clear()
        actions_dq.clear()
        states_dq.append(torch.zeros(1, 0, self.plm_embed_size))
        returns_dq.append(torch.zeros(1, 0, self.plm_embed_size))
        actions_dq.append(torch.zeros(1, 0, self.plm_embed_size))

    def _history_tuple(self, history):
        if history is None:
            return self.states_dq, self.returns_dq, self.actions_dq
        return history["states"], history["returns"], history["actions"]

    def _sample_from_logits(self, logits, deterministic=False):
        if deterministic:
            idx = int(torch.argmax(logits).item())
            probs = F.softmax(logits, dim=0).detach().cpu().numpy()
            return idx, float(np.log(probs[idx] + 1e-8))

        probs = F.softmax(logits, dim=0).detach().cpu().numpy()
        idx = random.choices(np.arange(len(probs)), weights=probs)[0]
        return int(idx), float(np.log(probs[idx] + 1e-8))

    def _plm_dtype(self):
        dtype = getattr(self.plm, "dtype", None)
        if dtype is not None:
            return dtype
        return next(self.plm.parameters()).dtype
