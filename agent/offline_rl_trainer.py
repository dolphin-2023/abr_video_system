import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from checkpoint_utils import load_netllm_checkpoint, save_netllm_checkpoint
from experience_dataset import ABRExperienceDataset
from network import NetLLMABR
from rl_trainer import configure_rl_trainable_scope
from settings import ACTION_DIM, BASE_MODEL_PATH, OFFLINE_RL_MODEL_PATH, SFT_MODEL_PATH, TRAINING_STATS_PATH
from training_eval import build_eval_trace_files, evaluate_model_policy


def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def compute_action_class_weights(dataset, power=0.1, max_weight=2.5):
    power = float(power)
    if power <= 0.0:
        return None, None

    actions = np.concatenate([
        np.asarray(trajectory["actions"], dtype=np.int64)
        for trajectory in dataset.trajectories
    ])
    counts = np.bincount(actions, minlength=ACTION_DIM).astype(np.float32)
    positive = counts > 0
    weights = np.zeros(ACTION_DIM, dtype=np.float32)
    if np.any(positive):
        mean_positive = float(np.mean(counts[positive]))
        weights[positive] = (mean_positive / np.maximum(counts[positive], 1.0)) ** power
        weights[positive] = np.minimum(weights[positive], float(max_weight))
        weights[positive] /= max(float(np.mean(weights[positive])), 1e-8)
    return weights, counts


def sample_weight(sample, high_return_threshold, high_return_weight):
    if float(high_return_weight) <= 0.0:
        return 1.0
    returns = np.asarray(sample["returns"], dtype=np.float32)
    if returns.size and float(np.max(returns)) >= float(high_return_threshold):
        return 1.0 + float(high_return_weight)
    return 1.0


def train_offline_rl(
    data_path,
    sft_model_path=SFT_MODEL_PATH,
    model_save_path=OFFLINE_RL_MODEL_PATH,
    epochs=8,
    accumulation_steps=8,
    lr=5e-5,
    weight_decay=1e-4,
    warmup_steps=200,
    grad_clip=0.25,
    max_length=20,
    sample_step=5,
    gamma=1.0,
    return_scale=1000.0,
    target_return_scale=1.0,
    class_weight_power=0.1,
    max_class_weight=2.5,
    high_return_weight=0.35,
    train_scope="non_plm_lora",
    validation_episodes=30,
    validation_split="train",
    validation_trace_dir=None,
    validation_sample_mode="stratified",
    validation_seed=20260507,
    validation_qoe_profile="pensieve",
    validation_jump_limit=1,
    validation_batch_size=8,
    validation_min_delta=1e-3,
    early_stop_patience=3,
    min_epochs=2,
    progress_interval=100,
    validation_progress_interval=0,
    stats_save_path=TRAINING_STATS_PATH,
    history_save_path=None,
    seed=100003,
    base_model_path=BASE_MODEL_PATH,
):
    """Offline return-conditioned fine-tuning initialized from the SFT checkpoint."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = ABRExperienceDataset(
        data_path=data_path,
        max_length=max_length,
        sample_step=sample_step,
        gamma=gamma,
        return_scale=return_scale,
    )
    stats_save_path = Path(stats_save_path or TRAINING_STATS_PATH)
    stats = dataset.save_stats(stats_save_path, target_return_scale=target_return_scale)
    validation_qoe_profile = validation_qoe_profile or stats.get("qoe_profile") or "pensieve"
    validation_split = validation_split or stats.get("trace_split") or "train"
    validation_trace_files = build_eval_trace_files(
        trace_split=validation_split,
        trace_dir=validation_trace_dir,
        episodes=int(validation_episodes),
        sample_mode=validation_sample_mode,
        seed=int(validation_seed),
        qoe_profile=validation_qoe_profile,
    )

    returns = np.asarray([float(np.max(sample["returns"])) for sample in dataset.samples], dtype=np.float32)
    high_return_threshold = float(np.percentile(returns, 75)) if returns.size else 0.0

    print(f"[OfflineRL] device={device}")
    print(f"[OfflineRL] base_model={base_model_path}")
    print(f"[OfflineRL] sft_model={sft_model_path}")
    print(
        f"[OfflineRL] trajectories={stats['num_trajectories']} "
        f"windows={stats['num_samples']} target_return={stats['target_return']:.6f}"
    )

    model = NetLLMABR(
        model_name_or_path=str(base_model_path),
        action_dim=ACTION_DIM,
        lora_rank=128,
    )
    loaded_path = load_netllm_checkpoint(model, sft_model_path, map_location=device, strict=False)
    if loaded_path is None:
        raise FileNotFoundError(f"SFT checkpoint not found: {sft_model_path}")
    print(f"[OfflineRL] loaded SFT checkpoint: {loaded_path}")
    model.to(device)
    model.train()

    scope, trainable, total = configure_rl_trainable_scope(model, train_scope)
    print(f"[OfflineRL] train_scope={scope} trainable={trainable}/{total}")

    optimizer = optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    def warmup_lambda(step):
        return min((step + 1) / max(int(warmup_steps), 1), 1.0)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    class_weights, action_counts = compute_action_class_weights(
        dataset,
        power=class_weight_power,
        max_weight=max_class_weight,
    )
    if action_counts is not None:
        print(
            "[OfflineRL] action_counts="
            + ", ".join(f"{idx}:{int(count)}" for idx, count in enumerate(action_counts))
        )
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    best_validation_qoe = -float("inf")
    epochs_without_improvement = 0
    validation_history = []
    global_step = 0

    for epoch in range(1, int(epochs) + 1):
        indices = np.random.permutation(len(dataset))
        epoch_start = time.time()
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, sample_idx in enumerate(indices):
            sample = dataset[int(sample_idx)]
            states = torch.tensor(sample["states"], dtype=torch.float32, device=device).unsqueeze(0)
            actions = (
                torch.tensor(sample["actions"], dtype=torch.long, device=device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            returns_t = (
                torch.tensor(sample["returns"], dtype=torch.float32, device=device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            timesteps = torch.tensor(sample["timesteps"], dtype=torch.long, device=device).unsqueeze(0)

            with autocast_context(device):
                logits = model(states, actions, returns_t, timesteps)[0]
                targets = actions[0, :, 0]
                per_token_loss = criterion(logits, targets)
                weight = sample_weight(sample, high_return_threshold, high_return_weight)
                loss = per_token_loss.mean() * float(weight)
                scaled_loss = loss / int(accumulation_steps)

            scaled_loss.backward()

            if ((batch_idx + 1) % int(accumulation_steps) == 0) or (batch_idx + 1 == len(indices)):
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    float(grad_clip),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

            epoch_loss += float(loss.detach().cpu())
            preds = torch.argmax(logits.detach(), dim=-1)
            correct_preds += int((preds == targets).sum().item())
            total_preds += int(targets.numel())

            if progress_interval and (batch_idx + 1) % int(progress_interval) == 0:
                elapsed = time.time() - epoch_start
                completed = batch_idx + 1
                eta = elapsed / max(completed, 1) * max(len(indices) - completed, 0)
                print(
                    f"[OfflineRL] epoch={epoch}/{epochs} "
                    f"step={completed}/{len(indices)} "
                    f"loss={epoch_loss / max(completed, 1):.4f} "
                    f"acc={correct_preds / max(total_preds, 1) * 100:.2f}% "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
                )

        avg_loss = epoch_loss / max(len(indices), 1)
        accuracy = correct_preds / max(total_preds, 1)
        print(
            f"[OfflineRL] epoch={epoch}/{epochs} "
            f"loss={avg_loss:.4f} acc={accuracy * 100:.2f}% "
            f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
        )

        if validation_trace_files:
            summary, _ = evaluate_model_policy(
                model=model,
                trace_files=validation_trace_files,
                trace_split=validation_split,
                trace_dir=validation_trace_dir,
                qoe_profile=validation_qoe_profile,
                target_return=stats["target_return"],
                device=device,
                jump_limit=validation_jump_limit,
                progress_interval=validation_progress_interval,
                label="offline-rl-validation",
                stats_path=stats_save_path,
                batch_size=validation_batch_size,
            )
            validation_history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_action_accuracy": accuracy,
                **summary,
            })
            print(
                f"[OfflineRL] validation epoch={epoch} "
                f"mean_qoe={summary['mean_qoe']:.3f} "
                f"p10_qoe={summary['p10_qoe']:.3f} "
                f"stall={summary['mean_stall_time']:.3f}"
            )
            if summary["mean_qoe"] > best_validation_qoe + float(validation_min_delta):
                best_validation_qoe = summary["mean_qoe"]
                epochs_without_improvement = 0
                saved_path = save_netllm_checkpoint(
                    model,
                    model_save_path,
                    metadata={"stage": "offline_rl", "epoch": epoch},
                )
                print(f"[OfflineRL] saved best checkpoint -> {saved_path}")
            else:
                epochs_without_improvement += 1

            if (
                early_stop_patience > 0
                and epoch >= int(min_epochs)
                and epochs_without_improvement >= int(early_stop_patience)
            ):
                print(
                    f"[OfflineRL] early stop at epoch={epoch}, "
                    f"best_validation_qoe={best_validation_qoe:.3f}"
                )
                break

    if not validation_trace_files:
        save_netllm_checkpoint(model, model_save_path, metadata={"stage": "offline_rl"})

    if history_save_path:
        history_path = Path(history_save_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(validation_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[OfflineRL] saved validation history: {history_path}")
    print(f"[OfflineRL] saved stats: {stats_save_path}")
    return model_save_path


if __name__ == "__main__":
    train_offline_rl(data_path="expert_data.npz")
