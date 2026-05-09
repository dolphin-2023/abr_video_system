import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from experience_dataset import ABRExperienceDataset
from checkpoint_utils import save_netllm_checkpoint
from network import NetLLMABR
from settings import ACTION_DIM, BASE_MODEL_PATH, MODEL_DIR, SFT_MODEL_PATH, TRAINING_STATS_PATH
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


def compute_action_class_weights(dataset, power=0.25, max_weight=3.0):
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


def train_sft(
    data_path="expert_data.npz",
    model_save_path=SFT_MODEL_PATH,
    epochs=20,
    accumulation_steps=8,
    lr=1e-4,
    weight_decay=1e-4,
    warmup_steps=200,
    grad_clip=0.25,
    early_stop_acc=0.85,
    early_stop_patience=2,
    min_epochs=2,
    max_length=20,
    sample_step=5,
    gamma=1.0,
    return_scale=1000.0,
    target_return_scale=1.0,
    class_weight_power=0.25,
    max_class_weight=3.0,
    validation_episodes=10,
    validation_split="train",
    validation_trace_dir=None,
    validation_sample_mode="stratified",
    validation_seed=20260507,
    validation_qoe_profile=None,
    validation_jump_limit=1,
    validation_min_delta=1e-3,
    validation_progress_interval=0,
    progress_interval=0,
    stats_save_path=TRAINING_STATS_PATH,
    history_save_path=None,
    seed=100003,
    base_model_path=BASE_MODEL_PATH,
):
    """Behavior-clone NetLLM from expert ABR trajectories.

    This follows the original NetLLM training shape more closely than the old
    full-episode loop: rewards are normalized, returns are scaled, and each
    trajectory is split into sliding windows.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

    print(f"[SFT] device={device}")
    print(f"[SFT] base_model={base_model_path}")
    print(
        f"[SFT] trajectories={stats['num_trajectories']} "
        f"windows={stats['num_samples']} "
        f"max_return={stats['max_return']:.6f} "
            f"target_return={stats['target_return']:.6f}"
    )
    if validation_trace_files:
        print(
            f"[SFT] validation=on traces={len(validation_trace_files)} "
            f"split={validation_split} qoe={validation_qoe_profile}"
        )
    else:
        print("[SFT] validation=off")

    model = NetLLMABR(
        model_name_or_path=str(base_model_path),
        action_dim=ACTION_DIM,
        lora_rank=128,
    )
    model.to(device)
    model.train()

    optimizer = optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    def warmup_lambda(step):
        return min((step + 1) / max(warmup_steps, 1), 1.0)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    class_weights, action_counts = compute_action_class_weights(
        dataset,
        power=class_weight_power,
        max_weight=max_class_weight,
    )
    if action_counts is not None:
        print(
            "[SFT] action_counts="
            + ", ".join(f"{idx}:{int(count)}" for idx, count in enumerate(action_counts))
        )
    if class_weights is not None:
        print(
            "[SFT] class_weights="
            + ", ".join(f"{idx}:{weight:.3f}" for idx, weight in enumerate(class_weights))
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    global_step = 0
    consecutive_good_epochs = 0
    best_validation_qoe = -float("inf")
    epochs_without_validation_improvement = 0
    validation_history = []
    for epoch in range(epochs):
        indices = np.random.permutation(len(dataset))
        total_batches = len(indices)
        epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch_idx, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            states = torch.tensor(sample["states"], dtype=torch.float32, device=device).unsqueeze(0)
            actions = (
                torch.tensor(sample["actions"], dtype=torch.long, device=device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            returns = (
                torch.tensor(sample["returns"], dtype=torch.float32, device=device)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            timesteps = torch.tensor(sample["timesteps"], dtype=torch.long, device=device).unsqueeze(0)

            with autocast_context(device):
                logits = model(states, actions, returns, timesteps)[0]
                targets = actions[0, :, 0]
                loss = criterion(logits, targets)
                scaled_loss = loss / accumulation_steps

            scaled_loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(indices)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

            epoch_loss += float(loss.detach().cpu())
            preds = torch.argmax(logits.detach(), dim=-1)
            correct_preds += int((preds == targets).sum().item())
            total_preds += int(targets.numel())

            if progress_interval and (batch_idx + 1) % int(progress_interval) == 0:
                completed = batch_idx + 1
                elapsed = time.time() - epoch_start
                eta = elapsed / max(completed, 1) * max(total_batches - completed, 0)
                running_loss = epoch_loss / max(completed, 1)
                running_acc = correct_preds / max(total_preds, 1)
                print(
                    f"[SFT] epoch={epoch + 1}/{epochs} "
                    f"step={completed}/{total_batches} "
                    f"({completed / max(total_batches, 1) * 100:.1f}%) "
                    f"loss={running_loss:.4f} "
                    f"acc={running_acc * 100:.2f}% "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta={format_duration(eta)}"
                )

        avg_loss = epoch_loss / max(len(indices), 1)
        accuracy = correct_preds / max(total_preds, 1)
        print(
            f"[SFT] epoch={epoch + 1}/{epochs} "
            f"loss={avg_loss:.4f} "
            f"acc={accuracy * 100:.2f}% "
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
                label="sft-validation",
                stats_path=stats_save_path,
            )
            validation_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_action_accuracy": accuracy,
                **summary,
            })
            print(
                f"[SFT] validation epoch={epoch + 1} "
                f"mean_qoe={summary['mean_qoe']:.3f} "
                f"p10_qoe={summary['p10_qoe']:.3f} "
                f"stall={summary['mean_stall_time']:.3f} "
                f"switch={summary['quality_switch_count']:.2f}"
            )

            if summary["mean_qoe"] > best_validation_qoe + float(validation_min_delta):
                best_validation_qoe = summary["mean_qoe"]
                epochs_without_validation_improvement = 0
                saved_path = save_netllm_checkpoint(
                    model,
                    model_save_path,
                    metadata={"stage": "sft", "epoch": epoch + 1},
                )
                print(
                    f"[SFT] saved new best validation model: "
                    f"mean_qoe={best_validation_qoe:.3f} -> {saved_path}"
                )
            else:
                epochs_without_validation_improvement += 1

            if (
                early_stop_patience > 0
                and epochs_without_validation_improvement >= early_stop_patience
                and epoch + 1 >= min_epochs
            ):
                print(
                    f"[SFT] early stop at epoch {epoch + 1}, "
                    f"best_validation_qoe={best_validation_qoe:.3f}, "
                    f"epochs_without_improvement={epochs_without_validation_improvement}"
                )
                break
        else:
            if accuracy >= early_stop_acc:
                consecutive_good_epochs += 1
            else:
                consecutive_good_epochs = 0

            if (
                early_stop_patience > 0
                and consecutive_good_epochs >= early_stop_patience
                and epoch + 1 >= min_epochs
            ):
                print(
                    f"[SFT] early stop at epoch {epoch + 1}, "
                    f"acc={accuracy * 100:.2f}%, "
                    f"consecutive_good_epochs={consecutive_good_epochs}"
                )
                break

    if validation_trace_files:
        history_path = Path(history_save_path) if history_save_path else MODEL_DIR / "sft_validation_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(validation_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[SFT] saved best validation weights: {model_save_path}")
        print(f"[SFT] saved validation history: {history_path}")
    else:
        saved_path = save_netllm_checkpoint(
            model,
            model_save_path,
            metadata={"stage": "sft"},
        )
        print(f"[SFT] saved weights: {saved_path}")
    print(f"[SFT] saved stats: {stats_save_path}")
    return model_save_path


if __name__ == "__main__":
    train_sft()
