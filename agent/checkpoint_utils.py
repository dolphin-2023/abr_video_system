import json
from pathlib import Path

import torch


MODULES_EXCEPT_PLM_NAME = "modules_except_plm.bin"
CHECKPOINT_META_NAME = "checkpoint_meta.json"
LEGACY_SUFFIXES = (".pth", ".pt", ".bin")


def _save_dir(path):
    """新 checkpoint 使用目录；旧 .pth 路径会自动映射到同名目录。"""
    path = Path(path)
    if path.suffix.lower() in LEGACY_SUFFIXES:
        return path.with_suffix("")
    return path


def _candidate_paths(path):
    path = Path(path)
    candidates = [path]
    if path.suffix.lower() in LEGACY_SUFFIXES:
        candidates.append(path.with_suffix(""))
    else:
        candidates.extend(path.with_suffix(suffix) for suffix in LEGACY_SUFFIXES)
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def is_slim_checkpoint(path):
    path = Path(path)
    return path.is_dir() and (path / MODULES_EXCEPT_PLM_NAME).exists()


def resolve_checkpoint_path(path):
    """优先找新目录 checkpoint，找不到再兼容旧 .pth/.pt/.bin 文件。"""
    for candidate in _candidate_paths(path):
        if candidate.exists():
            return candidate
    return Path(path)


def save_netllm_checkpoint(model, path, metadata=None):
    """保存瘦身版 checkpoint。

    大模型 base 权重不保存，只保存 LoRA adapter 和 ABR 任务相关小模块。
    这样能避免每次 SFT/RL 都写出 3GB 以上的完整 state_dict。
    """
    checkpoint_dir = _save_dir(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "plm") and hasattr(model.plm, "save_pretrained"):
        model.plm.save_pretrained(str(checkpoint_dir))
    else:
        torch.save(model.state_dict(), checkpoint_dir / "model.bin")

    if hasattr(model, "modules_except_plm"):
        torch.save(
            model.modules_except_plm.state_dict(),
            checkpoint_dir / MODULES_EXCEPT_PLM_NAME,
        )

    payload = {
        "format": "netllm_slim",
        "contains": ["plm_adapter", "modules_except_plm"],
    }
    if metadata:
        payload.update(metadata)
    (checkpoint_dir / CHECKPOINT_META_NAME).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return checkpoint_dir


def _load_plm_adapter(model, checkpoint_dir):
    if not hasattr(model, "plm") or not hasattr(model.plm, "load_adapter"):
        return

    try:
        try:
            model.plm.load_adapter(str(checkpoint_dir), adapter_name="default")
        except TypeError:
            model.plm.load_adapter(str(checkpoint_dir), "default")
        return
    except Exception:
        pass

    if _load_adapter_state_into_existing(model, checkpoint_dir):
        return

    adapter_name = "checkpoint"
    try:
        try:
            model.plm.load_adapter(str(checkpoint_dir), adapter_name=adapter_name)
        except TypeError:
            model.plm.load_adapter(str(checkpoint_dir), adapter_name)
    except Exception:
        return

    if hasattr(model.plm, "set_adapter"):
        model.plm.set_adapter(adapter_name)


def _load_adapter_state_into_existing(model, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    adapter_state = None
    safetensors_path = checkpoint_dir / "adapter_model.safetensors"
    torch_path = checkpoint_dir / "adapter_model.bin"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file

            adapter_state = load_file(str(safetensors_path))
        except Exception:
            adapter_state = None
    if adapter_state is None and torch_path.exists():
        adapter_state = torch.load(torch_path, map_location="cpu")
    if adapter_state is None:
        return False

    model.plm.load_state_dict(adapter_state, strict=False)
    return True


def load_netllm_checkpoint(model, path, map_location=None, strict=False):
    """加载新瘦身 checkpoint；如果不存在，则自动兼容旧完整 .pth。"""
    resolved = resolve_checkpoint_path(path)
    if not resolved.exists():
        return None

    if is_slim_checkpoint(resolved):
        _load_plm_adapter(model, resolved)
        modules_state = torch.load(
            resolved / MODULES_EXCEPT_PLM_NAME,
            map_location=map_location,
        )
        model.modules_except_plm.load_state_dict(modules_state, strict=strict)
        return resolved

    state = torch.load(resolved, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return resolved
