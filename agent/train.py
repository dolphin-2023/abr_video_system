import argparse
import json
import os
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from config_utils import dump_config, get_nested, load_config, resolve_path, section
from run_manager import create_run_paths, make_run_id
from settings import AGENT_DIR, BASE_MODEL_PATH, MODEL_DIR, TRAINING_STATS_PATH


DEFAULT_CONFIG = AGENT_DIR / "configs" / "train_local_selected_expert_window5_safety.yaml"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class TeeContext:
    def __init__(self, path, enabled=True):
        self.enabled = bool(enabled)
        self.path = Path(path)
        self.handle = None
        self.stdout_context = None
        self.stderr_context = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self.stdout_context = redirect_stdout(TeeStream(sys.stdout, self.handle))
        self.stderr_context = redirect_stderr(TeeStream(sys.stderr, self.handle))
        self.stdout_context.__enter__()
        self.stderr_context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        self.stderr_context.__exit__(exc_type, exc, tb)
        self.stdout_context.__exit__(exc_type, exc, tb)
        self.handle.close()
        return False


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def run_result_paths(paths):
    return {
        "run_dir": str(paths.root),
        "data_dir": str(paths.data),
        "checkpoints_dir": str(paths.checkpoints),
        "eval_dir": str(paths.eval),
        "log_file": str(paths.logs / "train.log"),
        "config": str(paths.config),
        "manifest": str(paths.manifest),
    }


def write_manifest(paths, updates):
    manifest = read_json(paths.manifest)
    manifest.update(updates)
    manifest["updated_at"] = now_iso()
    manifest["result_paths"] = run_result_paths(paths)
    write_json(paths.manifest, manifest)
    return manifest


def write_trace_list(path, trace_files):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(item) for item in trace_files) + ("\n" if trace_files else ""),
        encoding="utf-8",
    )


def sync_checkpoint_dir(source, target):
    source = Path(source)
    target = Path(target)
    if not source.exists() or not source.is_dir():
        return None
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def sync_file(source, target):
    source = Path(source)
    target = Path(target)
    if not source.exists() or not source.is_file():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def write_active_run(paths, config, stage):
    payload = {
        "updated_at": now_iso(),
        "stage": stage,
        "run_dir": str(paths.root),
        "stats_path": str(sft_stats_path(config, paths)),
        "models_dir": str(MODEL_DIR),
        "sft_model_path": str(agent_path(section(config, "sft").get("model_path", "models/netllm_sft"))),
        "offline_rl_model_path": str(agent_path(section(config, "offline_rl").get("model_path", "models/netllm_offline_rl"))),
        "rl_model_path": str(agent_path(section(config, "rl").get("model_path", "models/netllm_rl"))),
        "pensieve_model_path": str(agent_path(section(config, "pensieve").get("model_path", "models/pensieve_torch"))),
        "eval_dir": str(paths.eval),
        "manifest": str(paths.manifest),
    }
    write_json(MODEL_DIR / "active_run.json", payload)
    return payload


def agent_path(value):
    return resolve_path(value, base_dir=AGENT_DIR)


def model_path(config):
    return resolve_path(get_nested(config, ("env", "llm_path"), BASE_MODEL_PATH))


def common(config, key, default=None):
    return section(config, "common").get(key, default)


def validation(config, key, default=None):
    return section(config, "validation").get(key, default)


def bool_value(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def stage_in(requested, *stages):
    requested = str(requested).strip().lower()
    return requested == "all" or requested in {str(stage).strip().lower() for stage in stages}


def apply_environment(config):
    env_cfg = section(config, "env")
    runtime_cfg = section(config, "runtime")

    if env_cfg.get("llm_path"):
        os.environ["ABR_LLM_PATH"] = str(resolve_path(env_cfg["llm_path"]))
    if env_cfg.get("chunk_size_path"):
        os.environ["ABR_CHUNK_SIZE_PATH"] = str(agent_path(env_cfg["chunk_size_path"]))
    if env_cfg.get("video_id"):
        os.environ["ABR_VIDEO_ID"] = str(env_cfg["video_id"])
    if env_cfg.get("dash_root"):
        os.environ["ABR_DASH_ROOT"] = str(resolve_path(env_cfg["dash_root"]))
    if env_cfg.get("env_verbose") is not None:
        os.environ["ABR_ENV_VERBOSE"] = "1" if bool(env_cfg["env_verbose"]) else "0"
    if runtime_cfg.get("gradient_checkpointing") is not None:
        os.environ["ABR_GRADIENT_CHECKPOINTING"] = (
            "1" if bool(runtime_cfg["gradient_checkpointing"]) else "0"
        )

    torch_threads = runtime_cfg.get("torch_num_threads")
    if torch_threads:
        os.environ["OMP_NUM_THREADS"] = str(torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(torch_threads)
        import torch

        torch.set_num_threads(int(torch_threads))
        print(f"[Train] torch_num_threads={int(torch_threads)}")

    torch_interop_threads = runtime_cfg.get("torch_num_interop_threads")
    if torch_interop_threads:
        import torch

        torch.set_num_interop_threads(int(torch_interop_threads))
        print(f"[Train] torch_num_interop_threads={int(torch_interop_threads)}")


def build_validation_traces(config):
    from training_eval import build_eval_trace_files

    episodes = int(validation(config, "episodes", 10))
    if episodes == 0:
        return []
    trace_split = validation(config, "split", common(config, "trace_split", "train"))
    trace_dir = validation(config, "trace_dir", common(config, "trace_dir", None))
    trace_files = build_eval_trace_files(
        trace_split=trace_split,
        trace_dir=trace_dir,
        episodes=episodes,
        sample_mode=validation(config, "sample_mode", "stratified"),
        seed=int(validation(config, "seed", 20260507)),
        qoe_profile=common(config, "qoe_profile", "pensieve"),
    )
    if not trace_files:
        location = trace_dir or (
            f"simulator/traces/{trace_split} or "
            f"simulator/traces/real_world_split/{trace_split}"
        )
        raise FileNotFoundError(
            f"No network trace files found for validation split={trace_split!r} "
            f"at {location!r}. Generate traces with trace_tools before training."
        )
    return trace_files


def validation_uses_training_pool(config):
    train_split = str(common(config, "trace_split", "train")).strip().lower()
    valid_split = str(validation(config, "split", train_split)).strip().lower()
    train_dir = common(config, "trace_dir", None)
    valid_dir = validation(config, "trace_dir", train_dir)
    return train_split == valid_split and (train_dir or None) == (valid_dir or None)


def collect_data(config, paths, validation_traces):
    from data_collector import collect_expert_trajectories, collect_selected_expert_trajectories
    from merge_expert_data import merge_expert_datasets

    data_cfg = section(config, "data")
    exclude_files = validation_traces if validation_uses_training_pool(config) else []
    if exclude_files:
        print(f"[Train] holdout_validation_traces={len(exclude_files)}")

    collection_mode = str(data_cfg.get("collection_mode", "mixed")).strip().lower().replace("-", "_")
    if collection_mode in {"selected", "selected_expert", "selected_expert_pool"}:
        from abr_env import ABREnv
        from trace_utils import sample_trace_files

        item = section(data_cfg, "selected")
        save_path = paths.data / item.get("save_name", data_cfg.get("combined_name", "expert_selected.npz"))
        if data_cfg.get("reuse_existing", False) and save_path.exists():
            print(f"[Train] reuse selected dataset path={save_path}")
            return save_path

        trace_split = item.get("trace_split", common(config, "trace_split", "train"))
        trace_dir = item.get("trace_dir", common(config, "trace_dir", None))
        qoe_profile = item.get("qoe_profile", common(config, "qoe_profile", "pensieve"))
        env = ABREnv(
            trace_split=trace_split,
            trace_dir=trace_dir,
            random_start=False,
            qoe_profile=qoe_profile,
        )
        excluded_trace_names = {Path(t).name for t in exclude_files}
        trace_files = [
            trace_file for trace_file in sorted(env.trace_files)
            if Path(trace_file).name not in excluded_trace_names
        ]
        if not trace_files:
            location = trace_dir or (
                f"simulator/traces/{trace_split} or "
                f"simulator/traces/real_world_split/{trace_split}"
            )
            raise FileNotFoundError(
                f"No network trace files found for selected expert collection "
                f"split={trace_split!r} at {location!r}."
            )
        max_traces = int(item.get("max_traces", data_cfg.get("max_traces", 0)))
        if max_traces > 0 and len(trace_files) > max_traces:
            trace_files = sample_trace_files(
                trace_files,
                episodes=max_traces,
                mode=item.get("sample_mode", data_cfg.get("sample_mode", "stratified")),
                seed=int(item.get("seed", data_cfg.get("expert_seed", 20260507))),
            )
        policies_value = item.get("expert_policies", data_cfg.get("expert_policies", ["bola", "mpc", "bba"]))
        if isinstance(policies_value, str):
            expert_policies = [p.strip() for p in policies_value.split(",") if p.strip()]
        else:
            expert_policies = list(policies_value)

        print(
            f"[Train] collect selected dataset traces={len(trace_files)} "
            f"selection={item.get('selection_unit', 'window')} -> {save_path}"
        )
        collect_selected_expert_trajectories(
            trace_files=trace_files,
            save_path=save_path,
            gamma=float(item.get("gamma", data_cfg.get("gamma", 1.0))),
            trace_split=trace_split,
            trace_dir=trace_dir,
            qoe_profile=qoe_profile,
            expert_policies=expert_policies,
            bba_reservoir=float(item.get("bba_reservoir", data_cfg.get("bba_reservoir", 5.0))),
            bba_cushion=float(item.get("bba_cushion", data_cfg.get("bba_cushion", 15.0))),
            mpc_safety_factor=float(item.get("mpc_safety_factor", data_cfg.get("mpc_safety_factor", 0.9))),
            selection_unit=item.get("selection_unit", data_cfg.get("selection_unit", "window")),
            selection_window=int(item.get("selection_window", common(config, "window", 20))),
            selection_stride=int(item.get("selection_stride", common(config, "sample_step", 5))),
            selection_min_return=item.get("selection_min_return", data_cfg.get("selection_min_return", None)),
            negative_return_weight=float(item.get("negative_return_weight", data_cfg.get("negative_return_weight", 1.0))),
            positive_return_weight=float(item.get("positive_return_weight", data_cfg.get("positive_return_weight", 1.0))),
            progress_interval=int(item.get("progress_interval", data_cfg.get("progress_interval", 10))),
            seed=int(item.get("seed", data_cfg.get("expert_seed", 20260507))),
        )
        return save_path

    datasets = []
    for name in ("general", "high"):
        item = section(data_cfg, name)
        if not item.get("enabled", False):
            continue

        save_path = paths.data / item.get("save_name", f"expert_{name}.npz")
        if data_cfg.get("reuse_existing", False) and save_path.exists():
            print(f"[Train] reuse dataset={name} path={save_path}")
            datasets.append(save_path)
            continue

        print(f"[Train] collect dataset={name} episodes={item.get('episodes')} -> {save_path}")
        collect_expert_trajectories(
            num_episodes=int(item.get("episodes", data_cfg.get("episodes", 100))),
            save_path=save_path,
            gamma=float(item.get("gamma", data_cfg.get("gamma", 1.0))),
            trace_split=item.get("trace_split", common(config, "trace_split", "train")),
            trace_dir=item.get("trace_dir", common(config, "trace_dir", None)),
            stats_path=save_path.with_suffix(".stats.json"),
            qoe_profile=item.get("qoe_profile", common(config, "qoe_profile", "pensieve")),
            expert_policy=item.get("expert_policy", data_cfg.get("expert_policy", "mixed")),
            bba_ratio=float(item.get("bba_ratio", data_cfg.get("bba_ratio", 0.0))),
            bba_reservoir=float(item.get("bba_reservoir", data_cfg.get("bba_reservoir", 5.0))),
            bba_cushion=float(item.get("bba_cushion", data_cfg.get("bba_cushion", 15.0))),
            mpc_ratio=float(item.get("mpc_ratio", data_cfg.get("mpc_ratio", 0.35))),
            high_bitrate_ratio=float(item.get("high_bitrate_ratio", data_cfg.get("high_bitrate_ratio", 0.0))),
            expert_seed=int(item.get("expert_seed", data_cfg.get("expert_seed", 20260507))),
            mpc_safety_factor=float(item.get("mpc_safety_factor", data_cfg.get("mpc_safety_factor", 0.9))),
            high_mpc_safety_factor=float(item.get("high_mpc_safety_factor", data_cfg.get("high_mpc_safety_factor", 1.1))),
            high_mpc_min_buffer=float(item.get("high_mpc_min_buffer", data_cfg.get("high_mpc_min_buffer", 8.0))),
            high_mpc_max_stall=float(item.get("high_mpc_max_stall", data_cfg.get("high_mpc_max_stall", 0.25))),
            high_mpc_jump_limit=int(item.get("high_mpc_jump_limit", data_cfg.get("high_mpc_jump_limit", 1))),
            trace_filter=item.get("trace_filter", data_cfg.get("trace_filter", "all")),
            min_mean_throughput_kbps=float(
                item.get("min_mean_throughput_kbps", data_cfg.get("min_mean_throughput_kbps", 5000.0))
            ),
            high_bandwidth_percentile=float(
                item.get("high_bandwidth_percentile", data_cfg.get("high_bandwidth_percentile", 70.0))
            ),
            exclude_trace_files=exclude_files,
            progress_interval=int(data_cfg.get("progress_interval", 10)),
            workers=int(item.get("workers", data_cfg.get("workers", 1))),
        )
        datasets.append(save_path)

    if not datasets:
        configured_path = data_cfg.get("path")
        if configured_path:
            return Path(configured_path)
        raise RuntimeError("No data source is configured.")

    if len(datasets) == 1:
        return datasets[0]

    combined_path = paths.data / data_cfg.get("combined_name", "expert_combined.npz")
    merge_expert_datasets(
        input_paths=datasets,
        output_path=combined_path,
        stats_path=combined_path.with_suffix(".stats.json"),
    )
    return combined_path


def find_data_path(config, paths):
    data_cfg = section(config, "data")
    if data_cfg.get("path"):
        return Path(data_cfg["path"])
    combined = paths.data / data_cfg.get("combined_name", "expert_combined.npz")
    if combined.exists():
        return combined
    return combined


def sft_checkpoint_path(config, paths):
    sft = section(config, "sft")
    run_path = paths.checkpoints / sft.get("save_name", "sft")
    if run_path.exists():
        return run_path
    configured = sft.get("model_path")
    if configured:
        return agent_path(configured)
    return run_path


def sft_stats_path(config, paths):
    sft = section(config, "sft")
    run_path = paths.data / sft.get("stats_name", "training_stats.json")
    if run_path.exists():
        return run_path
    configured = sft.get("stats_path")
    if configured:
        return agent_path(configured)
    return run_path


def stats_sync_target(config):
    configured = section(config, "sft").get("stats_path")
    if configured:
        return agent_path(configured)
    return TRAINING_STATS_PATH


def offline_rl_checkpoint_path(config, paths):
    offline = section(config, "offline_rl")
    run_path = paths.checkpoints / offline.get("save_name", "offline_rl")
    if run_path.exists():
        return run_path
    configured = offline.get("model_path")
    if configured:
        return agent_path(configured)
    return run_path


def rl_initial_checkpoint_path(config, paths):
    offline = section(config, "offline_rl")
    if bool(offline.get("enabled", False)):
        candidate = offline_rl_checkpoint_path(config, paths)
        if Path(candidate).exists():
            return candidate
    return sft_checkpoint_path(config, paths)


def rl_checkpoint_path(config, paths):
    rl = section(config, "rl")
    run_path = paths.checkpoints / rl.get("save_name", "rl")
    if run_path.exists():
        return run_path
    configured = rl.get("model_path")
    if configured:
        return agent_path(configured)
    return run_path


def pensieve_checkpoint_path(config, paths):
    pensieve = section(config, "pensieve")
    run_path = paths.checkpoints / pensieve.get("save_name", "pensieve")
    if run_path.exists():
        return run_path
    configured = pensieve.get("model_path")
    if configured:
        return agent_path(configured)
    return run_path


def train_sft_stage(config, paths, data_path):
    from sft_trainer import train_sft

    sft = section(config, "sft")
    run_checkpoint = train_sft(
        data_path=data_path,
        model_save_path=paths.checkpoints / sft.get("save_name", "sft"),
        epochs=int(sft.get("epochs", 20)),
        accumulation_steps=int(sft.get("accumulation_steps", 8)),
        lr=float(sft.get("lr", 1e-4)),
        weight_decay=float(sft.get("weight_decay", 1e-4)),
        warmup_steps=int(sft.get("warmup_steps", 200)),
        grad_clip=float(sft.get("grad_clip", 0.25)),
        early_stop_acc=float(sft.get("early_stop_acc", 0.85)),
        early_stop_patience=int(sft.get("early_stop_patience", 2)),
        min_epochs=int(sft.get("min_epochs", 2)),
        max_length=int(common(config, "window", 20)),
        sample_step=int(common(config, "sample_step", 5)),
        return_scale=float(common(config, "return_scale", 1000.0)),
        reward_normalization=common(config, "reward_normalization", "minmax"),
        reward_percentile_low=float(common(config, "reward_percentile_low", 0.0)),
        reward_percentile_high=float(common(config, "reward_percentile_high", 100.0)),
        clip_normalized_reward=bool_value(common(config, "clip_normalized_reward", True), True),
        target_return_scale=float(common(config, "target_return_scale", 1.0)),
        context_len=int(common(config, "context_len", common(config, "window", 20))),
        class_weight_power=float(sft.get("class_weight_power", 0.25)),
        max_class_weight=float(sft.get("max_class_weight", 3.0)),
        batch_size=int(sft.get("batch_size", common(config, "batch_size", 8))),
        num_workers=int(sft.get("num_workers", common(config, "num_workers", 0))),
        validation_episodes=int(sft.get("validation_episodes", validation(config, "episodes", 10))),
        validation_batch_size=int(sft.get("validation_batch_size", common(config, "validation_batch_size", 8))),
        validation_split=sft.get("validation_split", validation(config, "split", common(config, "trace_split", "train"))),
        validation_trace_dir=sft.get("validation_trace_dir", validation(config, "trace_dir", common(config, "trace_dir", None))),
        validation_sample_mode=sft.get("validation_sample_mode", validation(config, "sample_mode", "stratified")),
        validation_seed=int(sft.get("validation_seed", validation(config, "seed", 20260507))),
        validation_qoe_profile=common(config, "qoe_profile", "pensieve"),
        validation_jump_limit=int(sft.get("validation_jump_limit", 1)),
        validation_progress_interval=int(sft.get("validation_progress_interval", 0)),
        progress_interval=int(sft.get("progress_interval", 100)),
        stats_save_path=paths.data / sft.get("stats_name", "training_stats.json"),
        history_save_path=paths.eval / "sft_validation_history.json",
        base_model_path=model_path(config),
    )
    configured = sft.get("model_path")
    if configured:
        synced_path = sync_checkpoint_dir(run_checkpoint, agent_path(configured))
        if synced_path is not None:
            print(f"[Train] synced SFT best checkpoint -> {synced_path}")
    return run_checkpoint


def train_offline_rl_stage(config, paths, data_path):
    from offline_rl_trainer import train_offline_rl

    offline = section(config, "offline_rl")
    init_from_sft = bool_value(offline.get("init_from_sft", True), True)
    run_checkpoint = train_offline_rl(
        data_path=data_path,
        sft_model_path=sft_checkpoint_path(config, paths) if init_from_sft else None,
        init_from_sft=init_from_sft,
        model_save_path=paths.checkpoints / offline.get("save_name", "offline_rl"),
        epochs=int(offline.get("epochs", 8)),
        batch_size=int(offline.get("batch_size", common(config, "batch_size", 1))),
        accumulation_steps=int(offline.get("accumulation_steps", 8)),
        lr=float(offline.get("lr", 5e-5)),
        weight_decay=float(offline.get("weight_decay", 1e-4)),
        warmup_steps=int(offline.get("warmup_steps", 200)),
        grad_clip=float(offline.get("grad_clip", 0.25)),
        max_length=int(common(config, "window", 20)),
        sample_step=int(common(config, "sample_step", 5)),
        return_scale=float(offline.get("return_scale", common(config, "return_scale", 1000.0))),
        reward_normalization=offline.get("reward_normalization", common(config, "reward_normalization", "minmax")),
        reward_percentile_low=float(offline.get("reward_percentile_low", common(config, "reward_percentile_low", 0.0))),
        reward_percentile_high=float(offline.get("reward_percentile_high", common(config, "reward_percentile_high", 100.0))),
        clip_normalized_reward=bool_value(
            offline.get("clip_normalized_reward", common(config, "clip_normalized_reward", True)),
            True,
        ),
        target_return_scale=float(offline.get("target_return_scale", common(config, "target_return_scale", 1.0))),
        context_len=int(offline.get("context_len", common(config, "context_len", common(config, "window", 20)))),
        class_weight_power=float(offline.get("class_weight_power", 0.1)),
        max_class_weight=float(offline.get("max_class_weight", 2.5)),
        high_return_weight=float(offline.get("high_return_weight", 0.35)),
        safety_penalty_weight=float(offline.get("safety_penalty_weight", 0.0)),
        safety_buffer_threshold=float(offline.get("safety_buffer_threshold", 8.0)),
        safety_safe_action_max=int(offline.get("safety_safe_action_max", 0)),
        train_scope=offline.get("train_scope", "non_plm_lora"),
        validation_episodes=int(offline.get("validation_episodes", validation(config, "episodes", 10))),
        validation_split=offline.get("validation_split", validation(config, "split", common(config, "trace_split", "train"))),
        validation_trace_dir=offline.get("validation_trace_dir", validation(config, "trace_dir", common(config, "trace_dir", None))),
        validation_sample_mode=offline.get("validation_sample_mode", validation(config, "sample_mode", "stratified")),
        validation_seed=int(offline.get("validation_seed", validation(config, "seed", 20260507))),
        validation_qoe_profile=common(config, "qoe_profile", "pensieve"),
        validation_jump_limit=int(offline.get("validation_jump_limit", 1)),
        validation_batch_size=int(offline.get("validation_batch_size", 8)),
        validation_min_delta=float(offline.get("validation_min_delta", 1e-3)),
        early_stop_patience=int(offline.get("early_stop_patience", 3)),
        min_epochs=int(offline.get("min_epochs", 2)),
        progress_interval=int(offline.get("progress_interval", 100)),
        validation_progress_interval=int(offline.get("validation_progress_interval", 0)),
        stats_save_path=paths.data / offline.get("stats_name", section(config, "sft").get("stats_name", "training_stats.json")),
        history_save_path=paths.eval / "offline_rl_validation_history.json",
        num_workers=int(offline.get("num_workers", common(config, "num_workers", 0))),
        base_model_path=model_path(config),
    )
    configured = offline.get("model_path")
    if configured:
        synced_path = sync_checkpoint_dir(run_checkpoint, agent_path(configured))
        if synced_path is not None:
            print(f"[Train] synced OfflineRL best checkpoint -> {synced_path}")
    return run_checkpoint


def train_pensieve_stage(config, paths):
    from pensieve_trainer import train_pensieve

    pensieve = section(config, "pensieve")
    run_checkpoint = train_pensieve(
        model_save_path=paths.checkpoints / pensieve.get("save_name", "pensieve"),
        pretrain_episodes=int(pensieve.get("pretrain_episodes", 300)),
        pretrain_lr=float(pensieve.get("pretrain_lr", pensieve.get("lr", 1e-4))),
        episodes=int(pensieve.get("episodes", 2000)),
        lr=float(pensieve.get("lr", 1e-4)),
        gamma=float(pensieve.get("gamma", 0.99)),
        value_coef=float(pensieve.get("value_coef", 0.5)),
        entropy_coef=float(pensieve.get("entropy_coef", 0.01)),
        grad_clip=float(pensieve.get("grad_clip", 0.5)),
        qoe_profile=common(config, "qoe_profile", "pensieve"),
        trace_split=pensieve.get("trace_split", common(config, "trace_split", "train")),
        trace_dir=pensieve.get("trace_dir", common(config, "trace_dir", None)),
        seed=int(pensieve.get("seed", 100003)),
        eval_interval=int(pensieve.get("eval_interval", 100)),
        validation_episodes=int(pensieve.get("validation_episodes", validation(config, "episodes", 30))),
        validation_split=pensieve.get("validation_split", validation(config, "split", common(config, "trace_split", "train"))),
        validation_trace_dir=pensieve.get("validation_trace_dir", validation(config, "trace_dir", common(config, "trace_dir", None))),
        validation_sample_mode=pensieve.get("validation_sample_mode", validation(config, "sample_mode", "stratified")),
        validation_seed=int(pensieve.get("validation_seed", validation(config, "seed", 20260507))),
        early_stop_patience=int(pensieve.get("early_stop_patience", 8)),
        min_episodes=int(pensieve.get("min_episodes", 400)),
        progress_interval=int(pensieve.get("progress_interval", 20)),
        history_save_path=paths.eval / "pensieve_validation_history.json",
    )
    configured = pensieve.get("model_path")
    if configured:
        synced_path = sync_checkpoint_dir(run_checkpoint, agent_path(configured))
        if synced_path is not None:
            print(f"[Train] synced Pensieve checkpoint -> {synced_path}")
    return run_checkpoint


def train_rl_stage(config, paths):
    from rl_trainer import train_rl

    rl = section(config, "rl")
    init_checkpoint = rl_initial_checkpoint_path(config, paths)
    print(f"[Train] RL init checkpoint={init_checkpoint}")
    run_checkpoint = train_rl(
        sft_model_path=init_checkpoint,
        rl_save_path=paths.checkpoints / rl.get("save_name", "rl"),
        episodes=int(rl.get("episodes", 200)),
        lr=float(rl.get("lr", 1e-5)),
        gamma=float(rl.get("gamma", 0.99)),
        target_return=rl.get("target_return", None),
        qoe_profile=common(config, "qoe_profile", "pensieve"),
        trace_split=rl.get("trace_split", common(config, "trace_split", "train")),
        trace_dir=rl.get("trace_dir", common(config, "trace_dir", None)),
        seed=int(rl.get("seed", 100003)),
        eval_interval=int(rl.get("eval_interval", 10)),
        validation_episodes=int(rl.get("validation_episodes", validation(config, "episodes", 10))),
        validation_split=rl.get("validation_split", validation(config, "split", common(config, "trace_split", "train"))),
        validation_trace_dir=rl.get("validation_trace_dir", validation(config, "trace_dir", common(config, "trace_dir", None))),
        validation_sample_mode=rl.get("validation_sample_mode", validation(config, "sample_mode", "stratified")),
        validation_seed=int(rl.get("validation_seed", validation(config, "seed", 20260507))),
        validation_jump_limit=int(rl.get("validation_jump_limit", 1)),
        validation_batch_size=int(rl.get("validation_batch_size", 8)),
        rollout_batch_size=int(rl.get("rollout_batch_size", 1)),
        early_stop_patience=int(rl.get("early_stop_patience", 3)),
        min_episodes=int(rl.get("min_episodes", 20)),
        action_jump_limit=int(rl.get("action_jump_limit", 1)),
        entropy_coef=float(rl.get("entropy_coef", 0.01)),
        kl_coef=float(rl.get("kl_coef", 0.0)),
        advantage_momentum=float(rl.get("advantage_momentum", 0.95)),
        advantage_clip=float(rl.get("advantage_clip", 5.0)),
        grad_clip=float(rl.get("grad_clip", 0.25)),
        train_scope=rl.get("train_scope", "action_head"),
        exclude_validation_traces=bool(rl.get("exclude_validation_traces", True)),
        validation_mean_tolerance=float(rl.get("validation_mean_tolerance", 5.0)),
        validation_p10_tolerance=float(rl.get("validation_p10_tolerance", 2.0)),
        validation_stall_tolerance=float(rl.get("validation_stall_tolerance", 0.25)),
        validation_switch_tolerance=float(rl.get("validation_switch_tolerance", 1.0)),
        selection_mean_weight=float(rl.get("selection_mean_weight", 1.0)),
        selection_p10_weight=float(rl.get("selection_p10_weight", 0.10)),
        selection_stall_weight=float(rl.get("selection_stall_weight", 0.25)),
        selection_switch_weight=float(rl.get("selection_switch_weight", 0.05)),
        save_candidates=bool(rl.get("save_candidates", True)),
        train_history_action_embedding=bool(rl.get("train_history_action_embedding", True)),
        stats_path=sft_stats_path(config, paths),
        history_save_path=paths.eval / "rl_validation_history.json",
        progress_interval=int(rl.get("progress_interval", 10)),
        base_model_path=model_path(config),
        clip_reward=bool(rl.get("clip_reward", True)),
    )
    configured = rl.get("model_path")
    if configured:
        synced_path = sync_checkpoint_dir(run_checkpoint, agent_path(configured))
        if synced_path is not None:
            print(f"[Train] synced RL best checkpoint -> {synced_path}")
    return run_checkpoint


def evaluate_stage(config, paths):
    from evaluate import run_evaluation

    eval_cfg = section(config, "eval")
    outputs = []
    for name, item in eval_cfg.items():
        if not isinstance(item, dict):
            continue
        item = section(eval_cfg, name)
        if not item.get("enabled", False):
            continue
        output_path = paths.eval / item.get("output_name", f"evaluation_{name}.json")
        payload = run_evaluation(
            trace_split=item.get("trace_split", "test"),
            trace_dir=item.get("trace_dir", None),
            qoe_profile=item.get("qoe_profile", common(config, "qoe_profile", "pensieve")),
            episodes=int(item.get("episodes", 20)),
            sample_mode=item.get("sample_mode", "stratified"),
            policies=item.get("policies", eval_cfg.get("policies", ["bola", "mpc", "random"])),
            seed=int(item.get("seed", 20260507)),
            device=item.get("device", None),
            progress_interval=int(item.get("progress_interval", 10)),
            video_id=get_nested(config, ("env", "video_id"), None),
            chunk_size_path=agent_path(get_nested(config, ("env", "chunk_size_path"), None)),
            output=output_path,
            sft_model_path=sft_checkpoint_path(config, paths),
            offline_rl_model_path=offline_rl_checkpoint_path(config, paths),
            rl_model_path=rl_checkpoint_path(config, paths),
            pensieve_model_path=pensieve_checkpoint_path(config, paths),
            stats_path=sft_stats_path(config, paths),
            base_model_path=model_path(config),
            env_verbose=bool(get_nested(config, ("env", "env_verbose"), False)),
        )
        outputs.append({
            "name": name,
            "output": str(output_path),
            "trace_split": payload.get("trace_split"),
            "trace_count": payload.get("trace_count"),
            "policies": [item.get("policy") for item in payload.get("policies", [])],
            "complete": payload.get("complete", True),
        })
    if outputs:
        write_json(paths.eval / "evaluation_index.json", {"updated_at": now_iso(), "outputs": outputs})
    return outputs


def run_pipeline(args):
    config = load_config(args.config)
    run_cfg = section(config, "run")
    run_id = args.run_id or run_cfg.get("id") or make_run_id(run_cfg.get("name", "train"))
    paths = create_run_paths(run_id=run_id, run_dir=args.run_dir, runs_root=run_cfg.get("root", "runs"))
    dump_config(config, paths.config)

    log_path = paths.logs / run_cfg.get("log_name", "train.log")
    with TeeContext(log_path, enabled=bool(run_cfg.get("log_to_file", True))):
        print(f"[Train] run_dir={paths.root}")
        print(f"[Train] config={paths.config}")
        print(f"[Train] stage={args.stage}")
        apply_environment(config)

        validation_traces = build_validation_traces(config)
        write_trace_list(paths.data / "validation_traces.txt", validation_traces)
        manifest = {
            "run_dir": str(paths.root),
            "stage": args.stage,
            "validation_trace_count": len(validation_traces),
            "validation_traces": [str(item) for item in validation_traces],
        }

        if args.stage == "plan":
            print("[Train] plan only; no training started.")
            write_manifest(paths, manifest)
            return

        data_path = find_data_path(config, paths)
        if stage_in(args.stage, "collect"):
            data_path = collect_data(config, paths, validation_traces)
            manifest["data_path"] = str(data_path)
            write_manifest(paths, manifest)

        if stage_in(args.stage, "sft"):
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Training data not found: {data_path}")
            print(f"[Train] SFT data={data_path}")
            train_sft_stage(config, paths, data_path)
            manifest["sft_model_path"] = str(sft_checkpoint_path(config, paths))
            synced_stats = sync_file(sft_stats_path(config, paths), stats_sync_target(config))
            if synced_stats is not None:
                print(f"[Train] synced training stats -> {synced_stats}")
            write_active_run(paths, config, "sft")
            write_manifest(paths, manifest)

        if stage_in(args.stage, "offline_rl") and bool(section(config, "offline_rl").get("enabled", False)):
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Offline RL data not found: {data_path}")
            print(f"[Train] OfflineRL data={data_path}")
            train_offline_rl_stage(config, paths, data_path)
            manifest["offline_rl_model_path"] = str(offline_rl_checkpoint_path(config, paths))
            manifest["sft_model_path"] = str(sft_checkpoint_path(config, paths))
            manifest["stats_path"] = str(sft_stats_path(config, paths))
            synced_stats = sync_file(sft_stats_path(config, paths), stats_sync_target(config))
            if synced_stats is not None:
                print(f"[Train] synced training stats -> {synced_stats}")
            write_active_run(paths, config, "offline_rl")
            write_manifest(paths, manifest)

        if stage_in(args.stage, "rl") and bool(section(config, "rl").get("enabled", False)):
            print("[Train] RL start")
            train_rl_stage(config, paths)
            manifest["rl_model_path"] = str(rl_checkpoint_path(config, paths))
            manifest["offline_rl_model_path"] = str(offline_rl_checkpoint_path(config, paths))
            manifest["sft_model_path"] = str(sft_checkpoint_path(config, paths))
            manifest["stats_path"] = str(sft_stats_path(config, paths))
            write_active_run(paths, config, "rl")
            write_manifest(paths, manifest)

        if stage_in(args.stage, "pensieve") and bool(section(config, "pensieve").get("enabled", False)):
            print("[Train] Pensieve baseline start")
            train_pensieve_stage(config, paths)
            manifest["pensieve_model_path"] = str(pensieve_checkpoint_path(config, paths))
            write_active_run(paths, config, "pensieve")
            write_manifest(paths, manifest)

        if stage_in(args.stage, "eval", "evaluate") and bool(section(config, "eval").get("enabled", True)):
            print("[Train] evaluation start")
            manifest["evaluation_outputs"] = evaluate_stage(config, paths)
            write_active_run(paths, config, "eval")
            write_manifest(paths, manifest)

        print(f"[Train] done: {paths.root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Config-driven training for the ABR NetLLM agent.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML/JSON training config.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["plan", "all", "collect", "sft", "offline_rl", "rl", "pensieve", "eval", "evaluate"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--run-id", default=None, help="Name for a run under agent/runs.")
    parser.add_argument("--run-dir", default=None, help="Existing or explicit run directory.")
    return parser.parse_args()


def main():
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
