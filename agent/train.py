import argparse
import json
import os
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from config_utils import dump_config, get_nested, load_config, resolve_path, section
from run_manager import create_run_paths, make_run_id
from settings import AGENT_DIR, BASE_MODEL_PATH


DEFAULT_CONFIG = AGENT_DIR / "configs" / "train_v3.yaml"


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


def agent_path(value):
    return resolve_path(value, base_dir=AGENT_DIR)


def model_path(config):
    return resolve_path(get_nested(config, ("env", "llm_path"), BASE_MODEL_PATH))


def common(config, key, default=None):
    return section(config, "common").get(key, default)


def validation(config, key, default=None):
    return section(config, "validation").get(key, default)


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
    return build_eval_trace_files(
        trace_split=validation(config, "split", common(config, "trace_split", "train")),
        trace_dir=validation(config, "trace_dir", common(config, "trace_dir", None)),
        episodes=episodes,
        sample_mode=validation(config, "sample_mode", "stratified"),
        seed=int(validation(config, "seed", 20260507)),
        qoe_profile=common(config, "qoe_profile", "pensieve"),
    )


def validation_uses_training_pool(config):
    train_split = str(common(config, "trace_split", "train")).strip().lower()
    valid_split = str(validation(config, "split", train_split)).strip().lower()
    train_dir = common(config, "trace_dir", None)
    valid_dir = validation(config, "trace_dir", train_dir)
    return train_split == valid_split and (train_dir or None) == (valid_dir or None)


def collect_data(config, paths, validation_traces):
    from data_collector import collect_expert_trajectories
    from merge_expert_data import merge_expert_datasets

    data_cfg = section(config, "data")
    exclude_files = validation_traces if validation_uses_training_pool(config) else []
    if exclude_files:
        print(f"[Train] holdout_validation_traces={len(exclude_files)}")

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
            mpc_ratio=float(item.get("mpc_ratio", data_cfg.get("mpc_ratio", 0.35))),
            high_bitrate_ratio=float(item.get("high_bitrate_ratio", data_cfg.get("high_bitrate_ratio", 0.0))),
            expert_seed=int(item.get("expert_seed", data_cfg.get("expert_seed", 20260507))),
            mpc_safety_factor=float(item.get("mpc_safety_factor", data_cfg.get("mpc_safety_factor", 0.9))),
            high_mpc_safety_factor=float(item.get("high_mpc_safety_factor", data_cfg.get("high_mpc_safety_factor", 1.1))),
            high_mpc_min_buffer=float(item.get("high_mpc_min_buffer", data_cfg.get("high_mpc_min_buffer", 8.0))),
            trace_filter=item.get("trace_filter", data_cfg.get("trace_filter", "all")),
            min_mean_throughput_kbps=float(
                item.get("min_mean_throughput_kbps", data_cfg.get("min_mean_throughput_kbps", 5000.0))
            ),
            high_bandwidth_percentile=float(
                item.get("high_bandwidth_percentile", data_cfg.get("high_bandwidth_percentile", 70.0))
            ),
            exclude_trace_files=exclude_files,
            progress_interval=int(data_cfg.get("progress_interval", 10)),
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


def rl_checkpoint_path(config, paths):
    rl = section(config, "rl")
    run_path = paths.checkpoints / rl.get("save_name", "rl")
    if run_path.exists():
        return run_path
    configured = rl.get("model_path")
    if configured:
        return agent_path(configured)
    return run_path


def train_sft_stage(config, paths, data_path):
    from sft_trainer import train_sft

    sft = section(config, "sft")
    return train_sft(
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
        target_return_scale=float(common(config, "target_return_scale", 1.0)),
        class_weight_power=float(sft.get("class_weight_power", 0.25)),
        max_class_weight=float(sft.get("max_class_weight", 3.0)),
        validation_episodes=int(sft.get("validation_episodes", validation(config, "episodes", 10))),
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


def train_rl_stage(config, paths):
    from rl_trainer import train_rl

    rl = section(config, "rl")
    sft = section(config, "sft")
    run_checkpoint = train_rl(
        sft_model_path=sft_checkpoint_path(config, paths),
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
    sft = section(config, "sft")
    rl = section(config, "rl")
    for name in ("smoke", "final"):
        item = section(eval_cfg, name)
        if not item.get("enabled", False):
            continue
        run_evaluation(
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
            output=paths.eval / item.get("output_name", f"evaluation_{name}.json"),
            sft_model_path=sft_checkpoint_path(config, paths),
            rl_model_path=rl_checkpoint_path(config, paths),
            stats_path=sft_stats_path(config, paths),
            base_model_path=model_path(config),
            env_verbose=bool(get_nested(config, ("env", "env_verbose"), False)),
        )


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
            write_json(paths.manifest, manifest)
            return

        data_path = find_data_path(config, paths)
        if stage_in(args.stage, "collect"):
            data_path = collect_data(config, paths, validation_traces)
            manifest["data_path"] = str(data_path)
            write_json(paths.manifest, manifest)

        if stage_in(args.stage, "sft"):
            if not Path(data_path).exists():
                raise FileNotFoundError(f"Training data not found: {data_path}")
            print(f"[Train] SFT data={data_path}")
            train_sft_stage(config, paths, data_path)
            manifest["sft_model_path"] = str(sft_checkpoint_path(config, paths))
            write_json(paths.manifest, manifest)

        if stage_in(args.stage, "rl") and bool(section(config, "rl").get("enabled", False)):
            print("[Train] RL start")
            train_rl_stage(config, paths)
            manifest["rl_model_path"] = str(rl_checkpoint_path(config, paths))
            manifest["sft_model_path"] = str(sft_checkpoint_path(config, paths))
            manifest["stats_path"] = str(sft_stats_path(config, paths))
            write_json(paths.manifest, manifest)

        if stage_in(args.stage, "eval", "evaluate") and bool(section(config, "eval").get("enabled", True)):
            print("[Train] evaluation start")
            evaluate_stage(config, paths)
            write_json(paths.manifest, manifest)

        print(f"[Train] done: {paths.root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Config-driven training for the ABR NetLLM agent.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML/JSON training config.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["plan", "all", "collect", "sft", "rl", "eval", "evaluate"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--run-id", default=None, help="Name for a run under agent/runs.")
    parser.add_argument("--run-dir", default=None, help="Existing or explicit run directory.")
    return parser.parse_args()


def main():
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
