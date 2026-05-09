from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from settings import AGENT_DIR


RUNS_DIR = AGENT_DIR / "runs"


@dataclass
class RunPaths:
    root: Path
    data: Path
    checkpoints: Path
    eval: Path
    logs: Path
    config: Path
    manifest: Path


def make_run_id(name=None):
    prefix = str(name or "train").strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def create_run_paths(run_id=None, run_dir=None, runs_root=None):
    if run_dir:
        root = Path(run_dir)
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
    else:
        root = Path(runs_root) if runs_root else RUNS_DIR
        if not root.is_absolute():
            root = (AGENT_DIR / root).resolve()
        root = root / (run_id or make_run_id())

    paths = RunPaths(
        root=root,
        data=root / "data",
        checkpoints=root / "checkpoints",
        eval=root / "eval",
        logs=root / "logs",
        config=root / "config.yaml",
        manifest=root / "manifest.json",
    )
    for path in (paths.data, paths.checkpoints, paths.eval, paths.logs):
        path.mkdir(parents=True, exist_ok=True)
    return paths
