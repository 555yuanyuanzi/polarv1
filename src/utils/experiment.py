from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ExperimentDirs:
    root: Path
    checkpoints: Path
    tensorboard: Path
    train_log: Path
    metrics_jsonl: Path
    resolved_config: Path


def create_experiment_dirs(output_root: str, experiment_name: str) -> ExperimentDirs:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_root) / experiment_name / timestamp
    checkpoints = root / "checkpoints"
    tensorboard = root / "tensorboard"
    checkpoints.mkdir(parents=True, exist_ok=True)
    tensorboard.mkdir(parents=True, exist_ok=True)
    return ExperimentDirs(
        root=root,
        checkpoints=checkpoints,
        tensorboard=tensorboard,
        train_log=root / "train.log",
        metrics_jsonl=root / "metrics.jsonl",
        resolved_config=root / "resolved_config.yaml",
    )


def load_experiment_dirs(root: str | Path) -> ExperimentDirs:
    root_path = Path(root)
    checkpoints = root_path / "checkpoints"
    tensorboard = root_path / "tensorboard"
    checkpoints.mkdir(parents=True, exist_ok=True)
    tensorboard.mkdir(parents=True, exist_ok=True)
    return ExperimentDirs(
        root=root_path,
        checkpoints=checkpoints,
        tensorboard=tensorboard,
        train_log=root_path / "train.log",
        metrics_jsonl=root_path / "metrics.jsonl",
        resolved_config=root_path / "resolved_config.yaml",
    )
