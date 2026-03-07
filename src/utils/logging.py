from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency in some environments
    SummaryWriter = None


def create_logger(log_path: str | Path, is_main: bool = True) -> logging.Logger:
    logger = logging.getLogger(f"polarformer_v1_{Path(log_path).stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    if not is_main:
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def create_tensorboard_writer(log_dir: str | Path, enabled: bool, is_main: bool):
    if not enabled or not is_main or SummaryWriter is None:
        return None
    return SummaryWriter(log_dir=str(log_dir))


def create_wandb_run(
    *,
    enabled: bool,
    is_main: bool,
    project: str,
    entity: str,
    mode: str,
    config: dict[str, Any],
    run_name: str,
    run_dir: str | Path,
    run_id: str = "",
):
    if not enabled or not is_main:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("W&B logging is enabled but `wandb` is not installed.") from exc

    kwargs = {
        "project": project,
        "config": config,
        "name": run_name,
        "dir": str(run_dir),
        "sync_tensorboard": True,
        "mode": mode,
    }
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = "allow"
    if entity:
        kwargs["entity"] = entity
    return wandb.init(**kwargs)
