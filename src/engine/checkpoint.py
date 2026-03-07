from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import AppConfig, config_to_dict
from src.engine.ema import ModelEMA
from src.utils.distributed import is_main_process, unwrap_model


def save_checkpoint(
    checkpoint_path: str | Path,
    config: AppConfig,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_psnr: float,
    best_ssim: float,
    main_process: bool,
    wandb_run_id: str = "",
) -> None:
    if not main_process:
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap_model(model).state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "wandb_run_id": wandb_run_id,
        "config": config_to_dict(config),
    }
    torch.save(state, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location=map_location)
    unwrap_model(model).load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    return {
        "epoch": int(state.get("epoch", 0)),
        "global_step": int(state.get("global_step", 0)),
        "best_psnr": float(state.get("best_psnr", float("-inf"))),
        "best_ssim": float(state.get("best_ssim", float("-inf"))),
        "wandb_run_id": str(state.get("wandb_run_id", "")),
        "config": state.get("config", {}),
    }
