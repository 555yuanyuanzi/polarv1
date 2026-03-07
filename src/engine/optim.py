from __future__ import annotations

import torch

from src.config import AppConfig


def build_optimizer(model: torch.nn.Module, config: AppConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        betas=tuple(config.optim.betas),
        weight_decay=config.optim.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: AppConfig) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.optim.epochs,
        eta_min=config.scheduler.eta_min,
    )
