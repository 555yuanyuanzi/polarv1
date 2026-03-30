from __future__ import annotations

import math

import torch

from src.config import AppConfig


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_iterations: int,
        warmup_iterations: int,
        warmup_start_lr: float,
        eta_min: float,
        last_epoch: int = -1,
    ) -> None:
        if total_iterations <= 0:
            raise ValueError("`total_iterations` must be positive.")
        if warmup_iterations < 0:
            raise ValueError("`warmup_iterations` must be >= 0.")
        if warmup_iterations >= total_iterations:
            raise ValueError("`warmup_iterations` must be smaller than `total_iterations`.")
        if warmup_start_lr < 0.0:
            raise ValueError("`warmup_start_lr` must be >= 0.")
        if eta_min < 0.0:
            raise ValueError("`eta_min` must be >= 0.")

        self.total_iterations = total_iterations
        self.warmup_iterations = warmup_iterations
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.scheduler_type = "warmup_cosine"
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_warmup_lr(self, base_lr: float, step: int) -> float:
        warmup_progress = step / self.warmup_iterations
        return self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_progress

    def _get_cosine_lr(self, base_lr: float, step: int) -> float:
        cosine_span = self.total_iterations - self.warmup_iterations
        cosine_step = min(max(step - self.warmup_iterations, 0), cosine_span)
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_span))
        return self.eta_min + (base_lr - self.eta_min) * cosine_term

    def get_lr(self) -> list[float]:
        step = min(max(self.last_epoch, 0), self.total_iterations)
        lrs: list[float] = []
        for base_lr in self.base_lrs:
            if self.warmup_iterations > 0 and step < self.warmup_iterations:
                lrs.append(self._get_warmup_lr(base_lr, step))
            else:
                lrs.append(self._get_cosine_lr(base_lr, step))
        return lrs

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        state = dict(state_dict)
        scheduler_type = state.get("scheduler_type")
        if scheduler_type in (None, "cosine"):
            state = {
                key: value
                for key, value in state.items()
                if key in {"last_epoch", "_step_count", "_last_lr", "base_lrs", "verbose", "_get_lr_called_within_step"}
            }
        state.pop("scheduler_type", None)
        super().load_state_dict(state)


def build_optimizer(model: torch.nn.Module, config: AppConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.lr,
        betas=tuple(config.optim.betas),
        weight_decay=config.optim.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: AppConfig) -> torch.optim.lr_scheduler._LRScheduler:
    return WarmupCosineScheduler(
        optimizer,
        total_iterations=config.optim.total_iterations,
        warmup_iterations=config.scheduler.warmup_iterations,
        warmup_start_lr=config.scheduler.warmup_start_lr,
        eta_min=config.scheduler.eta_min,
    )
