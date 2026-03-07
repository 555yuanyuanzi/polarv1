from __future__ import annotations

import copy

import torch

from src.utils.distributed import unwrap_model


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = copy.deepcopy(unwrap_model(model)).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source_state = unwrap_model(model).state_dict()
        target_state = self.module.state_dict()
        for key, target_value in target_state.items():
            source_value = source_state[key].detach()
            if torch.is_floating_point(target_value):
                target_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                target_value.copy_(source_value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state_dict)

    def to(self, device: torch.device) -> "ModelEMA":
        self.module.to(device)
        return self
