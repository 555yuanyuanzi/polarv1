from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((prediction - target) ** 2 + self.eps))


class FrequencyLoss(nn.Module):
    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.fft2(prediction.float(), dim=(-2, -1))
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
        return self.weight * F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
