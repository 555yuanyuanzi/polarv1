from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(prediction: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(prediction, target).item()
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


def calculate_ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> float:
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    def gaussian(size: int, sigma: float) -> torch.Tensor:
        values = torch.tensor([np.exp(-((x - size // 2) ** 2) / (2 * sigma**2)) for x in range(size)], dtype=torch.float32)
        return values / values.sum()

    def create_window(size: int, channel: int) -> torch.Tensor:
        one_d = gaussian(size, 1.5).unsqueeze(1)
        two_d = one_d @ one_d.t()
        return two_d.expand(channel, 1, size, size).contiguous()

    channel = prediction.size(1)
    window = create_window(window_size, channel).to(prediction.device)
    mu1 = F.conv2d(prediction, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(prediction * prediction, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(prediction * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()
