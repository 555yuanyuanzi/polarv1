from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import window_partition


class LocalPolarPrior(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        n_theta: int = 16,
        n_r: int = 8,
        polar_proj_dim: int = 32,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.n_theta = n_theta
        self.n_r = n_r
        self.proj = nn.Conv2d(channels, polar_proj_dim, kernel_size=1, bias=True)
        self.register_buffer("polar_grid", self._build_polar_grid(n_r, n_theta), persistent=False)

    @staticmethod
    def _build_polar_grid(n_r: int, n_theta: int) -> torch.Tensor:
        radius = torch.linspace(0.0, 1.0, steps=n_r)
        theta = torch.linspace(0.0, 2.0 * math.pi, steps=n_theta + 1)[:-1]
        rr, tt = torch.meshgrid(radius, theta, indexing="ij")
        grid_x = rr * torch.cos(tt)
        grid_y = rr * torch.sin(tt)
        return torch.stack([grid_x, grid_y], dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        features = self.proj(x)
        windows, hg, wg = window_partition(features, self.window_size)

        spectrum = torch.fft.fft2(windows.float(), dim=(-2, -1), norm="ortho")
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        magnitude = torch.log1p(spectrum.abs())

        grid = self.polar_grid.to(device=magnitude.device, dtype=magnitude.dtype)
        grid = grid.unsqueeze(0).expand(magnitude.shape[0], -1, -1, -1)
        polar = F.grid_sample(
            magnitude,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        direction_logits = polar.mean(dim=(1, 2))
        direction_dist = F.softmax(direction_logits, dim=-1)
        d_local = direction_dist.view(batch_size, hg, wg, self.n_theta).permute(0, 3, 1, 2).contiguous()

        entropy = -(d_local * d_local.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        confidence = 1.0 - entropy / math.log(self.n_theta)
        confidence = confidence.clamp_(0.0, 1.0)

        return d_local.to(dtype=x.dtype), confidence.to(dtype=x.dtype)
