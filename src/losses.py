from __future__ import annotations

import math
from typing import Sequence

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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.fft2(prediction.float(), dim=(-2, -1))
        target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class PolarSpectralLoss(nn.Module):
    """
    极坐标频谱一致性损失。

    设计原则：
    - 不是逐点比较整张频谱，而是把频谱划成“半径环带 x 角度扇区”的方向-频率小 bin；
    - 每个小扇区只统计平均能量，让损失更关注分布结构而不是单个频点噪声；
    - 通过环带内归一化，减弱图像整体亮度和超低频主导，更多比较同尺度下的方向分布；
    - 作为去模糊主损失之外的辅助项使用，默认让中高频略高于低频。
    """

    def __init__(
        self,
        num_angle_bins: int = 8,
        num_radial_bins: int = 3,
        radial_min: float = 0.08,
        radial_max: float = 0.90,
        radial_weights: Sequence[float] | None = None,
        use_log_magnitude: bool = True,
        normalize_per_ring: bool = True,
        use_hann_window: bool = True,
        distance: str = "charbonnier",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_angle_bins <= 0:
            raise ValueError("`num_angle_bins` must be positive.")
        if num_radial_bins <= 0:
            raise ValueError("`num_radial_bins` must be positive.")
        if not (0.0 <= radial_min < radial_max <= 1.0):
            raise ValueError("Expected 0 <= radial_min < radial_max <= 1.")
        if distance not in {"l1", "l2", "charbonnier"}:
            raise ValueError("`distance` must be one of {'l1', 'l2', 'charbonnier'}.")

        self.num_angle_bins = num_angle_bins
        self.num_radial_bins = num_radial_bins
        self.radial_min = radial_min
        self.radial_max = radial_max
        self.use_log_magnitude = use_log_magnitude
        self.normalize_per_ring = normalize_per_ring
        self.use_hann_window = use_hann_window
        self.distance = distance
        self.eps = eps

        if radial_weights is None:
            if num_radial_bins == 3:
                weights = torch.tensor([0.5, 1.0, 1.2], dtype=torch.float32)
            else:
                weights = torch.ones(num_radial_bins, dtype=torch.float32)
        else:
            if len(radial_weights) != num_radial_bins:
                raise ValueError(
                    f"`radial_weights` length must match num_radial_bins={num_radial_bins}, "
                    f"got {len(radial_weights)}."
                )
            weights = torch.tensor(list(radial_weights), dtype=torch.float32)
        self.register_buffer("radial_weights", weights.view(1, num_radial_bins, 1), persistent=False)

    def _to_luma(self, image: torch.Tensor) -> torch.Tensor:
        """将 RGB 图转换成亮度图；若本来就是单通道则直接返回。"""

        if image.ndim != 4:
            raise ValueError(f"Expected image shape [B, C, H, W], got {tuple(image.shape)}.")
        if image.shape[1] == 1:
            return image
        if image.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {image.shape[1]}.")
        r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    def _build_hann_window(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
        wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
        return (wy[:, None] * wx[None, :]).view(1, 1, height, width)

    def _compute_log_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        """
        计算 log-magnitude 频谱。

        输出 shape:
        [B, 1, H, W]
        """

        x = self._to_luma(image).float()
        if self.use_hann_window:
            x = x * self._build_hann_window(x.shape[-2], x.shape[-1], x.device, x.dtype)
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        magnitude = spectrum.abs()
        if self.use_log_magnitude:
            magnitude = torch.log1p(magnitude)
        return magnitude

    def _build_polar_masks(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        构建 [num_radial_bins, num_angle_bins, H, W] 的 hard polar masks。

        说明：
        - 半径在归一化频率平面上定义，范围 [0, 1]
        - 角度使用 [0, pi) 即可表达实值图像幅度谱的无向方向
        """

        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(self.eps)

        angle = torch.atan2(yy, xx)
        angle = torch.remainder(angle, math.pi)

        radial_edges = torch.linspace(
            self.radial_min,
            self.radial_max,
            steps=self.num_radial_bins + 1,
            device=device,
            dtype=dtype,
        )
        angle_edges = torch.linspace(0.0, math.pi, steps=self.num_angle_bins + 1, device=device, dtype=dtype)

        masks: list[torch.Tensor] = []
        for radial_idx in range(self.num_radial_bins):
            radial_masks: list[torch.Tensor] = []
            r0 = radial_edges[radial_idx]
            r1 = radial_edges[radial_idx + 1]
            if radial_idx == self.num_radial_bins - 1:
                radial_mask = (radius >= r0) & (radius <= r1)
            else:
                radial_mask = (radius >= r0) & (radius < r1)

            for angle_idx in range(self.num_angle_bins):
                t0 = angle_edges[angle_idx]
                t1 = angle_edges[angle_idx + 1]
                if angle_idx == self.num_angle_bins - 1:
                    angle_mask = (angle >= t0) & (angle <= t1)
                else:
                    angle_mask = (angle >= t0) & (angle < t1)
                radial_masks.append((radial_mask & angle_mask).to(dtype))
            masks.append(torch.stack(radial_masks, dim=0))
        return torch.stack(masks, dim=0)

    def compute_sector_energies(self, image: torch.Tensor) -> torch.Tensor:
        """
        计算极坐标方向-频率小扇区的平均能量。

        返回:
        [B, num_radial_bins, num_angle_bins]
        """

        magnitude = self._compute_log_magnitude(image)
        masks = self._build_polar_masks(
            height=magnitude.shape[-2],
            width=magnitude.shape[-1],
            device=magnitude.device,
            dtype=magnitude.dtype,
        )
        mask_sum = masks.sum(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        masked_sum = (magnitude.unsqueeze(1) * masks.unsqueeze(0)).sum(dim=(-2, -1))
        energies = masked_sum / mask_sum.squeeze(-1).squeeze(-1).unsqueeze(0)

        if self.normalize_per_ring:
            # 在每个环带内做归一化，比较同尺度下不同方向的相对分布。
            energies = energies / energies.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return energies

    def _distance(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        if self.distance == "l1":
            return diff.abs()
        if self.distance == "l2":
            return diff.square()
        return torch.sqrt(diff.square() + self.eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_energy = self.compute_sector_energies(prediction)
        target_energy = self.compute_sector_energies(target)
        distance = self._distance(pred_energy, target_energy)
        weighted = distance * self.radial_weights.to(device=distance.device, dtype=distance.dtype)
        return weighted.mean()
