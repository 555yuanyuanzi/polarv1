from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d


def _inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class SEBranch(nn.Module):
    """
    单个频带分支上的轻量通道注意力。

    先通过全局平均池化聚合每个通道的全局响应，
    再预测 [0, 1] 范围内的通道权重，对该频带进行自适应重标定。
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.gelu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class FrequencyBandEnhancementBlock(nn.Module):
    """
    频带增强模块 FBEB。

    主要职责：
    - 对输入特征做归一化
    - 在 FFT 域中分解出 low / mid / high 三路频带
    - 用轻量 SE 模块分别重加权三路频带
    - 将三路频带重新融合后，以残差形式回注到主干特征

    输入 / 输出：
        x: [B, C, H, W]
        y: [B, C, H, W]

    说明：
    - 输入已经是规则的 CNN 特征图，因此不再需要 token flatten 或 reshape。
    - 频带划分不是硬阈值，而是可学习的软径向掩码。
    """

    def __init__(
        self,
        channels: int,
        init_r1: float = 0.22,
        init_r2: float = 0.58,
        init_tau: float = 0.05,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)

        # 先学习无约束参数，再映射到稳定、可解释的 r1/r2/tau 范围。
        self.p1 = nn.Parameter(torch.tensor(_inverse_sigmoid((init_r1 - 0.10) / 0.25), dtype=torch.float32))
        self.p2 = nn.Parameter(torch.tensor(_inverse_sigmoid((init_r2 - 0.50) / 0.25), dtype=torch.float32))
        self.p3 = nn.Parameter(torch.tensor(_inverse_sigmoid((init_tau - 0.03) / 0.07), dtype=torch.float32))

        self.se_low = SEBranch(channels, reduction=se_reduction)
        self.se_mid = SEBranch(channels, reduction=se_reduction)
        self.se_high = SEBranch(channels, reduction=se_reduction)

        self.fuse_in = nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True)
        self.fuse_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self._last_band_stats: dict[str, torch.Tensor] = {}
        self._last_band_visuals: dict[str, torch.Tensor] = {}

    def _build_radius_map(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Build a normalized radius map on the square FFT plane.

        The image is square in the spatial domain, but radius is defined in the
        shifted frequency plane relative to the spectrum center. The returned map
        has shape [1, 1, H, W] and values in [0, 1].
        """

        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(1e-6)
        return radius.view(1, 1, height, width)

    def get_band_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map unconstrained learnable parameters to bounded, interpretable values.

        r1: low / mid cutoff in [0.10, 0.35]
        r2: mid / high cutoff in [0.50, 0.75]
        tau: transition softness in [0.03, 0.10]
        """

        r1 = 0.10 + 0.25 * torch.sigmoid(self.p1)
        r2 = 0.50 + 0.25 * torch.sigmoid(self.p2)
        tau = 0.03 + 0.07 * torch.sigmoid(self.p3)
        return r1, r2, tau

    def _build_soft_masks(
        self,
        radius: torch.Tensor,
        r1: torch.Tensor,
        r2: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建 low / mid / high 三路归一化软径向掩码。

        low  : 更强调频谱中心区域
        mid  : 更强调 r1 到 r2 之间的环带
        high : 更强调外圈区域
        """

        low = torch.sigmoid((r1 - radius) / tau)
        high = torch.sigmoid((radius - r2) / tau)
        mid = torch.sigmoid((radius - r1) / tau) * torch.sigmoid((r2 - radius) / tau)
        denom = (low + mid + high).clamp_min(1e-6)
        return low / denom, mid / denom, high / denom

    def _to_visual_map(self, feature: torch.Tensor) -> torch.Tensor:
        """
        将多通道特征压成单通道可视化图。

        这里使用通道维绝对值均值，保留空间响应强弱，便于在训练时观察
        low / mid / high 三路频带在不同阶段的响应分布。
        """

        return feature.detach().abs().mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 先归一化，再做 FFT，避免学习到的频带划分过度依赖输入分布的绝对尺度。
        x_norm = self.norm(x)

        # FFT 强制用 float32 计算，提升数值稳定性。
        fft_x = torch.fft.fftshift(torch.fft.fft2(x_norm.float(), dim=(-2, -1)), dim=(-2, -1))

        radius = self._build_radius_map(x.shape[-2], x.shape[-1], x.device)
        r1, r2, tau = self.get_band_params()
        mask_low, mask_mid, mask_high = self._build_soft_masks(radius, r1, r2, tau)

        low_fft = fft_x * mask_low
        mid_fft = fft_x * mask_mid
        high_fft = fft_x * mask_high

        # 将每一路频带重新投回空间域，方便后续继续使用卷积式主干。
        low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real.to(dtype=x.dtype)
        mid = torch.fft.ifft2(torch.fft.ifftshift(mid_fft, dim=(-2, -1)), dim=(-2, -1)).real.to(dtype=x.dtype)
        high = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1)).real.to(dtype=x.dtype)

        low = self.se_low(low)
        mid = self.se_mid(mid)
        high = self.se_high(high)

        fused = torch.cat([low, mid, high], dim=1)
        fused = self.fuse_out(self.fuse_in(fused))

        self._last_band_stats = {
            "r1": r1.detach(),
            "r2": r2.detach(),
            "tau": tau.detach(),
            "low_energy": low.abs().mean().detach(),
            "mid_energy": mid.abs().mean().detach(),
            "high_energy": high.abs().mean().detach(),
        }
        self._last_band_visuals = {
            "low": self._to_visual_map(low),
            "mid": self._to_visual_map(mid),
            "high": self._to_visual_map(high),
        }
        return identity + self.gamma * fused

    def get_last_band_stats(self) -> dict[str, torch.Tensor]:
        return self._last_band_stats

    def get_last_band_visuals(self) -> dict[str, torch.Tensor]:
        return self._last_band_visuals
