from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MaskedDWConv2d, make_direction_masks
from .naf import NAFBlock


class SharedExpert(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(NAFBlock(channels), NAFBlock(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class IsoResidualUnit(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.pwconv(self.dwconv(x)))


class IsoExpert(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.units = nn.Sequential(
            IsoResidualUnit(channels, kernel_size),
            IsoResidualUnit(channels, kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.units(x)


class DirectionalBasisBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.dwconv = MaskedDWConv2d(channels, kernel_size, mask, bias=True)
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pwconv(self.dwconv(x)))


class DirectionalBasisMixer(nn.Module):
    def __init__(self, channels: int, kernel_size: int, n_theta: int = 16) -> None:
        super().__init__()
        masks = make_direction_masks(kernel_size)
        self.beta_proj = nn.Conv2d(n_theta, 4, kernel_size=1, bias=True)
        self.branches = nn.ModuleList(
            [
                DirectionalBasisBranch(channels, kernel_size, masks["horizontal"]),
                DirectionalBasisBranch(channels, kernel_size, masks["vertical"]),
                DirectionalBasisBranch(channels, kernel_size, masks["main_diagonal"]),
                DirectionalBasisBranch(channels, kernel_size, masks["anti_diagonal"]),
            ]
        )

    def forward(self, x: torch.Tensor, d_local: torch.Tensor) -> torch.Tensor:
        beta = F.softmax(self.beta_proj(d_local), dim=1)
        beta_up = F.interpolate(beta, size=x.shape[-2:], mode="nearest")

        outputs = [branch(x) for branch in self.branches]
        mixed = sum(beta_up[:, index : index + 1] * branch_out for index, branch_out in enumerate(outputs))
        return x + mixed


class AnisoExpert(nn.Module):
    def __init__(self, channels: int, kernel_size: int, n_theta: int = 16) -> None:
        super().__init__()
        self.mixers = nn.ModuleList(
            [
                DirectionalBasisMixer(channels, kernel_size, n_theta=n_theta),
                DirectionalBasisMixer(channels, kernel_size, n_theta=n_theta),
            ]
        )

    def forward(self, x: torch.Tensor, d_local: torch.Tensor) -> torch.Tensor:
        for mixer in self.mixers:
            x = mixer(x, d_local)
        return x


class ShortIsoExpert(IsoExpert):
    def __init__(self, channels: int) -> None:
        super().__init__(channels, kernel_size=3)


class LongIsoExpert(IsoExpert):
    def __init__(self, channels: int) -> None:
        super().__init__(channels, kernel_size=7)


class ShortAnisoExpert(AnisoExpert):
    def __init__(self, channels: int, n_theta: int = 16) -> None:
        super().__init__(channels, kernel_size=5, n_theta=n_theta)


class LongAnisoExpert(AnisoExpert):
    def __init__(self, channels: int, n_theta: int = 16) -> None:
        super().__init__(channels, kernel_size=9, n_theta=n_theta)
