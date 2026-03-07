from __future__ import annotations

import torch
import torch.nn as nn

from .common import LayerNorm2d, SimpleGate


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels, bias=True)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channels = ffn_expand * channels
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        return x + y * self.gamma
