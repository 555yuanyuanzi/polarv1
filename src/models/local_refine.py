from __future__ import annotations

import torch
import torch.nn as nn

from .common import LayerNorm2d


class LocalRefinementBlock(nn.Module):
    """
    FBEB 之后使用的共享局部恢复块。

    职责：
    - 锐化局部边缘
    - 精修纹理细节
    - 保持结构简单、训练稳定

    结构：
    LN -> 3x3 DWConv -> 1x1 Conv -> GELU -> 5x5 DWConv -> 1x1 Conv -> residual

    该模块可以接收一张 importance map，[B, 1, H, W]。
    当提供 importance map 时，局部残差会被如下形式放大：

        1 + importance_strength * importance_map

    这样既能保持默认行为不变，也方便和 Importance Head 组合使用。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.dwconv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.pwconv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.dwconv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=True,
        )
        self.pwconv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def predict_delta(self, x: torch.Tensor) -> torch.Tensor:
        """预测局部修复残差，不直接与输入相加。"""

        y = self.norm(x)
        y = self.dwconv1(y)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.dwconv2(y)
        y = self.pwconv2(y)
        return self.gamma * y

    def forward(
        self,
        x: torch.Tensor,
        importance_map: torch.Tensor | None = None,
        importance_strength: float = 1.0,
    ) -> torch.Tensor:
        delta = self.predict_delta(x)
        if importance_map is not None:
            if importance_map.ndim != 4 or importance_map.shape[1] != 1:
                raise ValueError(
                    f"importance_map must have shape [B, 1, H, W], got {tuple(importance_map.shape)}."
                )
            if importance_map.shape[0] != x.shape[0] or importance_map.shape[-2:] != x.shape[-2:]:
                raise ValueError(
                    "importance_map must match the input batch and spatial dimensions: "
                    f"got x={tuple(x.shape)}, importance_map={tuple(importance_map.shape)}."
                )
            # 难区域 importance 更高，因此对应的局部残差会被放大。
            delta = delta * (1.0 + float(importance_strength) * importance_map.to(dtype=delta.dtype))
        return x + delta
