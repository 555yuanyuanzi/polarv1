from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import LayerNorm2d


class RawGuidancePyramid(nn.Module):
    """
    为 decoder2 和 decoder3 生成轻量 raw-image guidance。

    对输入模糊图 [B, 3, H, W]，输出：
    - decoder2 guidance: [B, G, H/2, W/2]
    - decoder3 guidance: [B, G, H/4, W/4]
    """

    def __init__(self, inp_channels: int = 3, guide_channels: int = 16) -> None:
        super().__init__()
        self.guide_channels = guide_channels
        self.stem = nn.Conv2d(inp_channels, guide_channels, kernel_size=3, padding=1, bias=True)
        self.down2 = nn.Conv2d(guide_channels, guide_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.down3 = nn.Conv2d(guide_channels, guide_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        g1 = self.act(self.stem(x))
        g2 = self.act(self.down2(g1))
        g3 = self.act(self.down3(g2))
        return {"decoder2": g2, "decoder3": g3}


class RestorationImportanceHead(nn.Module):
    """
    预测单通道的恢复重要性图。

    这个 head 不负责重建图像，而是输出一张空间控制图，
    后续用于调制 LocalRefinementBlock 的局部修复强度。

    输入：
        stage_feat   : 当前 decoder stage 的特征，[B, C, H, W]
        fbeb_feat    : FBEB 输出的频带增强特征，[B, C, H, W]
        raw_guidance : 从原始模糊图提取的浅层 guidance，[B, G, H, W]

    输出：
        importance_map: [B, 1, H, W]，取值范围 [0, 1]
    """

    def __init__(
        self,
        channels: int,
        guidance_channels: int,
        hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = channels if hidden_channels is None else hidden_channels
        in_channels = channels * 2

        self.norm = LayerNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        # 由 raw guidance 生成一张单通道的 gating 图，作为显式的 raw-guided 调制。
        self.raw_gate = nn.Conv2d(guidance_channels, 1, kernel_size=3, padding=1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=True,
        )
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self._last_stats: dict[str, torch.Tensor] = {}
        self._last_visuals: dict[str, torch.Tensor] = {}

    def forward(
        self,
        stage_feat: torch.Tensor,
        fbeb_feat: torch.Tensor,
        raw_guidance: torch.Tensor,
    ) -> torch.Tensor:
        if stage_feat.shape != fbeb_feat.shape:
            raise ValueError(
                f"stage_feat and fbeb_feat must have the same shape, got {stage_feat.shape} and {fbeb_feat.shape}."
            )
        if raw_guidance.ndim != 4 or raw_guidance.shape[0] != stage_feat.shape[0]:
            raise ValueError(
                f"raw_guidance must have shape [B, G, H, W] with matching batch size, got {tuple(raw_guidance.shape)}."
            )
        if raw_guidance.shape[-2:] != stage_feat.shape[-2:]:
            raise ValueError(
                "raw_guidance must match the stage spatial size: "
                f"got stage={tuple(stage_feat.shape)}, raw_guidance={tuple(raw_guidance.shape)}."
            )

        h = torch.cat([stage_feat, fbeb_feat], dim=1)
        h = self.norm(h)
        h = self.conv1(h)
        h = self.act(h)
        raw_gate = torch.sigmoid(self.raw_gate(raw_guidance))
        # 用 raw guidance 生成的空间门控图放大或抑制中间特征响应。
        h = h * (1.0 + raw_gate)
        h = self.dwconv(h)
        importance = torch.sigmoid(self.conv2(h))

        self._last_stats = {
            "importance/mean": importance.mean().detach(),
            "importance/std": importance.std(unbiased=False).detach(),
            "importance/high_ratio": (importance > 0.6).float().mean().detach(),
            "importance/raw_gate_mean": raw_gate.mean().detach(),
        }
        self._last_visuals = {
            "importance": importance.detach(),
            "raw_gate": raw_gate.detach(),
        }
        return importance

    def get_last_importance_stats(self) -> dict[str, torch.Tensor]:
        return self._last_stats

    def get_last_visuals(self) -> dict[str, torch.Tensor]:
        return self._last_visuals
