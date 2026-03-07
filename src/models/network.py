from __future__ import annotations

import torch
import torch.nn as nn

from .lpeb import LPEB
from .naf import NAFBlock
from .restormer_lite import RestormerLiteBlock


def _build_naf_stage(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)])


def _build_restormer_stage(channels: int, num_blocks: int, ffn_expansion: float) -> nn.Sequential:
    num_heads = {96: 2, 192: 4, 384: 8}[channels]
    return nn.Sequential(
        *[RestormerLiteBlock(channels, num_heads=num_heads, ffn_expansion=ffn_expansion) for _ in range(num_blocks)]
    )


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=True),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Fuse(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        merged_channels = channels * 2
        self.in_proj = nn.Conv2d(merged_channels, merged_channels, kernel_size=1, bias=True)
        self.block = NAFBlock(merged_channels)
        self.out_proj = nn.Conv2d(merged_channels, merged_channels, kernel_size=1, bias=True)

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([enc, dec], dim=1)
        fused = self.out_proj(self.block(self.in_proj(fused)))
        left, right = fused.chunk(2, dim=1)
        return left + right


class PolarFormer(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        enc_blocks: tuple[int, int, int] = (3, 4, 6),
        bottleneck_base_blocks: int = 3,
        dec3_base_blocks: int = 2,
        dec2_base_blocks: int = 2,
        dec1_base_blocks: int = 3,
        polar_window: int = 8,
        n_theta: int = 16,
        n_r: int = 8,
        polar_proj_dim: int = 32,
        router_hidden: int = 32,
        router_topk: int = 2,
        restormer_ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        if dim != 48:
            raise ValueError("V1 keeps `dim=48` fixed.")
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=True)

        self.encoder1 = _build_naf_stage(dim, enc_blocks[0])
        self.down1 = Downsample(dim, dim * 2)
        self.encoder2 = _build_naf_stage(dim * 2, enc_blocks[1])
        self.down2 = Downsample(dim * 2, dim * 4)
        self.encoder3 = _build_naf_stage(dim * 4, enc_blocks[2])
        self.down3 = Downsample(dim * 4, dim * 8)

        self.bottleneck_base = _build_restormer_stage(dim * 8, bottleneck_base_blocks, restormer_ffn_expansion)
        self.bottleneck_lpeb = LPEB(
            channels=dim * 8,
            window_size=polar_window,
            n_theta=n_theta,
            n_r=n_r,
            polar_proj_dim=polar_proj_dim,
            router_hidden=router_hidden,
            router_topk=router_topk,
            ffn_expansion=restormer_ffn_expansion,
        )

        self.up3 = Upsample(dim * 8, dim * 4)
        self.fuse3 = Fuse(dim * 4)
        self.decoder3_base = _build_restormer_stage(dim * 4, dec3_base_blocks, restormer_ffn_expansion)
        self.decoder3_lpeb = LPEB(
            channels=dim * 4,
            window_size=polar_window,
            n_theta=n_theta,
            n_r=n_r,
            polar_proj_dim=polar_proj_dim,
            router_hidden=router_hidden,
            router_topk=router_topk,
            ffn_expansion=restormer_ffn_expansion,
        )

        self.up2 = Upsample(dim * 4, dim * 2)
        self.fuse2 = Fuse(dim * 2)
        self.decoder2_base = _build_restormer_stage(dim * 2, dec2_base_blocks, restormer_ffn_expansion)
        self.decoder2_lpeb = LPEB(
            channels=dim * 2,
            window_size=polar_window,
            n_theta=n_theta,
            n_r=n_r,
            polar_proj_dim=polar_proj_dim,
            router_hidden=router_hidden,
            router_topk=router_topk,
            ffn_expansion=restormer_ffn_expansion,
        )

        self.up1 = Upsample(dim * 2, dim)
        self.fuse1 = Fuse(dim)
        self.decoder1 = _build_naf_stage(dim, dec1_base_blocks)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)
        self._last_router_stats: dict[str, torch.Tensor] = {}

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.shape[-2] % 8 != 0 or inp.shape[-1] % 8 != 0:
            raise ValueError("Input height and width must be multiples of 8.")

        x = self.patch_embed(inp)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        enc3 = self.encoder3(self.down2(enc2))

        bottleneck = self.down3(enc3)
        bottleneck = self.bottleneck_base(bottleneck)
        bottleneck = self.bottleneck_lpeb(bottleneck)

        dec3 = self.up3(bottleneck)
        dec3 = self.fuse3(enc3, dec3)
        dec3 = self.decoder3_base(dec3)
        dec3 = self.decoder3_lpeb(dec3)

        dec2 = self.up2(dec3)
        dec2 = self.fuse2(enc2, dec2)
        dec2 = self.decoder2_base(dec2)
        dec2 = self.decoder2_lpeb(dec2)

        dec1 = self.up1(dec2)
        dec1 = self.fuse1(enc1, dec1)
        dec1 = self.decoder1(dec1)

        self._last_router_stats = self._aggregate_router_stats()
        return self.output(dec1) + inp

    def _aggregate_router_stats(self) -> dict[str, torch.Tensor]:
        stats = [
            self.bottleneck_lpeb.get_last_router_stats(),
            self.decoder3_lpeb.get_last_router_stats(),
            self.decoder2_lpeb.get_last_router_stats(),
        ]
        stats = [item for item in stats if item]
        if not stats:
            return {}
        return {
            "mean_confidence": torch.stack([item["mean_confidence"] for item in stats]).mean(),
            "top2_entropy": torch.stack([item["top2_entropy"] for item in stats]).mean(),
            "expert_usage": torch.stack([item["expert_usage"] for item in stats]).mean(dim=0),
        }

    def get_last_router_stats(self) -> dict[str, torch.Tensor]:
        return self._last_router_stats
