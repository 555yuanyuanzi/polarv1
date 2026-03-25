from __future__ import annotations

import torch
import torch.nn as nn

from .fbeb import FrequencyBandEnhancementBlock
from .importance import RawGuidancePyramid, RestorationImportanceHead
from .local_refine import LocalRefinementBlock
from .naf import NAFBlock
from .restormer_lite import RestormerLiteBlock


_VISUAL_FBEb_KEYS = {
    ("decoder3", "low"),
    ("decoder3", "high"),
    ("decoder2", "high"),
}
_VISUAL_IMPORTANCE_KEYS = {
    ("decoder3", "importance"),
    ("decoder2", "importance"),
}


def _build_naf_stage(
    channels: int,
    num_blocks: int,
    dw_expand: int,
    ffn_expand: int,
) -> nn.Sequential:
    """构建由多个 NAFBlock 组成的 stage。"""

    return nn.Sequential(*[NAFBlock(channels, dw_expand=dw_expand, ffn_expand=ffn_expand) for _ in range(num_blocks)])


def _build_restormer_stage(channels: int, num_blocks: int, ffn_expansion: float) -> nn.Sequential:
    """构建由多个 RestormerLiteBlock 组成的 stage。"""

    num_heads = {96: 2, 192: 4, 384: 8}[channels]
    return nn.Sequential(
        *[RestormerLiteBlock(channels, num_heads=num_heads, ffn_expansion=ffn_expansion) for _ in range(num_blocks)]
    )


class Downsample(nn.Module):
    """用 stride=2 的卷积完成下采样。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Upsample(nn.Module):
    """用 PixelShuffle 完成上采样。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=True),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Fuse(nn.Module):
    """融合 encoder skip 特征和 decoder 特征。"""

    def __init__(self, channels: int, naf_dw_expand: int, naf_ffn_expand: int) -> None:
        super().__init__()
        merged_channels = channels * 2
        self.in_proj = nn.Conv2d(merged_channels, merged_channels, kernel_size=1, bias=True)
        self.block = NAFBlock(merged_channels, dw_expand=naf_dw_expand, ffn_expand=naf_ffn_expand)
        self.out_proj = nn.Conv2d(merged_channels, merged_channels, kernel_size=1, bias=True)

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([enc, dec], dim=1)
        fused = self.out_proj(self.block(self.in_proj(fused)))
        left, right = fused.chunk(2, dim=1)
        return left + right


class PolarFormer(nn.Module):
    """
    当前 V1 的主网络。

    主线结构：
    - Encoder: NAF stages
    - Bottleneck / decoder3 / decoder2: RestormerLite stages
    - decoder3 / decoder2: 可选 FBEB + Importance Head + Local Refinement
    """

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        enc_blocks: tuple[int, int, int] = (3, 4, 6),
        bottleneck_base_blocks: int = 3,
        dec3_base_blocks: int = 3,
        dec2_base_blocks: int = 3,
        dec1_base_blocks: int = 4,
        restormer_ffn_expansion: float = 2.0,
        naf_dw_expand: int = 2,
        naf_ffn_expand: int = 2,
        fbeb_enabled: bool = False,
        fbeb_stages: tuple[str, ...] = (),
        local_refine_enabled: bool = True,
        local_refine_stages: tuple[str, ...] = ("decoder3", "decoder2"),
        fbeb_init_r1: float = 0.22,
        fbeb_init_r2: float = 0.58,
        fbeb_init_tau: float = 0.05,
    ) -> None:
        super().__init__()
        if dim != 48:
            raise ValueError("V1 keeps `dim=48` fixed.")
        valid_fbeb_stages = {"bottleneck", "decoder3", "decoder2"}
        valid_local_refine_stages = {"decoder3", "decoder2"}
        if any(stage not in valid_fbeb_stages for stage in fbeb_stages):
            raise ValueError(f"fbeb_stages must be a subset of {sorted(valid_fbeb_stages)}.")
        if any(stage not in valid_local_refine_stages for stage in local_refine_stages):
            raise ValueError(f"local_refine_stages must be a subset of {sorted(valid_local_refine_stages)}.")
        # FBEB 的启用位置和 local refinement 的启用位置分别独立控制。
        fbeb_stage_set = set(fbeb_stages) if fbeb_enabled else set()
        local_refine_stage_set = set(local_refine_stages) if local_refine_enabled else set()
        guide_channels = max(dim // 3, 16)
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=True)
        # 从原始模糊图中提取给 decoder2 / decoder3 使用的 raw guidance。
        self.raw_guidance = RawGuidancePyramid(inp_channels=inp_channels, guide_channels=guide_channels)

        self.encoder1 = _build_naf_stage(dim, enc_blocks[0], naf_dw_expand, naf_ffn_expand)
        self.down1 = Downsample(dim, dim * 2)
        self.encoder2 = _build_naf_stage(dim * 2, enc_blocks[1], naf_dw_expand, naf_ffn_expand)
        self.down2 = Downsample(dim * 2, dim * 4)
        self.encoder3 = _build_naf_stage(dim * 4, enc_blocks[2], naf_dw_expand, naf_ffn_expand)
        self.down3 = Downsample(dim * 4, dim * 8)

        self.bottleneck_base = _build_restormer_stage(dim * 8, bottleneck_base_blocks, restormer_ffn_expansion)
        self.bottleneck_fbeb = (
            FrequencyBandEnhancementBlock(
                channels=dim * 8,
                init_r1=fbeb_init_r1,
                init_r2=fbeb_init_r2,
                init_tau=fbeb_init_tau,
            )
            if "bottleneck" in fbeb_stage_set
            else nn.Identity()
        )

        self.up3 = Upsample(dim * 8, dim * 4)
        self.fuse3 = Fuse(dim * 4, naf_dw_expand, naf_ffn_expand)
        self.decoder3_base = _build_restormer_stage(dim * 4, dec3_base_blocks, restormer_ffn_expansion)
        self.decoder3_fbeb = (
            FrequencyBandEnhancementBlock(
                channels=dim * 4,
                init_r1=fbeb_init_r1,
                init_r2=fbeb_init_r2,
                init_tau=fbeb_init_tau,
            )
            if "decoder3" in fbeb_stage_set
            else nn.Identity()
        )
        self.decoder3_importance = (
            RestorationImportanceHead(dim * 4, guidance_channels=guide_channels)
            if "decoder3" in local_refine_stage_set
            else None
        )
        self.decoder3_local_refine = (
            LocalRefinementBlock(dim * 4) if "decoder3" in local_refine_stage_set else nn.Identity()
        )

        self.up2 = Upsample(dim * 4, dim * 2)
        self.fuse2 = Fuse(dim * 2, naf_dw_expand, naf_ffn_expand)
        self.decoder2_base = _build_restormer_stage(dim * 2, dec2_base_blocks, restormer_ffn_expansion)
        self.decoder2_fbeb = (
            FrequencyBandEnhancementBlock(
                channels=dim * 2,
                init_r1=fbeb_init_r1,
                init_r2=fbeb_init_r2,
                init_tau=fbeb_init_tau,
            )
            if "decoder2" in fbeb_stage_set
            else nn.Identity()
        )
        self.decoder2_importance = (
            RestorationImportanceHead(dim * 2, guidance_channels=guide_channels)
            if "decoder2" in local_refine_stage_set
            else None
        )
        self.decoder2_local_refine = (
            LocalRefinementBlock(dim * 2) if "decoder2" in local_refine_stage_set else nn.Identity()
        )

        self.up1 = Upsample(dim * 2, dim)
        self.fuse1 = Fuse(dim, naf_dw_expand, naf_ffn_expand)
        self.decoder1 = _build_naf_stage(dim, dec1_base_blocks, naf_dw_expand, naf_ffn_expand)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)
        self._last_fbeb_stats: dict[str, torch.Tensor] = {}
        self._last_importance_stats: dict[str, torch.Tensor] = {}
        self._last_visuals: dict[str, torch.Tensor] = {}

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.shape[-2] % 8 != 0 or inp.shape[-1] % 8 != 0:
            raise ValueError("Input height and width must be multiples of 8.")

        # raw guidance 从原始模糊图直接提取，后续用于 Importance Head。
        raw_guidance = self.raw_guidance(inp)
        x = self.patch_embed(inp)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        enc3 = self.encoder3(self.down2(enc2))

        bottleneck = self.down3(enc3)
        bottleneck = self.bottleneck_base(bottleneck)
        bottleneck = self.bottleneck_fbeb(bottleneck)

        # decoder3: base -> FBEB -> Importance -> LocalRefine
        dec3 = self.up3(bottleneck)
        dec3 = self.fuse3(enc3, dec3)
        dec3_stage = self.decoder3_base(dec3)
        dec3_fbeb = self.decoder3_fbeb(dec3_stage)
        if self.decoder3_importance is not None:
            dec3_importance = self.decoder3_importance(dec3_stage, dec3_fbeb, raw_guidance["decoder3"])
            dec3 = self.decoder3_local_refine(dec3_fbeb, importance_map=dec3_importance)
        else:
            dec3 = self.decoder3_local_refine(dec3_fbeb)

        # decoder2: base -> FBEB -> Importance -> LocalRefine
        dec2 = self.up2(dec3)
        dec2 = self.fuse2(enc2, dec2)
        dec2_stage = self.decoder2_base(dec2)
        dec2_fbeb = self.decoder2_fbeb(dec2_stage)
        if self.decoder2_importance is not None:
            dec2_importance = self.decoder2_importance(dec2_stage, dec2_fbeb, raw_guidance["decoder2"])
            dec2 = self.decoder2_local_refine(dec2_fbeb, importance_map=dec2_importance)
        else:
            dec2 = self.decoder2_local_refine(dec2_fbeb)

        dec1 = self.up1(dec2)
        dec1 = self.fuse1(enc1, dec1)
        dec1 = self.decoder1(dec1)

        self._last_fbeb_stats = self._aggregate_fbeb_stats()
        self._last_importance_stats = self._aggregate_importance_stats()
        self._last_visuals = self._aggregate_visuals()
        return self.output(dec1) + inp

    def _aggregate_fbeb_stats(self) -> dict[str, torch.Tensor]:
        """聚合所有启用的 FBEB 模块统计量，便于日志记录。"""

        stats = []
        for module in (self.bottleneck_fbeb, self.decoder3_fbeb, self.decoder2_fbeb):
            getter = getattr(module, "get_last_band_stats", None)
            if getter is None:
                continue
            stage_stats = getter()
            if stage_stats:
                stats.append(stage_stats)
        stats = [item for item in stats if item]
        if not stats:
            return {}
        return {
            "fbeb/r1": torch.stack([item["r1"] for item in stats]).mean(),
            "fbeb/r2": torch.stack([item["r2"] for item in stats]).mean(),
            "fbeb/tau": torch.stack([item["tau"] for item in stats]).mean(),
            "fbeb/low_energy": torch.stack([item["low_energy"] for item in stats]).mean(),
            "fbeb/mid_energy": torch.stack([item["mid_energy"] for item in stats]).mean(),
            "fbeb/high_energy": torch.stack([item["high_energy"] for item in stats]).mean(),
        }

    def _aggregate_importance_stats(self) -> dict[str, torch.Tensor]:
        """聚合 decoder3 / decoder2 上的 importance 统计量。"""

        stats = []
        for module in (self.decoder3_importance, self.decoder2_importance):
            if module is None:
                continue
            stage_stats = module.get_last_importance_stats()
            if stage_stats:
                stats.append(stage_stats)
        if not stats:
            return {}
        keys = stats[0].keys()
        return {key: torch.stack([item[key] for item in stats]).mean() for key in keys}

    def _aggregate_visuals(self) -> dict[str, torch.Tensor]:
        """鑱氬悎 FBEB 鍜?importance head 鐨勫彲瑙嗗寲鍥俱€?"""

        visuals: dict[str, torch.Tensor] = {}
        for stage_name, module in (
            ("bottleneck", self.bottleneck_fbeb),
            ("decoder3", self.decoder3_fbeb),
            ("decoder2", self.decoder2_fbeb),
        ):
            getter = getattr(module, "get_last_band_visuals", None)
            if getter is None:
                continue
            stage_visuals = getter()
            if not stage_visuals:
                continue
            for key, value in stage_visuals.items():
                if (stage_name, key) in _VISUAL_FBEb_KEYS:
                    visuals[f"fbeb/{stage_name}_{key}"] = value

        for stage_name, module in (
            ("decoder3", self.decoder3_importance),
            ("decoder2", self.decoder2_importance),
        ):
            if module is None:
                continue
            stage_visuals = module.get_last_visuals()
            if not stage_visuals:
                continue
            for key, value in stage_visuals.items():
                if (stage_name, key) in _VISUAL_IMPORTANCE_KEYS:
                    visuals[f"importance/{stage_name}_{key}"] = value

        return visuals

    def get_last_fbeb_stats(self) -> dict[str, torch.Tensor]:
        return self._last_fbeb_stats

    def get_last_importance_stats(self) -> dict[str, torch.Tensor]:
        return self._last_importance_stats

    def get_last_visuals(self) -> dict[str, torch.Tensor]:
        return self._last_visuals
