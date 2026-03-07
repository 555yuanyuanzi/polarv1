from __future__ import annotations

import torch
import torch.nn as nn

from .common import LayerNorm2d
from .experts import LongAnisoExpert, LongIsoExpert, SharedExpert, ShortAnisoExpert, ShortIsoExpert
from .polar import LocalPolarPrior
from .restormer_lite import GDFN
from .router import TopKRouter


class LPEB(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        n_theta: int = 16,
        n_r: int = 8,
        polar_proj_dim: int = 32,
        router_hidden: int = 32,
        router_topk: int = 2,
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.shared_expert = SharedExpert(channels)
        self.local_polar = LocalPolarPrior(
            channels=channels,
            window_size=window_size,
            n_theta=n_theta,
            n_r=n_r,
            polar_proj_dim=polar_proj_dim,
        )
        self.router = TopKRouter(
            channels=channels,
            window_size=window_size,
            n_theta=n_theta,
            hidden_dim=router_hidden,
            topk=router_topk,
        )
        self.short_iso = ShortIsoExpert(channels)
        self.long_iso = LongIsoExpert(channels)
        self.short_aniso = ShortAnisoExpert(channels, n_theta=n_theta)
        self.long_aniso = LongAnisoExpert(channels, n_theta=n_theta)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GDFN(channels, expansion=ffn_expansion)
        self.last_router_stats: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm1(x)
        f_base = self.shared_expert(xn)
        d_local, confidence = self.local_polar(xn)
        alpha, alpha_up, confidence_up = self.router(d_local, confidence, xn)

        f1 = self.short_iso(xn)
        f2 = self.long_iso(xn)
        f3 = self.short_aniso(xn, d_local)
        f4 = self.long_aniso(xn, d_local)

        f_local = (
            alpha_up[:, 0:1] * f1
            + alpha_up[:, 1:2] * f2
            + alpha_up[:, 2:3] * f3
            + alpha_up[:, 3:4] * f4
        )
        y1 = x + f_base + confidence_up * f_local
        y = y1 + self.ffn(self.norm2(y1))
        self.last_router_stats = self.router.get_last_stats()
        return y

    def get_last_router_stats(self) -> dict[str, torch.Tensor]:
        return self.last_router_stats
