from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        n_theta: int = 16,
        hidden_dim: int = 32,
        topk: int = 2,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.topk = topk
        self.context_proj = nn.Conv2d(channels, n_theta, kernel_size=1, bias=True)
        self.in_proj = nn.Conv2d(n_theta + 1 + n_theta, hidden_dim, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
            bias=True,
        )
        self.out_proj = nn.Conv2d(hidden_dim, 4, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.last_stats: dict[str, torch.Tensor] = {}

    def forward(
        self,
        d_local: torch.Tensor,
        confidence: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = F.avg_pool2d(x, kernel_size=self.window_size, stride=self.window_size)
        context = self.context_proj(context)
        router_input = torch.cat([d_local, confidence, context], dim=1)

        logits = self.out_proj(self.dwconv(self.act(self.in_proj(router_input))))
        top_values, top_indices = torch.topk(logits, k=self.topk, dim=1)
        masked_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        masked_logits.scatter_(1, top_indices, top_values)
        alpha = F.softmax(masked_logits, dim=1)

        alpha_up = F.interpolate(alpha, size=x.shape[-2:], mode="nearest")
        confidence_up = F.interpolate(confidence, size=x.shape[-2:], mode="nearest")

        self.last_stats = {
            "mean_confidence": confidence.mean().detach(),
            "top2_entropy": (-(alpha * alpha.clamp_min(1e-8).log()).sum(dim=1).mean()).detach(),
            "expert_usage": alpha.mean(dim=(0, 2, 3)).detach(),
        }
        return alpha, alpha_up, confidence_up

    def get_last_stats(self) -> dict[str, torch.Tensor]:
        return self.last_stats
