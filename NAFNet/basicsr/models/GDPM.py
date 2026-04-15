from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalDirectionalPriorModulation(nn.Module):
    """
    GDPM: Global Directional Prior Modulation

    A lightweight global directional prior module for motion deblurring.

    Pipeline:
        input blur image
        -> downsample
        -> FFT + fftshift
        -> log magnitude spectrum
        -> polar directional pooling on mid-high frequencies
        -> direction prior vector p in R^K
        -> MLP -> gamma, beta
        -> FiLM modulation on feature map

    Args:
        feat_channels: feature channels to modulate
        in_channels: input image channels
        prior_size: low-resolution size for FFT prior extraction
        num_dirs: number of direction bins, default 4
        r_min_ratio: minimum normalized radius for mid-high frequency pooling
        dir_sigma: softness of directional masks; if None, auto-set
        mlp_hidden: hidden dim of the small MLP
        use_grayscale: convert input image to grayscale before FFT
    """

    def __init__(
        self,
        feat_channels: int,
        in_channels: int = 3,
        prior_size: int = 64,
        num_dirs: int = 4,
        r_min_ratio: float = 0.15,
        dir_sigma: float | None = None,
        mlp_hidden: int = 32,
        use_grayscale: bool = True,
    ) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.in_channels = in_channels
        self.prior_size = prior_size
        self.num_dirs = num_dirs
        self.r_min_ratio = r_min_ratio
        self.use_grayscale = use_grayscale

        if dir_sigma is None:
            dir_sigma = math.pi / max(2 * num_dirs, 4)
        self.dir_sigma = dir_sigma

        # Map directional prior vector p in R^K to FiLM params [gamma, beta]
        self.mlp = nn.Sequential(
            nn.Linear(num_dirs, mlp_hidden, bias=True),
            nn.GELU(),
            nn.Linear(mlp_hidden, feat_channels * 2, bias=True),
        )

        # Learnable residual scale for stable FiLM injection
        self.scale = nn.Parameter(torch.zeros(1))

        # Optional cache for geometry tensors
        self._cache_key: tuple[int, str, int, float, float] | None = None
        self._cached_radius: torch.Tensor | None = None
        self._cached_dir_masks: torch.Tensor | None = None

    def _build_radius_map(self, size: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        radius = radius / radius.max().clamp_min(1e-6)
        return radius.view(1, 1, size, size)

    def _build_angle_map(self, size: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        theta = torch.atan2(yy, xx)              # [-pi, pi]
        theta = torch.remainder(theta, math.pi) # orientation in [0, pi)
        return theta.view(1, 1, size, size)

    def _build_direction_masks(self, angle_map: torch.Tensor) -> torch.Tensor:
        """
        Soft directional masks with K orientation centers.
        Output: [1, K, H, W]
        """
        centers = torch.linspace(
            0.0,
            math.pi * (self.num_dirs - 1) / self.num_dirs,
            steps=self.num_dirs,
            device=angle_map.device,
            dtype=angle_map.dtype,
        ).view(1, self.num_dirs, 1, 1)

        diff = torch.abs(angle_map - centers)
        diff = torch.minimum(diff, math.pi - diff)
        masks = torch.exp(-(diff.square()) / (2.0 * (self.dir_sigma ** 2)))
        masks = masks / masks.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return masks

    def _get_cached_geometry(self, size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (size, str(device), self.num_dirs, float(self.r_min_ratio), float(self.dir_sigma))
        if self._cache_key != key:
            with torch.no_grad():
                radius = self._build_radius_map(size, device)
                angle_map = self._build_angle_map(size, device)
                dir_masks = self._build_direction_masks(angle_map)
            self._cache_key = key
            self._cached_radius = radius
            self._cached_dir_masks = dir_masks

        assert self._cached_radius is not None
        assert self._cached_dir_masks is not None
        return self._cached_radius, self._cached_dir_masks

    def clear_cache(self) -> None:
        self._cache_key = None
        self._cached_radius = None
        self._cached_dir_masks = None

    def _extract_direction_prior(self, x_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract global direction prior vector p in R^K from the input blur image.

        Args:
            x_img: [B, C, H, W]

        Returns:
            p: [B, K]
            log_mag: [B, 1, S, S]
        """
        if self.use_grayscale and x_img.shape[1] > 1:
            x_gray = x_img.mean(dim=1, keepdim=True)
        else:
            x_gray = x_img[:, :1]

        x_small = F.interpolate(
            x_gray,
            size=(self.prior_size, self.prior_size),
            mode="bilinear",
            align_corners=False,
        )

        # zero-mean before FFT
        x_small = x_small - x_small.mean(dim=(-2, -1), keepdim=True)

        fft = torch.fft.fft2(x_small.float(), dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        log_mag = torch.log1p(torch.abs(fft))  # [B,1,S,S]

        radius, dir_masks = self._get_cached_geometry(self.prior_size, x_img.device)

        # only pool from mid-high frequencies
        freq_mask = (radius >= self.r_min_ratio).to(dtype=log_mag.dtype)  # [1,1,S,S]

        # directional pooling
        p_list = []
        for i in range(self.num_dirs):
            dmask = dir_masks[:, i : i + 1] * freq_mask  # [1,1,S,S]
            denom = dmask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            val = (log_mag * dmask).sum(dim=(-2, -1), keepdim=True) / denom
            p_list.append(val.squeeze(-1).squeeze(-1).squeeze(-1))  # [B]

        p = torch.stack(p_list, dim=1)  # [B, K]

        # remove common energy bias, keep directional contrast
        p = p - p.mean(dim=1, keepdim=True)

        return p, log_mag

    def forward(
        self,
        x_img: torch.Tensor,
        feat: torch.Tensor,
        return_prior: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x_img: input blur image, [B, Cin, H, W]
            feat: feature to modulate, [B, C, Hf, Wf]

        Returns:
            modulated feature, or (feature, aux_stats)
        """
        assert feat.shape[1] == self.feat_channels, (
            f"Expected feat channels {self.feat_channels}, got {feat.shape[1]}"
        )

        p, log_mag = self._extract_direction_prior(x_img)  # [B,K]

        film = self.mlp(p)  # [B, 2C]
        gamma, beta = torch.chunk(film, 2, dim=1)

        gamma = gamma.view(feat.shape[0], self.feat_channels, 1, 1)
        beta = beta.view(feat.shape[0], self.feat_channels, 1, 1)

        out = feat * (1.0 + self.scale * gamma) + self.scale * beta

        if return_prior:
            aux = {
                "direction_prior": p.detach(),                     # [B, K]
                "gamma_mean": gamma.mean().detach(),
                "beta_mean": beta.mean().detach(),
                "scale": self.scale.detach(),
                "log_mag": log_mag.detach(),                       # [B,1,S,S]
            }
            return out, aux

        return out