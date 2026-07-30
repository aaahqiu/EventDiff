"""Spatio-temporal denoising network for event latent sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(float(max_period), device=values.device))
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = values.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(time_embedding))[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)


class LaserPowerEncoder(nn.Module):
    """Encode a normalized laser-power curve for global and per-frame conditioning."""

    def __init__(
        self,
        model_channels: int,
        time_dim: int,
        power_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        if power_channels <= 0:
            raise ValueError("power_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.power_channels = power_channels
        self.backbone = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, time_dim),
        )
        self.frame_out = nn.Conv1d(hidden_channels, power_channels, kernel_size=3, padding=1)

    def forward(
        self,
        laser_power: torch.Tensor,
        target_frames: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if laser_power.ndim != 2:
            raise ValueError("laser_power must have shape (B,L)")
        if laser_power.shape[1] < target_frames:
            raise ValueError(
                f"laser_power length {laser_power.shape[1]} is shorter than target_frames {target_frames}"
            )
        curve = torch.nan_to_num(laser_power.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        delta = torch.zeros_like(curve)
        delta[:, 1:] = curve[:, 1:] - curve[:, :-1]
        features = self.backbone(torch.stack([curve, delta], dim=1))
        global_embedding = self.global_mlp(self.global_pool(features).flatten(1))
        frame_features = self.frame_out(features)[:, :, -target_frames:]
        frame_map = frame_features[:, :, :, None, None].expand(-1, -1, -1, height, width)
        return global_embedding, frame_map


@dataclass(frozen=True)
class EventDenoiserConfig:
    latent_channels: int
    model_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    dropout: float = 0.0
    use_laser_power: bool = False
    laser_power_channels: int = 8
    laser_power_hidden_channels: int = 64


class EventDenoiser(nn.Module):
    """3D UNet denoiser with E-Motion-style condition latent concatenation.

    Inputs and outputs use ``(B,T,C,H,W)``. Internally the condition frames are
    averaged in latent space and repeated for every future frame, matching
    E-Motion's repeated image-latent conditioning while staying event-native.
    """

    def __init__(
        self,
        latent_channels: int,
        model_channels: int = 128,
        channel_mult: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        use_laser_power: bool = False,
        laser_power_channels: int = 8,
        laser_power_hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if model_channels <= 0:
            raise ValueError("model_channels must be positive")
        if num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive")
        if not channel_mult:
            raise ValueError("channel_mult must not be empty")
        self.config = EventDenoiserConfig(
            latent_channels=latent_channels,
            model_channels=model_channels,
            channel_mult=tuple(int(value) for value in channel_mult),
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            use_laser_power=use_laser_power,
            laser_power_channels=laser_power_channels,
            laser_power_hidden_channels=laser_power_hidden_channels,
        )
        self.latent_channels = latent_channels
        self.use_laser_power = use_laser_power
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.condition_pool = nn.Sequential(
            nn.Conv3d(latent_channels, model_channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.condition_mlp = nn.Linear(model_channels, time_dim, bias=False)
        self.laser_power_encoder = (
            LaserPowerEncoder(
                model_channels=model_channels,
                time_dim=time_dim,
                power_channels=laser_power_channels,
                hidden_channels=laser_power_hidden_channels,
            )
            if use_laser_power
            else None
        )

        input_channels = latent_channels * 2 + (laser_power_channels if use_laser_power else 0)
        level_channels = [model_channels * int(mult) for mult in channel_mult]
        self.input = nn.Conv3d(input_channels, level_channels[0], kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        skip_channels: list[int] = [level_channels[0]]
        current = level_channels[0]
        for level, channels in enumerate(level_channels):
            for _ in range(num_res_blocks):
                self.down.append(ResBlock3D(current, channels, time_dim, dropout))
                current = channels
                skip_channels.append(current)
            if level != len(level_channels) - 1:
                self.down.append(Downsample3D(current))
                skip_channels.append(current)

        self.middle = nn.ModuleList(
            [
                ResBlock3D(current, current, time_dim, dropout),
                ResBlock3D(current, current, time_dim, dropout),
            ]
        )

        self.up = nn.ModuleList()
        for level, channels in reversed(list(enumerate(level_channels))):
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock3D(current + skip_channels.pop(), channels, time_dim, dropout))
                current = channels
            if level != 0:
                self.up.append(Upsample3D(current))
        if skip_channels:
            raise RuntimeError("Internal skip-channel bookkeeping mismatch")

        self.output = nn.Sequential(
            nn.GroupNorm(_num_groups(current), current),
            nn.SiLU(),
            nn.Conv3d(current, latent_channels, kernel_size=3, padding=1),
        )

    def model_config(self) -> dict[str, object]:
        return {
            "latent_channels": self.config.latent_channels,
            "model_channels": self.config.model_channels,
            "channel_mult": list(self.config.channel_mult),
            "num_res_blocks": self.config.num_res_blocks,
            "dropout": self.config.dropout,
            "use_laser_power": self.config.use_laser_power,
            "laser_power_channels": self.config.laser_power_channels,
            "laser_power_hidden_channels": self.config.laser_power_hidden_channels,
        }

    @staticmethod
    def repeat_condition(condition_latents: torch.Tensor, target_frames: int) -> torch.Tensor:
        if condition_latents.ndim != 5:
            raise ValueError("condition_latents must have shape (B,T,C,H,W)")
        condition = condition_latents.mean(dim=1, keepdim=True)
        return condition.repeat(1, target_frames, 1, 1, 1)

    def _time_embedding(
        self,
        c_noise: torch.Tensor,
        condition_latents: torch.Tensor,
        laser_power: torch.Tensor | None,
        target_frames: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embedding = sinusoidal_embedding(c_noise, self.config.model_channels)
        embedding = self.time_mlp(embedding)
        condition = condition_latents.permute(0, 2, 1, 3, 4)
        pooled = self.condition_pool(condition).flatten(1)
        embedding = embedding + self.condition_mlp(pooled)
        laser_map = None
        if self.laser_power_encoder is not None:
            if laser_power is None:
                raise ValueError("laser_power conditioning is enabled but no laser_power tensor was provided")
            laser_embedding, laser_map = self.laser_power_encoder(
                laser_power,
                target_frames=target_frames,
                height=height,
                width=width,
            )
            embedding = embedding + laser_embedding.to(dtype=embedding.dtype)
            laser_map = laser_map.to(dtype=condition_latents.dtype)
        return embedding, laser_map

    def forward(
        self,
        noisy_latents: torch.Tensor,
        c_noise: torch.Tensor,
        condition_latents: torch.Tensor,
        laser_power: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_latents.ndim != 5:
            raise ValueError("noisy_latents must have shape (B,T,C,H,W)")
        if condition_latents.ndim != 5:
            raise ValueError("condition_latents must have shape (B,T,C,H,W)")
        if noisy_latents.shape[0] != condition_latents.shape[0]:
            raise ValueError("Batch sizes of noisy_latents and condition_latents differ")
        if noisy_latents.shape[2] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {noisy_latents.shape[2]}"
            )
        if c_noise.ndim == 0:
            c_noise = c_noise.repeat(noisy_latents.shape[0])

        repeated_condition = self.repeat_condition(condition_latents, noisy_latents.shape[1])
        pieces = [noisy_latents, repeated_condition]
        target_frames, height, width = noisy_latents.shape[1], noisy_latents.shape[-2], noisy_latents.shape[-1]
        time_embedding, laser_map = self._time_embedding(
            c_noise,
            condition_latents,
            laser_power,
            target_frames=target_frames,
            height=height,
            width=width,
        )
        if laser_map is not None:
            pieces.append(laser_map.permute(0, 2, 1, 3, 4))
        x = torch.cat(pieces, dim=2)
        x = x.permute(0, 2, 1, 3, 4)

        h = self.input(x)
        skips = [h]
        for module in self.down:
            if isinstance(module, ResBlock3D):
                h = module(h, time_embedding)
            else:
                h = module(h)
            skips.append(h)

        for module in self.middle:
            h = module(h, time_embedding)

        for module in self.up:
            if isinstance(module, ResBlock3D):
                skip = skips.pop()
                if h.shape[-3:] != skip.shape[-3:]:
                    h = F.interpolate(h, size=skip.shape[-3:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = module(h, time_embedding)
            else:
                h = module(h)

        output = self.output(h)
        return output.permute(0, 2, 1, 3, 4)
