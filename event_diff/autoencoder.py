"""Frozen EventVAE architecture used as the latent codec."""

from __future__ import annotations

import math

import torch
from torch import nn


def _num_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class UpBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _num_groups(channels)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            ResidualBlock2D(out_channels),
        )


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            ResidualBlock2D(out_channels),
        )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


class ConvVAE(nn.Module):
    """Convolutional EventVAE copied for checkpoint compatibility."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 512,
        latent_dim: int = 256,
        latent_channels: int = 64,
        base_channels: int = 32,
        latent_type: str = "vector",
        downsample_factor: int = 64,
        architecture: str = "legacy",
    ) -> None:
        super().__init__()
        if image_size % downsample_factor != 0:
            raise ValueError("image_size must be divisible by downsample_factor")
        if not _is_power_of_two(downsample_factor):
            raise ValueError("downsample_factor must be a power of two")
        if latent_type not in {"vector", "spatial"}:
            raise ValueError("latent_type must be 'vector' or 'spatial'")
        if architecture not in {"legacy", "residual"}:
            raise ValueError("architecture must be 'legacy' or 'residual'")
        if architecture == "legacy" and downsample_factor != 64:
            raise ValueError("architecture='legacy' is only supported with downsample_factor=64")
        if latent_type == "vector" and (architecture != "legacy" or downsample_factor != 64):
            raise ValueError("latent_type='vector' is only supported by the legacy 64x bottleneck")
        if architecture == "residual" and latent_type != "spatial":
            raise ValueError("architecture='residual' requires latent_type='spatial'")

        self.latent_type = latent_type
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.architecture = architecture
        self.downsample_stages = int(math.log2(downsample_factor))

        if architecture == "legacy":
            channels = [
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 8,
                base_channels * 8,
                base_channels * 8,
            ]
            encoder_layers = []
            previous = in_channels
            for current in channels:
                encoder_layers.append(ConvBlock(previous, current))
                previous = current
        else:
            channels = [base_channels]
            encoder_layers = [
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                ResidualBlock2D(base_channels),
            ]
            previous = base_channels
            for stage in range(self.downsample_stages):
                current = min(base_channels * 2 ** (stage + 1), base_channels * 8)
                encoder_layers.append(DownsampleBlock(previous, current))
                previous = current
                channels.append(current)
            encoder_layers.append(ResidualBlock2D(previous))
        self.encoder = nn.Sequential(*encoder_layers)

        self.feature_size = image_size // downsample_factor
        self.feature_channels = channels[-1]
        flattened = self.feature_channels * self.feature_size**2
        if latent_type == "vector":
            self.to_mu = nn.Linear(flattened, latent_dim)
            self.to_logvar = nn.Linear(flattened, latent_dim)
            self.from_latent = nn.Linear(latent_dim, flattened)
        else:
            self.to_mu = nn.Conv2d(self.feature_channels, latent_channels, kernel_size=1)
            self.to_logvar = nn.Conv2d(self.feature_channels, latent_channels, kernel_size=1)
            self.from_latent = nn.Conv2d(latent_channels, self.feature_channels, kernel_size=1)

        if architecture == "legacy":
            reversed_channels = list(reversed(channels[:-1]))
            decoder_layers = []
            previous = channels[-1]
            for current in reversed_channels:
                decoder_layers.append(UpBlock(previous, current))
                previous = current
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(previous, in_channels, kernel_size=4, stride=2, padding=1),
                    nn.Tanh(),
                ]
            )
        else:
            decoder_layers = [ResidualBlock2D(self.feature_channels)]
            previous = self.feature_channels
            for current in reversed(channels[:-1]):
                decoder_layers.append(UpsampleBlock(previous, current))
                previous = current
            decoder_layers.extend(
                [
                    nn.GroupNorm(_num_groups(previous), previous),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(previous, in_channels, kernel_size=3, padding=1),
                    nn.Tanh(),
                ]
            )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        if self.latent_type == "vector":
            features = features.flatten(start_dim=1)
        return self.to_mu(features), self.to_logvar(features)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(z)
        if self.latent_type == "vector":
            features = features.view(-1, self.feature_channels, self.feature_size, self.feature_size)
        return self.decoder(features)

    def forward(
        self,
        x: torch.Tensor,
        sample_latent: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        if sample_latent is None:
            sample_latent = torch.is_grad_enabled()
        latent = self.reparameterize(mu, logvar) if sample_latent else mu
        reconstruction = self.decode(latent)
        return reconstruction, mu, logvar
