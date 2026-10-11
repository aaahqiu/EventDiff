"""Latent codecs for event diffusion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from event_diff.autoencoder import ConvVAE


DEFAULT_VAE_CHECKPOINT = Path(
    os.environ.get("EVENTDIFF_VAE_CHECKPOINT", "checkpoints/eventvae/best.pt")
).expanduser()


class LatentCodec:
    """Encode/decode normalized event frames for diffusion training.

    ``mode='vae'`` uses the frozen EventVAE spatial latent checkpoint. ``mode='raw'``
    keeps normalized voxel values as the diffusion state, optionally downsampled.
    """

    def __init__(
        self,
        mode: str = "vae",
        device: torch.device | str = "cpu",
        vae_checkpoint: str | Path | None = DEFAULT_VAE_CHECKPOINT,
        raw_latent_size: int | None = None,
        image_size: int | None = None,
        in_channels: int = 3,
    ) -> None:
        if mode not in {"vae", "raw"}:
            raise ValueError("mode must be 'vae' or 'raw'")
        self.mode = mode
        self.device = torch.device(device)
        self.vae_checkpoint = Path(vae_checkpoint).expanduser() if vae_checkpoint else None
        self.raw_latent_size = raw_latent_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.model: ConvVAE | None = None
        self.model_config: dict[str, Any] | None = None
        self.clip_value = 100.0

        if self.mode == "vae":
            if self.vae_checkpoint is None:
                raise ValueError("vae_checkpoint is required when mode='vae'")
            if not self.vae_checkpoint.exists():
                raise FileNotFoundError(f"VAE checkpoint not found: {self.vae_checkpoint}")
            checkpoint = torch.load(self.vae_checkpoint, map_location="cpu", weights_only=False)
            self.model_config = dict(checkpoint["model_config"])
            self.model = ConvVAE(**self.model_config)
            self.model.load_state_dict(checkpoint["model"])
            self.model.eval().requires_grad_(False)
            self.model.to(self.device)
            self.clip_value = float(checkpoint.get("clip_value", self.clip_value))
            if self.model.latent_type != "spatial":
                raise ValueError("EventDiff requires a spatial EventVAE checkpoint")
            self.latent_channels = int(self.model.latent_channels)
            self.latent_size = int(self.model.feature_size)
            self.image_size = int(self.model_config["image_size"])
            self.in_channels = int(self.model_config["in_channels"])
        else:
            if self.image_size is None:
                raise ValueError("image_size is required when mode='raw'")
            self.latent_channels = int(in_channels)
            self.latent_size = int(raw_latent_size or image_size)

    def to(self, device: torch.device | str) -> "LatentCodec":
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "vae_checkpoint": str(self.vae_checkpoint.resolve()) if self.vae_checkpoint else None,
            "raw_latent_size": self.raw_latent_size,
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "latent_size": self.latent_size,
            "clip_value": self.clip_value,
            "model_config": self.model_config,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], device: torch.device | str) -> "LatentCodec":
        return cls(
            mode=metadata["mode"],
            device=device,
            vae_checkpoint=metadata.get("vae_checkpoint"),
            raw_latent_size=metadata.get("raw_latent_size"),
            image_size=metadata.get("image_size"),
            in_channels=metadata.get("in_channels", 3),
        )

    @torch.no_grad()
    def encode(self, frames: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """Encode ``(B,T,C,H,W)`` normalized frames to latent sequences."""

        if frames.ndim != 5:
            raise ValueError(f"Expected frames with shape (B,T,C,H,W), got {tuple(frames.shape)}")
        frames = frames.to(self.device, non_blocking=True)
        batch, length, channels, height, width = frames.shape
        flat = frames.reshape(batch * length, channels, height, width)
        if self.mode == "raw":
            if height != self.latent_size or width != self.latent_size:
                flat = F.interpolate(
                    flat,
                    size=(self.latent_size, self.latent_size),
                    mode="bilinear",
                    align_corners=False,
                )
            return flat.reshape(batch, length, channels, self.latent_size, self.latent_size)

        assert self.model is not None
        outputs: list[torch.Tensor] = []
        for start in range(0, len(flat), chunk_size):
            mu, _ = self.model.encode(flat[start : start + chunk_size])
            outputs.append(mu.float())
        latents = torch.cat(outputs, dim=0)
        return latents.reshape(batch, length, *latents.shape[1:])

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
        """Decode latent sequences to normalized ``(B,T,C,H,W)`` frames."""

        if latents.ndim != 5:
            raise ValueError(f"Expected latents with shape (B,T,C,H,W), got {tuple(latents.shape)}")
        latents = latents.to(self.device, non_blocking=True)
        batch, length, channels, height, width = latents.shape
        flat = latents.reshape(batch * length, channels, height, width)
        if self.mode == "raw":
            if self.image_size is not None and (height != self.image_size or width != self.image_size):
                flat = F.interpolate(
                    flat,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            return flat.reshape(batch, length, channels, flat.shape[-2], flat.shape[-1]).clamp(-1.0, 1.0)

        assert self.model is not None
        outputs: list[torch.Tensor] = []
        for start in range(0, len(flat), chunk_size):
            outputs.append(self.model.decode(flat[start : start + chunk_size]).float())
        frames = torch.cat(outputs, dim=0)
        return frames.reshape(batch, length, *frames.shape[1:])
