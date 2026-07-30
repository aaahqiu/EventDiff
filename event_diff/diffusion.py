"""E-Motion-style EDM diffusion objective and sampler."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class EDMConfig:
    """Noise and sampling settings.

    ``p_mean``/``p_std`` and the preconditioning equations mirror the
    E-Motion training loop.
    """

    p_mean: float = 0.7
    p_std: float = 1.6
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    latent_magnitude_weight: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    dims = target_dims - x.ndim
    if dims < 0:
        raise ValueError(f"input has {x.ndim} dims but target has {target_dims}")
    return x[(...,) + (None,) * dims]


def edm_coefficients(sigma: torch.Tensor, target_dims: int) -> tuple[torch.Tensor, ...]:
    sigma = append_dims(sigma, target_dims)
    c_skip = 1 / (sigma.square() + 1)
    c_out = -sigma / (sigma.square() + 1).sqrt()
    c_in = 1 / (sigma.square() + 1).sqrt()
    c_noise = sigma.flatten().log() / 4
    loss_weight = (sigma.square() + 1) / sigma.square().clamp_min(1e-12)
    return c_skip, c_out, c_in, c_noise, loss_weight


def sample_training_sigmas(
    batch_size: int,
    config: EDMConfig,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    rnd = torch.randn((batch_size,), device=device, generator=generator)
    sigma = (rnd * config.p_std + config.p_mean).exp()
    return sigma.clamp(config.sigma_min, config.sigma_max)


def denoise(
    model,
    noisy_latents: torch.Tensor,
    sigma: torch.Tensor,
    condition_latents: torch.Tensor,
    laser_power: torch.Tensor | None = None,
) -> torch.Tensor:
    c_skip, c_out, c_in, c_noise, _ = edm_coefficients(sigma, noisy_latents.ndim)
    model_pred = model(
        c_in * noisy_latents,
        c_noise.to(noisy_latents.device),
        condition_latents,
        laser_power=laser_power,
    )
    return c_out * model_pred + c_skip * noisy_latents


def edm_training_loss(
    model,
    clean_latents: torch.Tensor,
    condition_latents: torch.Tensor,
    config: EDMConfig,
    *,
    laser_power: torch.Tensor | None = None,
    cfg_dropout: float = 0.1,
    two_pass_cfg_loss: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if not 0 <= cfg_dropout < 1:
        raise ValueError("cfg_dropout must be in [0, 1)")
    batch_size = clean_latents.shape[0]
    sigma = sample_training_sigmas(batch_size, config, clean_latents.device, generator)
    noise = torch.randn(clean_latents.shape, device=clean_latents.device, dtype=clean_latents.dtype)
    c_skip, c_out, c_in, c_noise, loss_weight = edm_coefficients(sigma, clean_latents.ndim)
    noisy_latents = clean_latents + noise * append_dims(sigma, clean_latents.ndim)

    def one_pass(cond: torch.Tensor, power: torch.Tensor | None) -> torch.Tensor:
        pred = model(
            c_in * noisy_latents,
            c_noise.to(clean_latents.device),
            cond,
            laser_power=power,
        )
        pred_x0 = c_out * pred + c_skip * noisy_latents
        per_element = (pred_x0 - clean_latents).square() * loss_weight
        if config.latent_magnitude_weight > 0:
            per_element = per_element * (1 + config.latent_magnitude_weight * clean_latents.abs())
        return per_element.mean()

    if two_pass_cfg_loss:
        zero_condition = torch.zeros_like(condition_latents)
        zero_power = torch.zeros_like(laser_power) if laser_power is not None else None
        conditional = one_pass(condition_latents, laser_power)
        unconditional = one_pass(zero_condition, zero_power)
        loss = conditional + unconditional
        return {"loss": loss, "conditional": conditional.detach(), "unconditional": unconditional.detach()}

    if cfg_dropout > 0:
        keep = torch.rand(
            (batch_size, 1, 1, 1, 1),
            device=condition_latents.device,
            dtype=condition_latents.dtype,
            generator=generator,
        ) >= cfg_dropout
        condition_latents = condition_latents * keep
        if laser_power is not None:
            laser_power = laser_power * keep.flatten(start_dim=1)
    loss = one_pass(condition_latents, laser_power)
    return {"loss": loss, "conditional": loss.detach(), "unconditional": torch.zeros_like(loss).detach()}


def karras_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros(1)])


@torch.inference_mode()
def sample_edm(
    model,
    shape: tuple[int, int, int, int, int],
    condition_latents: torch.Tensor,
    config: EDMConfig,
    *,
    laser_power: torch.Tensor | None = None,
    num_steps: int = 25,
    guidance_scale: float = 1.0,
    generator: torch.Generator | None = None,
    known_latents: torch.Tensor | None = None,
    event_guided_steps: int = 0,
) -> torch.Tensor:
    """Generate target latents with optional classifier-free guidance."""

    device = condition_latents.device
    dtype = condition_latents.dtype
    latents = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    sigmas = karras_sigmas(num_steps, config.sigma_min, config.sigma_max, config.rho, device)
    latents = latents * sigmas[0]

    known_noise = None
    if known_latents is not None:
        if known_latents.ndim != 5:
            raise ValueError("known_latents must have shape (B,T,C,H,W)")
        if known_latents.shape[0] != shape[0] or known_latents.shape[2:] != shape[2:]:
            raise ValueError("known_latents shape is incompatible with sample shape")
        known_noise = torch.randn(
            known_latents.shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    zero_condition = torch.zeros_like(condition_latents)
    zero_power = torch.zeros_like(laser_power) if laser_power is not None else None
    for step_index in range(num_steps):
        sigma = sigmas[step_index].repeat(shape[0])
        next_sigma = sigmas[step_index + 1]
        model_input = latents
        if known_latents is not None and known_noise is not None and 0 < step_index < event_guided_steps:
            prefix = known_latents + known_noise * sigmas[step_index]
            model_input = torch.cat([prefix, latents[:, known_latents.shape[1] :]], dim=1)

        denoised = denoise(model, model_input, sigma, condition_latents, laser_power=laser_power)
        if guidance_scale != 1.0:
            denoised_uncond = denoise(
                model,
                model_input,
                sigma,
                zero_condition,
                laser_power=zero_power,
            )
            denoised = denoised_uncond + guidance_scale * (denoised - denoised_uncond)
        derivative = (latents - denoised) / append_dims(sigma, latents.ndim).clamp_min(1e-12)
        latents = latents + derivative * (next_sigma - sigmas[step_index])
    return latents
