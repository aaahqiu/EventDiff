"""Train EventDiff on MPH event voxel recordings."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from event_diff.codec import DEFAULT_VAE_CHECKPOINT, LatentCodec
from event_diff.data import (
    DEFAULT_DATA_DIR,
    DistributedEvalSampler,
    EventSequenceDataset,
    LaserPowerNormalizer,
    SignedLogNormalizer,
    StridedSubsetSampler,
    discover_files,
    make_loader,
    split_files,
)
from event_diff.diffusion import EDMConfig, edm_training_loss
from event_diff.model import EventDenoiser
from event_diff.utils import (
    AverageMeter,
    autocast_context,
    cleanup_distributed,
    init_distributed,
    make_grad_scaler,
    save_json,
    seed_everything,
    serializable_args,
)


def parse_channel_mult(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("channel mult must contain at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("channel mult values must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_motion"))
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--latent-mode", choices=("vae", "raw"), default="vae")
    parser.add_argument("--raw-latent-size", type=int, default=128)
    parser.add_argument("--clip-value", type=float, default=100.0)
    parser.add_argument("--use-laser-power", action=argparse.BooleanOptionalAction, default=True,
                        help="Condition the diffusion model on laser_power_per_frame.npy curves")
    parser.add_argument("--laser-power-min", type=float, default=0.0,
                        help="Laser power value mapped to -1")
    parser.add_argument("--laser-power-max", type=float, default=200.0,
                        help="Laser power value mapped to +1")
    parser.add_argument("--cond-frames", type=int, default=3)
    parser.add_argument("--pred-frames", type=int, default=25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--max-windows-per-file", type=int, default=None)
    parser.add_argument("--val-max-samples", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--model-channels", type=int, default=128)
    parser.add_argument("--channel-mult", type=parse_channel_mult, default=(1, 2, 4))
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--laser-power-channels", type=int, default=8,
                        help="Number of per-frame laser conditioning channels")
    parser.add_argument("--laser-power-hidden-channels", type=int, default=64,
                        help="Hidden width of the 1D laser power curve encoder")
    parser.add_argument("--p-mean", type=float, default=0.7)
    parser.add_argument("--p-std", type=float, default=1.6)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--latent-magnitude-weight", type=float, default=0.0)
    parser.add_argument("--cfg-dropout", type=float, default=0.1)
    parser.add_argument("--two-pass-cfg-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_datasets(
    args: argparse.Namespace,
    normalizer: SignedLogNormalizer,
    laser_normalizer: LaserPowerNormalizer | None,
):
    files = discover_files(args.data_dir)
    train_files, val_files, test_files = split_files(files, args.val_ratio, args.test_ratio, args.seed)
    common = dict(
        normalizer=normalizer,
        cond_frames=args.cond_frames,
        pred_frames=args.pred_frames,
        frame_stride=args.frame_stride,
        window_stride=args.window_stride,
        max_windows_per_file=args.max_windows_per_file,
        laser_normalizer=laser_normalizer,
    )
    train_dataset = EventSequenceDataset(train_files, **common)
    val_dataset = EventSequenceDataset(val_files, **common)
    split = {
        "train": [str(path.resolve()) for path in train_files],
        "val": [str(path.resolve()) for path in val_files],
        "test": [str(path.resolve()) for path in test_files],
    }
    return train_dataset, val_dataset, split


def encode_batch(codec: LatentCodec, batch: dict[str, torch.Tensor], device: torch.device):
    condition = batch["condition"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    laser_power = batch.get("laser_power")
    if laser_power is not None:
        laser_power = laser_power.to(device, non_blocking=True)
    with torch.no_grad():
        condition_latents = codec.encode(condition)
        target_latents = codec.encode(target)
    return condition_latents, target_latents, laser_power


def run_epoch(
    model: torch.nn.Module,
    codec: LatentCodec,
    loader,
    optimizer: AdamW | None,
    scaler,
    device: torch.device,
    diffusion_config: EDMConfig,
    args: argparse.Namespace,
    epoch: int,
    phase: str,
    show_progress: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_meter = AverageMeter()
    cond_meter = AverageMeter()
    uncond_meter = AverageMeter()
    iterator = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} {phase}",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )
    for batch in iterator:
        condition_latents, target_latents, laser_power = encode_batch(codec, batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, args.mixed_precision):
            loss_dict = edm_training_loss(
                model,
                target_latents,
                condition_latents,
                diffusion_config,
                laser_power=laser_power,
                cfg_dropout=args.cfg_dropout,
                two_pass_cfg_loss=args.two_pass_cfg_loss,
            )
            loss = loss_dict["loss"]
        if training:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        batch_size = target_latents.shape[0]
        loss_meter.update(loss.detach().float().item(), batch_size)
        cond_meter.update(loss_dict["conditional"].float().item(), batch_size)
        uncond_meter.update(loss_dict["unconditional"].float().item(), batch_size)
        if show_progress:
            iterator.set_postfix(loss=f"{loss_meter.average:.4f}")

    values = torch.tensor(
        [loss_meter.total, cond_meter.total, uncond_meter.total, loss_meter.count],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    count = values[3].item()
    if count == 0:
        raise RuntimeError(f"The {phase} loader produced no samples")
    return {
        "loss": values[0].item() / count,
        "conditional": values[1].item() / count,
        "unconditional": values[2].item() / count,
    }


def unwrap_model(model: torch.nn.Module) -> EventDenoiser:
    return model.module if isinstance(model, DistributedDataParallel) else model


def distributed_barrier(context) -> None:
    if not context.distributed:
        return
    if context.device.type == "cuda":
        dist.barrier(device_ids=[context.local_rank])
    else:
        dist.barrier()


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: EventDenoiser,
    optimizer: AdamW,
    scaler,
    best_val: float,
    codec: LatentCodec,
    laser_normalizer: LaserPowerNormalizer | None,
    diffusion_config: EDMConfig,
    split: dict[str, list[str]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "model_config": model.model_config(),
            "codec_config": codec.metadata(),
            "diffusion_config": diffusion_config.to_dict(),
            "normalizer": {"clip_value": codec.clip_value if codec.mode == "vae" else args.clip_value},
            "laser_power_normalizer": (
                None
                if laser_normalizer is None
                else {
                    "min_power": laser_normalizer.min_power,
                    "max_power": laser_normalizer.max_power,
                }
            ),
            "split": split,
            "args": serializable_args(args),
        },
        path,
    )


def train(args: argparse.Namespace, context) -> None:
    seed_everything(args.seed + context.rank)
    device = context.device
    if context.is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    distributed_barrier(context)

    probe_files = discover_files(args.data_dir)
    laser_normalizer = (
        LaserPowerNormalizer(args.laser_power_min, args.laser_power_max)
        if args.use_laser_power
        else None
    )
    probe_dataset = EventSequenceDataset(
        [probe_files[0]],
        SignedLogNormalizer(args.clip_value),
        cond_frames=args.cond_frames,
        pred_frames=args.pred_frames,
        frame_stride=args.frame_stride,
        max_windows_per_file=1,
        laser_normalizer=laser_normalizer,
    )
    source_channels, source_size, _ = probe_dataset.source_frame_shape
    codec = LatentCodec(
        mode=args.latent_mode,
        device=device,
        vae_checkpoint=args.vae_checkpoint,
        raw_latent_size=args.raw_latent_size,
        image_size=source_size,
        in_channels=source_channels,
    )
    normalizer = SignedLogNormalizer(codec.clip_value if codec.mode == "vae" else args.clip_value)
    train_dataset, val_dataset, split = build_datasets(args, normalizer, laser_normalizer)
    if context.is_main:
        save_json(split, args.output_dir / "split.json")
        save_json(
            serializable_args(args)
            | {
                "codec": codec.metadata(),
                "laser_power_normalizer": (
                    None
                    if laser_normalizer is None
                    else {
                        "min_power": laser_normalizer.min_power,
                        "max_power": laser_normalizer.max_power,
                    }
                ),
                "distributed": context.distributed,
                "world_size": context.world_size,
            },
            args.output_dir / "config.json",
        )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
        )
        if context.distributed
        else None
    )
    val_indices = StridedSubsetSampler(val_dataset, args.val_max_samples).indices
    val_subset = Subset(val_dataset, val_indices)
    val_sampler = (
        DistributedEvalSampler(val_subset, rank=context.rank, world_size=context.world_size)
        if context.distributed
        else None
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        sampler=train_sampler,
    )
    val_loader = make_loader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        sampler=val_sampler,
    )

    model = EventDenoiser(
        latent_channels=codec.latent_channels,
        model_channels=args.model_channels,
        channel_mult=args.channel_mult,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        use_laser_power=args.use_laser_power,
        laser_power_channels=args.laser_power_channels,
        laser_power_hidden_channels=args.laser_power_hidden_channels,
    ).to(device)
    diffusion_config = EDMConfig(
        p_mean=args.p_mean,
        p_std=args.p_std,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        latent_magnitude_weight=args.latent_magnitude_weight,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, enabled=args.mixed_precision == "fp16")
    start_epoch = 1
    best_val = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))
    if context.distributed:
        ddp_options = (
            {"device_ids": [context.local_rank], "output_device": context.local_rank}
            if device.type == "cuda"
            else {}
        )
        model = DistributedDataParallel(model, **ddp_options)

    if context.is_main:
        print(f"Device: {device}; world size: {context.world_size}")
        print(f"Files train/val/test: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
        print(f"Windows train/val: {len(train_dataset)}/{len(val_dataset)}")
        print(
            f"Latent mode: {codec.mode}; latent shape: "
            f"({codec.latent_channels}, {codec.latent_size}, {codec.latent_size})"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        start = time.time()
        show_progress = context.is_main and args.progress
        train_metrics = run_epoch(
            model,
            codec,
            train_loader,
            optimizer,
            scaler,
            device,
            diffusion_config,
            args,
            epoch,
            "train",
            show_progress,
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                unwrap_model(model),
                codec,
                val_loader,
                None,
                scaler,
                device,
                diffusion_config,
                args,
                epoch,
                "val",
                show_progress,
            )
        is_best = val_metrics["loss"] < best_val
        best_val = min(best_val, val_metrics["loss"])
        if context.is_main:
            save_checkpoint(
                args.output_dir / "last.pt",
                epoch=epoch,
                model=unwrap_model(model),
                optimizer=optimizer,
                scaler=scaler,
                best_val=best_val,
                codec=codec,
                laser_normalizer=laser_normalizer,
                diffusion_config=diffusion_config,
                split=split,
                args=args,
            )
            if is_best:
                save_checkpoint(
                    args.output_dir / "best.pt",
                    epoch=epoch,
                    model=unwrap_model(model),
                    optimizer=optimizer,
                    scaler=scaler,
                    best_val=best_val,
                    codec=codec,
                    laser_normalizer=laser_normalizer,
                    diffusion_config=diffusion_config,
                    split=split,
                    args=args,
                )
            print(
                f"Epoch {epoch:03d} | train {train_metrics['loss']:.6f} | "
                f"val {val_metrics['loss']:.6f} | best {best_val:.6f} | "
                f"{time.time() - start:.1f}s"
            )
        distributed_barrier(context)


def main() -> None:
    args = parse_args()
    context = init_distributed(args.device)
    try:
        train(args, context)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
