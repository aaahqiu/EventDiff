"""Sample and evaluate an EventDiff checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from event_diff.codec import LatentCodec
from event_diff.data import (
    EventSequenceDataset,
    LaserPowerNormalizer,
    SignedLogNormalizer,
    StridedSubsetSampler,
    discover_files,
    make_loader,
)
from event_diff.diffusion import EDMConfig, sample_edm
from event_diff.model import EventDenoiser
from event_diff.utils import resolve_device, save_json, seed_everything
from event_diff.visualize import make_sequence_comparison, save_comparison_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs_motion_ddp/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_motion_ddp/test"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--known-future-frames", type=int, default=0)
    parser.add_argument("--event-guided-steps", type=int, default=14)
    parser.add_argument("--decode-chunk-size", type=int, default=8)
    parser.add_argument("--active-threshold", type=float, default=1e-3)
    parser.add_argument("--vis-mode", choices=("signed", "grid", "rgb", "gray"), default="signed")
    parser.add_argument("--vis-reduce", choices=("sum", "mean", "maxabs"), default="sum")
    parser.add_argument("--vis-channel", type=int, default=-1)
    parser.add_argument("--vis-theme", choices=("light", "dark"), default="light")
    parser.add_argument("--vis-gamma", type=float, default=0.55)
    parser.add_argument("--vis-clip-mode", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--vis-clip-value", type=float, default=5.0)
    parser.add_argument("--vis-clip-percentile", type=float, default=99.0)
    parser.add_argument("--vis-min-clip", type=float, default=0.05)
    parser.add_argument("--vis-max-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataset(
    args: argparse.Namespace,
    checkpoint: dict,
    normalizer: SignedLogNormalizer,
    laser_normalizer: LaserPowerNormalizer | None,
):
    train_args = checkpoint["args"]
    if args.data_dir is None:
        files = [Path(path) for path in checkpoint["split"]["test"]]
    else:
        files = discover_files(args.data_dir)
    return EventSequenceDataset(
        files,
        normalizer=normalizer,
        cond_frames=int(train_args["cond_frames"]),
        pred_frames=int(train_args["pred_frames"]),
        frame_stride=int(train_args["frame_stride"]),
        window_stride=int(train_args["window_stride"]),
        max_windows_per_file=None,
        laser_normalizer=laser_normalizer,
    )


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    codec = LatentCodec.from_metadata(checkpoint["codec_config"], device=device)
    normalizer = SignedLogNormalizer(float(checkpoint["normalizer"]["clip_value"]))
    laser_config = checkpoint.get("laser_power_normalizer")
    laser_normalizer = (
        None
        if laser_config is None
        else LaserPowerNormalizer(
            min_power=float(laser_config["min_power"]),
            max_power=float(laser_config["max_power"]),
        )
    )
    dataset = build_dataset(args, checkpoint, normalizer, laser_normalizer)
    sampler = StridedSubsetSampler(dataset, args.num_samples)
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        sampler=sampler,
    )
    model = EventDenoiser(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion_config = EDMConfig(**checkpoint["diffusion_config"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = args.output_dir / "npz"
    image_dir = args.output_dir / "visualizations"
    video_dir = args.output_dir / "videos"
    if args.save_npz:
        npz_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "normalized_abs_error": 0.0,
        "normalized_squared_error": 0.0,
        "original_abs_error": 0.0,
        "original_squared_error": 0.0,
        "target_abs_sum": 0.0,
        "prediction_abs_sum": 0.0,
        "active_abs_error": 0.0,
        "active_elements": 0.0,
        "elements": 0.0,
        "samples": 0,
    }
    channel = None if args.vis_channel == -1 else args.vis_channel
    if args.vis_gamma <= 0:
        raise ValueError("--vis-gamma must be positive")
    if args.vis_clip_value <= 0:
        raise ValueError("--vis-clip-value must be positive")
    if not 0 < args.vis_clip_percentile <= 100:
        raise ValueError("--vis-clip-percentile must be in (0, 100]")
    if args.vis_min_clip <= 0:
        raise ValueError("--vis-min-clip must be positive")
    clip_value = args.vis_clip_value if args.vis_clip_mode == "fixed" else None

    for batch in tqdm(loader, desc="Sampling", dynamic_ncols=True):
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        laser_power = batch.get("laser_power")
        if laser_power is not None:
            laser_power = laser_power.to(device, non_blocking=True)
        condition_latents = codec.encode(condition, chunk_size=args.decode_chunk_size)
        target_latents = codec.encode(target, chunk_size=args.decode_chunk_size)
        known_latents = None
        if args.known_future_frames > 0:
            known = min(args.known_future_frames, target_latents.shape[1])
            known_latents = target_latents[:, :known]
        predicted_latents = sample_edm(
            model,
            tuple(target_latents.shape),
            condition_latents,
            diffusion_config,
            laser_power=laser_power,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            known_latents=known_latents,
            event_guided_steps=args.event_guided_steps,
        )
        prediction = codec.decode(predicted_latents, chunk_size=args.decode_chunk_size).clamp(-1.0, 1.0)
        original_target = normalizer.decode(target)
        original_prediction = normalizer.decode(prediction)

        normalized_diff = prediction - target
        original_diff = original_prediction - original_target
        metrics["normalized_abs_error"] += normalized_diff.abs().sum().item()
        metrics["normalized_squared_error"] += normalized_diff.square().sum().item()
        metrics["original_abs_error"] += original_diff.abs().sum().item()
        metrics["original_squared_error"] += original_diff.square().sum().item()
        metrics["target_abs_sum"] += original_target.abs().sum().item()
        metrics["prediction_abs_sum"] += original_prediction.abs().sum().item()
        active_mask = original_target.abs() > args.active_threshold
        metrics["active_elements"] += active_mask.sum().item()
        if active_mask.any().item():
            metrics["active_abs_error"] += original_diff.abs()[active_mask].sum().item()
        metrics["elements"] += target.numel()

        condition_np = tensor_to_numpy(normalizer.decode(condition))
        target_np = tensor_to_numpy(original_target)
        prediction_np = tensor_to_numpy(original_prediction)
        laser_power_raw_np = (
            tensor_to_numpy(batch["laser_power_raw"])
            if "laser_power_raw" in batch
            else None
        )
        for item in range(condition.shape[0]):
            sample_name = f"{batch['name'][item]}_start_{int(batch['start_index'][item]):04d}"
            metrics["samples"] += 1
            if args.save_npz:
                np.savez_compressed(
                    npz_dir / f"{sample_name}.npz",
                    condition=condition_np[item].astype(np.float16),
                    target=target_np[item].astype(np.float16),
                    prediction=prediction_np[item].astype(np.float16),
                    **(
                        {}
                        if laser_power_raw_np is None
                        else {"laser_power": laser_power_raw_np[item].astype(np.float32)}
                    ),
                )
            if args.save_images:
                image = make_sequence_comparison(
                    condition_np[item],
                    target_np[item],
                    prediction_np[item],
                    max_frames=args.vis_max_frames,
                    mode=args.vis_mode,
                    reduce=args.vis_reduce,
                    channel=channel,
                    clip_value=clip_value,
                    clip_percentile=args.vis_clip_percentile,
                    min_clip=args.vis_min_clip,
                    theme=args.vis_theme,
                    gamma=args.vis_gamma,
                )
                image.save(image_dir / f"{sample_name}.png")
            if args.save_videos:
                save_comparison_video(
                    target_np[item],
                    prediction_np[item],
                    video_dir / f"{sample_name}.avi",
                    fps=args.fps,
                    mode=args.vis_mode,
                    reduce=args.vis_reduce,
                    channel=channel,
                    clip_value=clip_value,
                    clip_percentile=args.vis_clip_percentile,
                    min_clip=args.vis_min_clip,
                    theme=args.vis_theme,
                    gamma=args.vis_gamma,
                )

    elements = max(metrics["elements"], 1.0)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "test_windows": len(dataset),
        "evaluated_samples": metrics["samples"],
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "laser_power_conditioned": laser_normalizer is not None,
        "laser_power_normalizer": (
            None
            if laser_normalizer is None
            else {
                "min_power": laser_normalizer.min_power,
                "max_power": laser_normalizer.max_power,
            }
        ),
        "visualization": {
            "theme": args.vis_theme,
            "gamma": args.vis_gamma,
            "clip_mode": args.vis_clip_mode,
            "clip_value": args.vis_clip_value if args.vis_clip_mode == "fixed" else None,
            "clip_percentile": args.vis_clip_percentile,
            "min_clip": args.vis_min_clip,
        },
        "normalized_mae": metrics["normalized_abs_error"] / elements,
        "normalized_mse": metrics["normalized_squared_error"] / elements,
        "original_mae": metrics["original_abs_error"] / elements,
        "original_mse": metrics["original_squared_error"] / elements,
        "target_mean_abs": metrics["target_abs_sum"] / elements,
        "prediction_mean_abs": metrics["prediction_abs_sum"] / elements,
        "amplitude_ratio": metrics["prediction_abs_sum"] / max(metrics["target_abs_sum"], 1e-12),
        "active_fraction": metrics["active_elements"] / elements,
        "active_region_mae": (
            metrics["active_abs_error"] / metrics["active_elements"]
            if metrics["active_elements"] > 0
            else None
        ),
        "npz_dir": str(npz_dir.resolve()) if args.save_npz else None,
        "visualization_dir": str(image_dir.resolve()) if args.save_images else None,
        "video_dir": str(video_dir.resolve()) if args.save_videos else None,
    }
    save_json(summary, args.output_dir / "metrics.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
