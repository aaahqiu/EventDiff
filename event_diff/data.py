"""Datasets for MPH event voxel ``.npy`` recordings.

The MPH voxel files are large ``float16`` arrays with shape ``(N, C, H, W)``.
This module keeps them memory-mapped and constructs condition/future windows
without loading whole recordings into RAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


DEFAULT_DATA_DIR = Path("/data4/tkp/MPH-Data/voxel_output")


class SignedLogNormalizer:
    """Map heavy-tailed signed event voxel values to ``[-1, 1]``."""

    def __init__(self, clip_value: float = 100.0) -> None:
        if clip_value <= 0:
            raise ValueError("clip_value must be positive")
        self.clip_value = float(clip_value)
        self._scale = math.log1p(self.clip_value)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        x = x.clamp(-self.clip_value, self.clip_value)
        return x.sign() * torch.log1p(x.abs()) / self._scale

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        return x.sign() * torch.expm1(x.abs() * self._scale)


class LaserPowerNormalizer:
    """Map laser power values to ``[-1, 1]`` for conditioning."""

    def __init__(self, min_power: float = 0.0, max_power: float = 200.0) -> None:
        if max_power <= min_power:
            raise ValueError("max_power must be greater than min_power")
        self.min_power = float(min_power)
        self.max_power = float(max_power)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=self.min_power, posinf=self.max_power, neginf=self.min_power)
        x = x.clamp(self.min_power, self.max_power)
        normalized = (x - self.min_power) / (self.max_power - self.min_power)
        return normalized * 2.0 - 1.0

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        normalized = (x + 1.0) * 0.5
        return normalized * (self.max_power - self.min_power) + self.min_power


def resolve_voxel_path(record: str | Path) -> Path:
    path = Path(record)
    if path.is_dir():
        return path / "voxel_grid.npy"
    return path


def resolve_laser_power_path(record: str | Path) -> Path:
    path = Path(record)
    if path.is_dir():
        return path / "laser_power_per_frame.npy"
    return path.with_name("laser_power_per_frame.npy")


def discover_files(data_dir: str | Path) -> list[Path]:
    """Discover recording folders, falling back to legacy flat ``.npy`` files."""

    root = Path(data_dir).expanduser()
    records = sorted(path for path in root.iterdir() if path.is_dir() and (path / "voxel_grid.npy").exists())
    if records:
        return records
    files = sorted(root.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No recording folders or .npy files found in {root.resolve()}")
    return files


def split_files(
    files: Sequence[Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Split by recording file so windows from one recording never leak."""

    if not 0 <= val_ratio < 1 or not 0 <= test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    if len(files) < 3:
        raise ValueError("At least three files are required for train/val/test splitting")
    shuffled = list(files)
    np.random.default_rng(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_ratio))
    n_val = max(1, round(len(shuffled) * val_ratio))
    if n_test + n_val >= len(shuffled):
        raise ValueError("The requested split leaves no training files")
    return shuffled[n_test + n_val :], shuffled[n_test : n_test + n_val], shuffled[:n_test]


def infer_frame_shape(array: np.ndarray) -> tuple[str, tuple[int, int, int], int]:
    """Infer layout, ``(C,H,W)`` frame shape, and frame count."""

    if array.ndim != 4:
        raise ValueError(f"Expected a 4D voxel array, got shape {array.shape}")
    if array.shape[1] <= 32 and array.shape[1] <= min(array.shape[2], array.shape[3]):
        return "nchw", (array.shape[1], array.shape[2], array.shape[3]), array.shape[0]
    if array.shape[-1] <= 32 and array.shape[-1] <= min(array.shape[1], array.shape[2]):
        return "nhwc", (array.shape[-1], array.shape[1], array.shape[2]), array.shape[0]
    raise ValueError(f"Cannot infer NCHW/NHWC layout for shape {array.shape}")


@dataclass(frozen=True)
class WindowRef:
    file_index: int
    start_index: int


class EventSequenceDataset(Dataset[dict[str, torch.Tensor | str | int]]):
    """Return ``condition`` and ``target`` event voxel sequences.

    Each item contains:
    ``condition`` with shape ``(cond_frames,C,H,W)`` and ``target`` with shape
    ``(pred_frames,C,H,W)`` after signed-log normalization.
    When ``laser_normalizer`` is provided, it also returns normalized
    ``laser_power`` with shape ``(cond_frames + pred_frames,)``.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        normalizer: SignedLogNormalizer,
        cond_frames: int = 3,
        pred_frames: int = 25,
        frame_stride: int = 1,
        window_stride: int = 1,
        spatial_size: int | None = None,
        max_windows_per_file: int | None = None,
        laser_normalizer: LaserPowerNormalizer | None = None,
    ) -> None:
        if cond_frames <= 0:
            raise ValueError("cond_frames must be positive")
        if pred_frames <= 0:
            raise ValueError("pred_frames must be positive")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if spatial_size is not None and spatial_size <= 0:
            raise ValueError("spatial_size must be positive when provided")
        if max_windows_per_file is not None and max_windows_per_file <= 0:
            raise ValueError("max_windows_per_file must be positive when provided")

        self.files = [Path(path) for path in files]
        self.normalizer = normalizer
        self.cond_frames = int(cond_frames)
        self.pred_frames = int(pred_frames)
        self.frame_stride = int(frame_stride)
        self.window_stride = int(window_stride)
        self.spatial_size = spatial_size
        self.max_windows_per_file = max_windows_per_file
        self.laser_normalizer = laser_normalizer
        self.total_frames = self.cond_frames + self.pred_frames
        self._arrays: dict[int, np.ndarray] = {}
        self._laser_arrays: dict[int, np.ndarray] = {}
        self.layouts: list[str] = []
        self.frame_counts: list[int] = []
        self.windows: list[WindowRef] = []

        expected_shape: tuple[int, int, int] | None = None
        required_span = (self.total_frames - 1) * self.frame_stride + 1
        for file_index, path in enumerate(self.files):
            voxel_path = resolve_voxel_path(path)
            if not voxel_path.exists():
                raise FileNotFoundError(f"Voxel grid not found: {voxel_path}")
            array = np.load(voxel_path, mmap_mode="r")
            layout, frame_shape, frame_count = infer_frame_shape(array)
            laser_path = resolve_laser_power_path(path)
            if self.laser_normalizer is not None:
                if not laser_path.exists():
                    raise FileNotFoundError(f"Laser power file not found: {laser_path}")
                laser = np.load(laser_path, mmap_mode="r")
                if laser.ndim != 1:
                    raise ValueError(f"Expected 1D laser power array, got {laser.shape} in {laser_path}")
                if len(laser) != frame_count:
                    raise ValueError(
                        f"Laser/frame length mismatch in {path}: {len(laser)} vs {frame_count}"
                    )
            if frame_count < required_span:
                raise ValueError(
                    f"{path} has only {frame_count} frames, but {required_span} are needed"
                )
            if expected_shape is None:
                expected_shape = frame_shape
            elif frame_shape != expected_shape:
                raise ValueError(f"Inconsistent frame shape {frame_shape} in {path}")
            self.layouts.append(layout)
            self.frame_counts.append(frame_count)
            max_start = frame_count - required_span
            starts = list(range(0, max_start + 1, self.window_stride))
            if self.max_windows_per_file is not None and len(starts) > self.max_windows_per_file:
                positions = np.linspace(0, len(starts) - 1, self.max_windows_per_file)
                starts = [starts[int(round(pos))] for pos in positions]
            self.windows.extend(WindowRef(file_index, start) for start in starts)

        if expected_shape is None:
            raise ValueError("No files were provided")
        if spatial_size is None:
            self.frame_shape = expected_shape
        else:
            self.frame_shape = (expected_shape[0], spatial_size, spatial_size)
        self.source_frame_shape = expected_shape

    def __len__(self) -> int:
        return len(self.windows)

    def _array(self, file_index: int) -> np.ndarray:
        if file_index not in self._arrays:
            self._arrays[file_index] = np.load(resolve_voxel_path(self.files[file_index]), mmap_mode="r")
        return self._arrays[file_index]

    def _laser_array(self, file_index: int) -> np.ndarray:
        if file_index not in self._laser_arrays:
            self._laser_arrays[file_index] = np.load(
                resolve_laser_power_path(self.files[file_index]), mmap_mode="r"
            )
        return self._laser_arrays[file_index]

    def _read_sequence(self, ref: WindowRef) -> torch.Tensor:
        array = self._array(ref.file_index)
        indices = ref.start_index + np.arange(self.total_frames) * self.frame_stride
        sequence = np.asarray(array[indices], dtype=np.float32)
        if self.layouts[ref.file_index] == "nhwc":
            sequence = np.moveaxis(sequence, -1, 1)
        tensor = self.normalizer.encode(torch.from_numpy(np.array(sequence, copy=True)))
        if self.spatial_size is not None and tensor.shape[-1] != self.spatial_size:
            tensor = F.interpolate(
                tensor,
                size=(self.spatial_size, self.spatial_size),
                mode="bilinear",
                align_corners=False,
            )
        return tensor

    def _read_laser_power(self, ref: WindowRef) -> tuple[torch.Tensor, torch.Tensor]:
        if self.laser_normalizer is None:
            raise RuntimeError("laser_normalizer is not configured")
        array = self._laser_array(ref.file_index)
        indices = ref.start_index + np.arange(self.total_frames) * self.frame_stride
        values = np.asarray(array[indices], dtype=np.float32)
        raw = torch.from_numpy(np.array(values, copy=True))
        return self.laser_normalizer.encode(raw), raw

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ref = self.windows[index]
        sequence = self._read_sequence(ref)
        condition = sequence[: self.cond_frames]
        target = sequence[self.cond_frames :]
        return {
            "condition": condition,
            "target": target,
            "name": self.files[ref.file_index].stem,
            "start_index": ref.start_index,
        } | (
            {}
            if self.laser_normalizer is None
            else dict(zip(("laser_power", "laser_power_raw"), self._read_laser_power(ref)))
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
    )


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without duplicate padding."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


class StridedSubsetSampler(Sampler[int]):
    """Take a deterministic evenly-spaced subset for quick validation/testing."""

    def __init__(self, dataset: Dataset, max_samples: int | None) -> None:
        self.dataset = dataset
        if max_samples is None or max_samples >= len(dataset):
            self.indices = list(range(len(dataset)))
        elif max_samples <= 0:
            raise ValueError("max_samples must be positive")
        else:
            positions = np.linspace(0, len(dataset) - 1, max_samples)
            self.indices = [int(round(pos)) for pos in positions]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
