"""Shared utility functions."""

from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA is not available")
    return torch.device(requested)


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(requested_device: str) -> DistributedContext:
    """Initialize an ``env://`` process group when launched with torchrun."""

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not distributed:
        return DistributedContext(False, 0, 0, 1, resolve_device(requested_device))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    use_cuda = requested_device != "cpu" and torch.cuda.is_available()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA is not available")
    if use_cuda:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(True, rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "no":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError("precision must be one of: no, fp16, bf16")


def make_grad_scaler(device: torch.device, enabled: bool):
    enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def serializable_args(namespace) -> dict[str, Any]:
    result = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result
