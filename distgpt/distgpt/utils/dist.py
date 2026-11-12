"""Process-group lifecycle with explicit device placement and group reductions."""
from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def init() -> tuple[int, int, int]:
    """Initialize NCCL or Gloo; honor an already initialized process group."""
    world = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", 1))
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(0 if os.environ.get("DISTGPT_COLOCATE_RANKS") == "1" else local_rank)
    if world > 1 and not dist.is_initialized():
        backend = os.environ.get("DISTGPT_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=backend, rank=rank, world_size=world,
                                timeout=timedelta(seconds=180))
    return rank, local_rank, world


def destroy() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master(rank: int = 0) -> bool:
    return not dist.is_initialized() or dist.get_rank() == rank


def all_reduce_mean(t: torch.Tensor, group=None) -> torch.Tensor:
    """Average only the explicitly supplied group (None means world)."""
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        t /= dist.get_world_size(group)
    return t
