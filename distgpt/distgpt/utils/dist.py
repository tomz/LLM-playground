"""Process-group / distributed init helpers."""
from __future__ import annotations
import os
import torch
import torch.distributed as dist


def init() -> tuple[int, int, int]:
    """Init NCCL (or gloo on CPU). Return (rank, local_rank, world_size)."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def destroy() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_master(rank: int = 0) -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == rank


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return t
