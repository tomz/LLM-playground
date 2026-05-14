"""Parallelism abstraction. Backends: megatron | nemo | deepspeed | torch_native."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ParallelConfig:
    backend: str = "torch_native"
    dp: int = 1
    tp: int = 1
    pp: int = 1
    sp: bool = False
    ep: int = 1     # expert parallel for MoE
    cp: int = 1     # context parallel for very long ctx
    zero_stage: int = 3  # 0,1,2,3 (3 = full FSDP)
    activation_recompute: str = "selective"  # 'none'|'selective'|'full'

    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp


class ParallelEngine:
    """Wraps a model + optimizer with the configured parallelism strategy."""
    def __init__(self, model, optimizer, cfg: ParallelConfig): ...
    def forward_backward(self, batch) -> dict:
        """Runs one micro-batch (or a pipeline schedule) and returns metrics."""
        raise NotImplementedError
    def step(self) -> None:
        raise NotImplementedError
    def all_reduce_metric(self, value: float) -> float:
        raise NotImplementedError
