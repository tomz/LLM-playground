"""Parallelism abstraction. Only `torch_native` (single-process + optional DDP)
is wired here; other backends raise NotImplementedError with an informative
message pointing at the module you'd need to install/depend on."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ParallelConfig:
    backend: str = "torch_native"
    dp: int = 1
    tp: int = 1
    pp: int = 1
    sp: bool = False
    ep: int = 1
    cp: int = 1
    zero_stage: int = 3
    activation_recompute: str = "selective"
    grad_clip: float = 1.0  # mirrored from OptimConfig; set by Trainer

    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp


class ParallelEngine:
    """Thin wrapper that owns the model+optimizer and provides one
    `forward_backward` + `step` cycle per micro-batch."""

    def __init__(self, model, optimizer, cfg: ParallelConfig):
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        if cfg.backend != "torch_native":
            raise NotImplementedError(
                f"backend {cfg.backend!r} requires a different runtime "
                "(megatron-core / nemo / deepspeed); only 'torch_native' is wired."
            )
        if cfg.tp > 1 or cfg.pp > 1:
            raise NotImplementedError(
                "tensor/pipeline parallel not implemented in torch_native engine"
            )
        self.cfg = cfg
        self.optimizer = optimizer
        self.model = model
        if cfg.dp > 1 and dist.is_available() and dist.is_initialized():
            self.model = DDP(model)
        self._dist = dist

    def forward_backward(self, batch) -> dict:
        input_ids, targets = batch
        logits, loss = self.model(input_ids, targets=targets)
        loss.backward()
        return {
            "loss": float(loss.detach()),
            "tokens": int(input_ids.numel()),
        }

    def step(self) -> None:
        import torch

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def all_reduce_metric(self, value: float) -> float:
        if self._dist.is_available() and self._dist.is_initialized():
            import torch

            t = torch.tensor([value], dtype=torch.float32)
            self._dist.all_reduce(t, op=self._dist.ReduceOp.SUM)
            return float(t.item()) / self._dist.get_world_size()
        return float(value)
