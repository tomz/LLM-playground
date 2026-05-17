"""Top-level pretraining loop."""
from __future__ import annotations
from dataclasses import dataclass, field

from .optim import OptimConfig
from .parallel import ParallelConfig, ParallelEngine
from .stability import SpikeMonitor, RewindController


@dataclass
class TrainConfig:
    run_id: str
    seq_len: int = 8192
    micro_batch: int = 1
    global_batch_tokens: int = 4_000_000
    total_tokens: int = 2_000_000_000_000
    log_every: int = 10
    eval_every: int = 1000
    ckpt_every: int = 1000
    optim: OptimConfig = field(default_factory=OptimConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        ckpt_mgr,
        evaluator,
        cfg: TrainConfig,
        optimizer=None,
        scheduler=None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.ckpt_mgr = ckpt_mgr
        self.evaluator = evaluator
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.spike = SpikeMonitor()
        self.loss_history: list[float] = []

    def fit(self) -> dict:
        cfg = self.cfg
        if self.optimizer is None:
            raise ValueError("Trainer requires `optimizer` (build via build_optimizer)")

        # propagate grad-clip into the ParallelConfig
        cfg.parallel.grad_clip = cfg.optim.grad_clip

        start_step = (self.ckpt_mgr.latest() if self.ckpt_mgr is not None else None) or 0
        engine = ParallelEngine(self.model, self.optimizer, cfg.parallel)
        rewind = RewindController(self.ckpt_mgr) if self.ckpt_mgr is not None else None

        tokens_seen = 0
        metrics: dict = {"loss": float("nan"), "tokens": 0, "step": start_step}
        self.model.train()

        for step, batch in enumerate(self.dataloader, start=start_step):
            # batch is (np.ndarray, np.ndarray) from StreamingLoader; convert.
            batch = self._to_tensors(batch)

            metrics = engine.forward_backward(batch)
            engine.step()
            if self.scheduler is not None:
                self.scheduler.step()
            tokens_seen += metrics["tokens"]
            metrics["step"] = step
            self.loss_history.append(metrics["loss"])

            if self.spike.observe(metrics["loss"]) and rewind is not None:
                rewind.on_spike(engine, step)

            if self.evaluator is not None and cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
                self.evaluator.run_fast(engine, step)
            if self.ckpt_mgr is not None and cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
                self.ckpt_mgr.save_async(engine, self.dataloader, step)

            if tokens_seen >= cfg.total_tokens:
                break
            if step + 1 >= cfg.optim.total_steps:
                break

        return metrics

    # ---- internals -------------------------------------------------------

    def _to_tensors(self, batch):
        import torch
        import numpy as np

        x, y = batch
        device = next(self.model.parameters()).device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).long()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        return x.to(device), y.to(device)
