"""Top-level pretraining loop. Production runs days-to-months; must be boring."""
from __future__ import annotations
from dataclasses import dataclass, field
from .optim import OptimConfig  # cosine_with_warmup used inside fit() pseudocode
from .parallel import ParallelConfig
from .stability import SpikeMonitor


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
    def __init__(self, model, dataloader, ckpt_mgr, evaluator, cfg: TrainConfig):
        self.model = model
        self.dataloader = dataloader
        self.ckpt_mgr = ckpt_mgr
        self.evaluator = evaluator
        self.cfg = cfg
        self.spike = SpikeMonitor()

    def fit(self) -> None:
        """Main loop. Pseudocode:

            engine = ParallelEngine(model, optim, cfg.parallel)
            for step, batch in enumerate(dataloader, start=resume_step):
                metrics = engine.forward_backward(batch)
                engine.step()
                if self.spike.observe(metrics['loss']):
                    rewind.on_spike(engine, step)
                if step % cfg.log_every == 0:
                    log(metrics, lr=cosine_with_warmup(step, cfg.optim) * cfg.optim.peak_lr)
                if step % cfg.eval_every == 0:
                    evaluator.run_fast(engine, step)
                if step % cfg.ckpt_every == 0:
                    ckpt_mgr.save_async(engine, dataloader, step)
                if tokens_seen >= cfg.total_tokens:
                    break
        """
        raise NotImplementedError
