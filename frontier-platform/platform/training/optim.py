"""AdamW with WD-by-dim heuristic + cosine-with-warmup LR schedule."""
from __future__ import annotations
from dataclasses import dataclass

import math


@dataclass
class OptimConfig:
    name: str = "adamw"
    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 500_000


def cosine_with_warmup(step: int, cfg: OptimConfig) -> float:
    """Returns LR multiplier in [min_lr_ratio, 1.0]."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos


def build_optimizer(model, cfg: OptimConfig):
    """Return (optimizer, scheduler). Params with `dim >= 2` get weight decay;
    biases, norms, and 1-D params don't."""
    import torch

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    fused = bool(torch.cuda.is_available())
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.peak_lr,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        fused=fused,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cosine_with_warmup(step, cfg)
    )
    return optimizer, scheduler
