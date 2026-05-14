from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OptimConfig:
    name: str = "adamw"   # 'adamw' | 'lion' | 'shampoo'
    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 500_000


def build_optimizer(model, cfg: OptimConfig):
    """Return (optimizer, scheduler). Excludes biases, norms, embeddings from WD."""
    raise NotImplementedError


def cosine_with_warmup(step: int, cfg: OptimConfig) -> float:
    """Returns LR multiplier in [min_lr_ratio, 1.0]."""
    import math
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos
