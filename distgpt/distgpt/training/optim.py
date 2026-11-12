"""AdamW with WD on weights only; cosine or warmup-stable-decay LR."""
from __future__ import annotations
import math
import torch


def build_optimizer(model, lr: float, betas, weight_decay: float, fused: bool):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=tuple(betas), fused=fused,
    )


def cosine_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


def wsd_lr(step: int, warmup: int, total: int, lr: float, min_lr: float,
           decay_frac: float = 0.2, shape: str = "linear") -> float:
    """Warmup-stable-decay (Hägele et al. 2024). Constant plateau, then a
    cooldown over the final ``decay_frac`` of post-warmup steps. ``shape``:
    ``linear`` or ``1-sqrt``. The plateau LR does not depend on ``total``, so a
    plateau checkpoint can branch into differently timed cooldowns."""
    if not 0.0 < decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be in (0, 1], got {decay_frac}")
    if shape not in {"linear", "1-sqrt"}:
        raise ValueError(f"unknown WSD decay shape {shape!r}")
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    decay_steps = max(1, int(round(decay_frac * max(1, total - warmup))))
    start = total - decay_steps
    if step < start:
        return lr
    p = (step - start) / decay_steps
    f = 1.0 - p if shape == "linear" else 1.0 - math.sqrt(p)
    return min_lr + (lr - min_lr) * f


def scheduled_lr(step: int, optim: dict) -> float:
    """Dispatch on ``optim['lr_schedule']`` (default ``cosine``)."""
    kind = optim.get("lr_schedule", "cosine")
    args = (step, optim["warmup_steps"], optim["total_steps"], optim["lr"], optim["min_lr"])
    if kind == "cosine":
        return cosine_lr(*args)
    if kind == "wsd":
        return wsd_lr(*args, decay_frac=optim.get("wsd_decay_frac", 0.2),
                      shape=optim.get("wsd_decay_shape", "linear"))
    raise ValueError(f"unknown optim.lr_schedule {kind!r}; expected 'cosine' or 'wsd'")
