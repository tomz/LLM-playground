"""AdamW with WD-by-dim heuristic + cosine-with-warmup LR schedule."""
from __future__ import annotations
from dataclasses import dataclass

import math

import torch


@dataclass
class OptimConfig:
    name: str = "adamw"          # 'adamw' | 'muon' (Muon on 2D hidden + AdamW rest)
    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    total_steps: int = 500_000
    # Muon-specific (used when name == 'muon'). Muon's hidden-matrix LR is
    # typically higher than AdamW's; the IO/AdamW group uses peak_lr.
    muon_lr: float = 0.02
    muon_momentum: float = 0.95


def cosine_with_warmup(step: int, cfg: OptimConfig) -> float:
    """Returns LR multiplier in [min_lr_ratio, 1.0]."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos


def build_optimizer(model, cfg: OptimConfig):
    """Return (optimizer, scheduler).

    ``name == 'adamw'`` (default): single AdamW; ``dim >= 2`` params get weight
    decay, 1-D params don't.

    ``name == 'muon'``: Muon on 2D hidden weight matrices (~1.35x sample
    efficiency) + AdamW on embeddings/lm_head/MTP heads/router gates and all 1-D
    params. Returns a combined optimizer facade so the rest of the training stack
    is unchanged.
    """
    import torch

    if cfg.name == "muon":
        return _build_muon(model, cfg)

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


class _CombinedOptimizer(torch.optim.Optimizer):
    """Minimal facade over multiple torch optimizers (Muon + AdamW).

    Subclasses ``Optimizer`` so ``LambdaLR`` accepts it, but bypasses the base
    ``__init__`` and delegates ``param_groups``/``step``/``zero_grad``/state to
    the wrapped optimizers. ``LambdaLR`` reads ``param_groups`` to capture
    ``base_lrs`` and to write scaled LRs, both of which proxy through.
    """

    def __init__(self, optimizers: list):
        self.optimizers = optimizers
        self.state = {}
        self.defaults = {}

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    @param_groups.setter
    def param_groups(self, _value):
        # LambdaLR.__init__ may try to set this; ignore — groups live in the
        # wrapped optimizers and are returned live by the getter.
        pass

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, sd):
        for opt, osd in zip(self.optimizers, sd["optimizers"]):
            opt.load_state_dict(osd)


def _build_muon(model, cfg: OptimConfig):
    import torch

    from .muon import Muon, split_muon_params

    muon_params, adamw_params = split_muon_params(model)
    decay = [p for p in adamw_params if p.dim() >= 2]
    no_decay = [p for p in adamw_params if p.dim() < 2]
    fused = bool(torch.cuda.is_available())
    optimizers = []
    if muon_params:
        optimizers.append(Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum))
    adamw_groups = []
    if decay:
        adamw_groups.append({"params": decay, "weight_decay": cfg.weight_decay})
    if no_decay:
        adamw_groups.append({"params": no_decay, "weight_decay": 0.0})
    if adamw_groups:
        optimizers.append(
            torch.optim.AdamW(adamw_groups, lr=cfg.peak_lr, betas=tuple(cfg.betas),
                              eps=cfg.eps, fused=fused)
        )
    combined = _CombinedOptimizer(optimizers)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        combined, lr_lambda=lambda step: cosine_with_warmup(step, cfg)
    )
    return combined, scheduler
