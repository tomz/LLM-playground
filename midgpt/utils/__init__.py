"""Utilities: cosine LR, JSONL logger, simple ckpt manager."""
from __future__ import annotations
import json, math, os, time
import tempfile


def cosine_lr(it, warmup, decay, lr, min_lr):
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > decay:
        return min_lr
    p = (it - warmup) / max(1, decay - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


def wsd_lr(it, warmup, total, lr, min_lr, decay_frac=0.2, shape="linear"):
    """Warmup-stable-decay (Hägele et al. 2024): linear warmup, constant
    plateau, cooldown to ``min_lr`` over the final ``decay_frac`` of
    post-warmup steps. ``shape``: ``linear`` or ``1-sqrt``."""
    if not 0.0 < decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be in (0, 1], got {decay_frac}")
    if shape not in {"linear", "1-sqrt"}:
        raise ValueError(f"unknown WSD decay shape {shape!r}")
    if it < warmup:
        return lr * (it + 1) / max(1, warmup)
    if it >= total:
        return min_lr
    decay_steps = max(1, int(round(decay_frac * max(1, total - warmup))))
    start = total - decay_steps
    if it < start:
        return lr
    p = (it - start) / decay_steps
    f = 1.0 - p if shape == "linear" else 1.0 - math.sqrt(p)
    return min_lr + (lr - min_lr) * f


def scheduled_lr(it, optim):
    """Dispatch on ``optim['lr_schedule']`` (default ``cosine``)."""
    kind = optim.get("lr_schedule", "cosine")
    args = (it, optim["warmup_iters"], optim["lr_decay_iters"], optim["lr"], optim["min_lr"])
    if kind == "cosine":
        return cosine_lr(*args)
    if kind == "wsd":
        return wsd_lr(*args, decay_frac=optim.get("wsd_decay_frac", 0.2),
                      shape=optim.get("wsd_decay_shape", "linear"))
    raise ValueError(f"unknown lr_schedule {kind!r}; expected 'cosine' or 'wsd'")


class JsonlLogger:
    def __init__(self, path: str | None):
        self.fh = open(path, "a") if path else None
        self.t0 = time.time()

    def log(self, **kw):
        kw["wall"] = round(time.time() - self.t0, 3)
        if self.fh:
            self.fh.write(json.dumps(kw) + "\n"); self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def save_ckpt(path: str, state: dict) -> None:
    """Publish a complete file atomically; keep the old file on write failure."""
    import torch
    directory = os.path.dirname(os.path.abspath(path))
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=directory, prefix=".ckpt-", suffix=".tmp", delete=False,
        ) as fh:
            tmp = fh.name
            torch.save(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        tmp = None
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
