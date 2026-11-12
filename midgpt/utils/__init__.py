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
