"""Utilities: cosine LR, JSONL logger, simple ckpt manager."""
import json, math, os, time


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


def save_ckpt(path: str, state: dict):
    tmp = path + ".tmp"
    import torch
    torch.save(state, tmp)
    os.replace(tmp, path)
