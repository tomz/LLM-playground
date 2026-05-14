"""Loss-spike detector and rewind controller."""
from __future__ import annotations
from collections import deque


class SpikeMonitor:
    def __init__(self, window: int = 200, sigma: float = 5.0):
        self.window = window
        self.sigma = sigma
        self._buf: deque[float] = deque(maxlen=window)

    def observe(self, loss: float) -> bool:
        if len(self._buf) < self.window:
            self._buf.append(loss)
            return False
        mean = sum(self._buf) / len(self._buf)
        var = sum((x - mean) ** 2 for x in self._buf) / len(self._buf)
        std = max(var ** 0.5, 0.05 * abs(mean) + 1e-6)
        spike = (loss - mean) > self.sigma * std
        self._buf.append(loss)
        return spike


class RewindController:
    """On spike: reload last checkpoint, halve LR for K steps."""
    def __init__(self, ckpt_mgr, lr_floor: float = 1e-6):
        self.ckpt = ckpt_mgr
        self.lr_floor = lr_floor
        self.cooldown_left = 0
        self.scale = 1.0

    def on_spike(self, model, optim, loader, current_step: int) -> int:
        last = self.ckpt.latest()
        if last is None:
            return current_step
        next_step = self.ckpt.load(model, optim, loader, step=last)
        self.scale = max(self.scale * 0.5, self.lr_floor)
        self.cooldown_left = 1000
        return next_step

    def lr_multiplier(self) -> float:
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return self.scale
        return 1.0
