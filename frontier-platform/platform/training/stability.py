"""Loss-spike detector & auto-rewind controller."""
from __future__ import annotations
from collections import deque


class SpikeMonitor:
    def __init__(self, window: int = 200, sigma: float = 4.0):
        self.window = window
        self.sigma = sigma
        self._buf: deque[float] = deque(maxlen=window)

    def observe(self, loss: float) -> bool:
        """Return True if `loss` is a spike worth alarming on."""
        if len(self._buf) < self.window:
            self._buf.append(loss)
            return False
        mean = sum(self._buf) / len(self._buf)
        var = sum((x - mean) ** 2 for x in self._buf) / len(self._buf)
        std = max(var ** 0.5, 0.05 * abs(mean) + 1e-6)  # floor: ignore <5% jitter on flat losses
        is_spike = (loss - mean) > self.sigma * std
        self._buf.append(loss)
        return is_spike


class RewindController:
    """On spike: roll back N steps, halve LR for K steps, skip 1k batches."""
    def __init__(self, ckpt_mgr, lr_floor: float = 1e-6):
        self.ckpt_mgr = ckpt_mgr
        self.lr_floor = lr_floor
    def on_spike(self, engine, current_step: int) -> int:
        raise NotImplementedError
