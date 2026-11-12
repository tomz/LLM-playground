"""Loss-spike detector & auto-rewind controller."""
from __future__ import annotations
from collections import deque


class SpikeMonitor:
    def __init__(self, window: int = 200, sigma: float = 4.0):
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
        is_spike = (loss - mean) > self.sigma * std
        self._buf.append(loss)
        return is_spike


class RewindController:
    """On spike: roll back to the last good checkpoint, halve LR."""

    def __init__(self, ckpt_mgr, lr_floor: float = 1e-6):
        self.ckpt_mgr = ckpt_mgr
        self.lr_floor = float(lr_floor)

    def on_spike(self, engine, current_step: int) -> int:
        latest = self.ckpt_mgr.latest() if self.ckpt_mgr is not None else None
        if latest is not None:
            try:
                self.ckpt_mgr.load_into(engine, getattr(engine, "dataloader", None), step=latest)
            except (AttributeError, FileNotFoundError):
                pass
        # Halve LR (floored).
        if engine is not None and getattr(engine, "optimizer", None) is not None:
            for pg in engine.optimizer.param_groups:
                pg["lr"] = max(self.lr_floor, pg["lr"] * 0.5)
        return latest if latest is not None else current_step
