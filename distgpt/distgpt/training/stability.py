"""Checkpointable loss-spike history and bounded, collective trainer rewinds."""
from __future__ import annotations

from collections import deque
import math


class SpikeMonitor:
    """Require both a relative and absolute loss jump before rewinding."""

    def __init__(self, window: int = 200, sigma: float = 5.0,
                 min_abs_jump: float = 2.0):
        if window < 1 or sigma <= 0 or min_abs_jump < 0:
            raise ValueError("invalid spike monitor thresholds")
        self.window, self.sigma, self.min_abs_jump = window, sigma, min_abs_jump
        self._buf: deque[float] = deque(maxlen=window)

    def observe(self, loss: float) -> bool:
        if not math.isfinite(loss):
            raise ValueError("nonfinite loss must be handled collectively before spike observation")
        if len(self._buf) < self.window:
            self._buf.append(loss)
            return False
        mean = sum(self._buf) / len(self._buf)
        var = sum((x - mean) ** 2 for x in self._buf) / len(self._buf)
        std = max(var ** 0.5, 0.05 * abs(mean) + 1e-6)
        jump = loss - mean
        spike = jump > self.sigma * std and jump > self.min_abs_jump
        self._buf.append(loss)
        return spike

    def state_dict(self) -> dict:
        return {"window": self.window, "sigma": self.sigma,
                "min_abs_jump": self.min_abs_jump, "history": list(self._buf)}

    def load_state_dict(self, state: dict) -> None:
        if any(state[key] != getattr(self, key) for key in ("window", "sigma", "min_abs_jump")):
            raise ValueError("spike monitor configuration changed on resume")
        history = state["history"]
        if len(history) > self.window or not all(math.isfinite(x) for x in history):
            raise ValueError("invalid spike monitor history")
        self._buf = deque(history, maxlen=self.window)


class RewindController:
    """Reload at a completed-update boundary and reduce LR for 1000 updates.

    All ranks must call ``on_spike`` together. Rewind counts survive both job
    restart AND rollback; reloading old controller state must not reset the cap.
    Returning the unchanged completed-update count means no rewind took place.
    """

    def __init__(self, ckpt_mgr, lr_floor: float = 1e-3, max_rewinds: int = 5):
        if not 0 < lr_floor <= 1 or max_rewinds < 0:
            raise ValueError("invalid rewind limits")
        self.ckpt = ckpt_mgr
        self.lr_floor, self.max_rewinds = lr_floor, max_rewinds
        self.n_rewinds = 0
        self.cooldown_left = 0
        self.scale = 1.0

    def on_spike(self, model, optim, loader, current_step: int,
                 runtime_state: dict | None = None) -> int:
        if self.n_rewinds >= self.max_rewinds:
            return current_step
        last = self.ckpt.discover_latest()
        if last is None:
            return current_step
        next_step = self.ckpt.load(model, optim, loader, step=last,
                                   runtime_state=runtime_state)
        self.scale = max(self.scale * 0.5, self.lr_floor)
        self.cooldown_left = 1000
        self.n_rewinds += 1
        return next_step

    def lr_multiplier(self) -> float:
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return self.scale
        return 1.0

    def state_dict(self) -> dict:
        return {"lr_floor": self.lr_floor, "max_rewinds": self.max_rewinds,
                "n_rewinds": self.n_rewinds, "cooldown_left": self.cooldown_left,
                "scale": self.scale}

    def load_state_dict(self, state: dict) -> None:
        if state["lr_floor"] != self.lr_floor or state["max_rewinds"] != self.max_rewinds:
            raise ValueError("rewind configuration changed on resume")
        if (type(state["n_rewinds"]) is not int or not 0 <= state["n_rewinds"] <= self.max_rewinds
                or type(state["cooldown_left"]) is not int or not 0 <= state["cooldown_left"] <= 1000
                or not math.isfinite(state["scale"]) or not self.lr_floor <= state["scale"] <= 1):
            raise ValueError("invalid rewind controller state")
        self.n_rewinds = state["n_rewinds"]
        self.cooldown_left = state["cooldown_left"]
        self.scale = state["scale"]
