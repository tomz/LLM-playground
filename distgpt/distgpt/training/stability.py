"""Loss-spike detector and rewind controller.

Spike detection has two thresholds that BOTH must be exceeded before we
declare a spike:
  * relative: `loss - mean > sigma * std` (the classic z-score test)
  * absolute: `loss - mean > min_abs_jump` (guards against runaway rewind
    loops once the loss curve has plateaued and the running std is tiny)

The rewinder also caps total rewinds at `max_rewinds` so a chronically
spiky model doesn't get LR-multiplied into oblivion. Once that cap fires
the rest of the run continues at the cosine LR with no further rewinds.

Real bug history that motivated these guards
--------------------------------------------
On the first 416M FineWeb-Edu run, train loss plateaued at ~4.88 with
per-step jitter of ~0.3 (small effective batch -> noisy gradient). With
the old SpikeMonitor (sigma=5.0, no absolute floor) a single 0.3-σ jump
fired as a spike (std ~0.04 once converged -> 0.3 ~ 7σ), triggered the
rewinder, halved LR scale to 0.5x, trained back to the same plateau,
fired again, halved to 0.25x ... after 11 rewinds eff_lr was ~1e-10
and the cosine schedule was meaningless. The run wasted ~6 hours
training in a loop on the same 100 steps. min_abs_jump + max_rewinds
together make that failure mode impossible.
"""
from __future__ import annotations
from collections import deque


class SpikeMonitor:
    def __init__(self, window: int = 200, sigma: float = 5.0,
                 min_abs_jump: float = 2.0):
        self.window = window
        self.sigma = sigma
        self.min_abs_jump = min_abs_jump
        self._buf: deque[float] = deque(maxlen=window)

    def observe(self, loss: float) -> bool:
        if len(self._buf) < self.window:
            self._buf.append(loss)
            return False
        mean = sum(self._buf) / len(self._buf)
        var = sum((x - mean) ** 2 for x in self._buf) / len(self._buf)
        std = max(var ** 0.5, 0.05 * abs(mean) + 1e-6)
        jump = loss - mean
        spike = (jump > self.sigma * std) and (jump > self.min_abs_jump)
        self._buf.append(loss)
        return spike


class RewindController:
    """On spike: reload last checkpoint, halve LR for K steps.

    `max_rewinds` caps total rewinds across the whole run -- once exceeded
    we stop rewinding (and stop reducing the LR multiplier) so a spike-
    happy model can still finish training at the cosine schedule's LR.
    """
    def __init__(self, ckpt_mgr, lr_floor: float = 1e-3, max_rewinds: int = 5):
        self.ckpt = ckpt_mgr
        self.lr_floor = lr_floor
        self.max_rewinds = max_rewinds
        self.n_rewinds = 0
        self.cooldown_left = 0
        self.scale = 1.0

    def on_spike(self, model, optim, loader, current_step: int) -> int:
        if self.n_rewinds >= self.max_rewinds:
            return current_step
        last = self.ckpt.latest()
        if last is None:
            return current_step
        next_step = self.ckpt.load(model, optim, loader, step=last)
        self.scale = max(self.scale * 0.5, self.lr_floor)
        self.cooldown_left = 1000
        self.n_rewinds += 1
        return next_step

    def lr_multiplier(self) -> float:
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return self.scale
        return 1.0
