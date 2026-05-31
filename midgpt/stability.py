"""Loss-spike detector and rewind controller.

Long pretraining runs (the 2.5h GPT-2-350M FineWeb-Edu run is the worked
example here) occasionally see a loss "spike" — a sudden jump well outside
the running noise floor — that, if left alone, can permanently destabilise
the model. The standard recipe (see modded-nanogpt, OLMo, DeepSeek-V3):

  1. Detect the spike against a running window of recent losses.
  2. Rewind to the last good checkpoint.
  3. Restart with a halved LR for a short cooldown so the model glides
     back over the spike-inducing batch instead of slamming into it.

Spike detection has TWO thresholds that BOTH must trigger (ported from
distgpt; the rationale is documented there with a real bug history):

  * relative: ``loss - mean > sigma * std`` (classic z-score test)
  * absolute: ``loss - mean > min_abs_jump`` (guards against runaway
    rewind loops once the loss curve plateaus and the running std is tiny)

The rewinder caps total rewinds at ``max_rewinds`` so a chronically spiky
model doesn't get LR-halved into oblivion. Once that cap fires the rest of
the run continues at the cosine LR with no further rewinds.

Default-off in midgpt: existing recipes train without spike protection
(none have triggered in measured runs at this scale). Enable by setting
``stability.spike_monitor: true`` in a config; tests cover both modes.

Why this lives in midgpt (not just distgpt)
-------------------------------------------
midgpt's headline run (2.5h, 4000 iters) is short enough that *if* a spike
happens you lose most of the run. The whole point of a single-node trainer
is that a single losses-spike-driven rewind shouldn't require operator
intervention.
"""
from __future__ import annotations
from collections import deque
from typing import Callable


class SpikeMonitor:
    """Two-threshold spike detector. ``observe(loss)`` returns True when a
    spike fires; it always appends ``loss`` to the rolling window."""

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
        # Floor the std so a tightly-converged window can't pass the
        # sigma test on what's really just float jitter; the absolute
        # threshold below carries the real load there.
        std = max(var ** 0.5, 0.05 * abs(mean) + 1e-6)
        jump = loss - mean
        spike = (jump > self.sigma * std) and (jump > self.min_abs_jump)
        self._buf.append(loss)
        return spike


class RewindController:
    """On spike: reload the last checkpoint, halve LR for a cooldown window.

    ``load_ckpt_fn(path) -> int`` is supplied by the trainer; it should
    re-load model/optim/scaler/RNG state from ``path`` and return the
    iteration to resume from. ``last_ckpt_path_fn() -> str | None`` returns
    the path of the most recent good checkpoint (or None before any has
    landed).

    ``max_rewinds`` caps total rewinds across the run; once exceeded the
    controller stops rewinding (and stops scaling the LR down further) so
    a spike-happy model can still finish training.
    """

    def __init__(self,
                 load_ckpt_fn: Callable[[str], int],
                 last_ckpt_path_fn: Callable[[], str | None],
                 lr_floor: float = 1e-3,
                 max_rewinds: int = 5,
                 cooldown_steps: int = 1000):
        self._load = load_ckpt_fn
        self._last = last_ckpt_path_fn
        self.lr_floor = lr_floor
        self.max_rewinds = max_rewinds
        self.cooldown_steps = cooldown_steps
        self.n_rewinds = 0
        self.cooldown_left = 0
        self.scale = 1.0

    def on_spike(self, current_step: int) -> int:
        """Trigger a rewind. Returns the iteration to resume from (or
        ``current_step`` if no checkpoint exists / max_rewinds reached)."""
        if self.n_rewinds >= self.max_rewinds:
            return current_step
        path = self._last()
        if path is None:
            return current_step
        next_step = self._load(path)
        self.scale = max(self.scale * 0.5, self.lr_floor)
        self.cooldown_left = self.cooldown_steps
        self.n_rewinds += 1
        return next_step

    def lr_multiplier(self) -> float:
        """Return the current LR scale (1.0 outside cooldown). Call once per
        step — decrements the cooldown counter as a side-effect."""
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return self.scale
        return 1.0
