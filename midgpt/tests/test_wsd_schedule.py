"""WSD (warmup-stable-decay) schedule contract tests."""
from __future__ import annotations
import math, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import cosine_lr, scheduled_lr, wsd_lr

W, T, LR, MIN = 10, 110, 1.0, 0.1

def cfg(**kw):
    c = {"warmup_iters": W, "lr_decay_iters": T, "lr": LR, "min_lr": MIN}
    c.update(kw); return c

def test_warmup_matches_cosine():
    for it in range(W):
        assert wsd_lr(it, W, T, LR, MIN) == pytest.approx(cosine_lr(it, W, T, LR, MIN))

def test_plateau_is_constant_peak():
    assert all(wsd_lr(it, W, T, LR, MIN) == LR for it in range(W, 90))

def test_linear_cooldown_monotone_to_min():
    vals = [wsd_lr(it, W, T, LR, MIN) for it in range(90, T)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert wsd_lr(T, W, T, LR, MIN) == MIN
    assert wsd_lr(100, W, T, LR, MIN) == pytest.approx(MIN + (LR - MIN) * 0.5)

def test_one_minus_sqrt_shape():
    got = wsd_lr(95, W, T, LR, MIN, shape="1-sqrt")
    assert got == pytest.approx(MIN + (LR - MIN) * (1 - math.sqrt(0.25)))

def test_plateau_independent_of_horizon_for_branching():
    assert wsd_lr(50, W, T, LR, MIN) == wsd_lr(50, W, 10_000, LR, MIN)

def test_dispatch_default_is_cosine_and_wsd_opt_in():
    assert scheduled_lr(40, cfg()) == cosine_lr(40, W, T, LR, MIN)
    assert scheduled_lr(40, cfg(lr_schedule="wsd")) == LR
    assert scheduled_lr(95, cfg(lr_schedule="wsd", wsd_decay_shape="1-sqrt")) == wsd_lr(95, W, T, LR, MIN, shape="1-sqrt")

@pytest.mark.parametrize("bad", [dict(lr_schedule="step"), dict(lr_schedule="wsd", wsd_decay_frac=0.0),
                                 dict(lr_schedule="wsd", wsd_decay_frac=1.5), dict(lr_schedule="wsd", wsd_decay_shape="cos")])
def test_invalid_settings_rejected(bad):
    with pytest.raises(ValueError):
        scheduled_lr(0, cfg(**bad))
