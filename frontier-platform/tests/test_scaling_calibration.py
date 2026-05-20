"""Lock down the simulator's headline numbers so accidental re-fits get caught.

These are the **raw** predictor outputs (no sft/rlhf multipliers, no noise) —
the numbers in the README are slightly different because the orchestrator
multiplies by sft_quality / rlhf_quality and adds a small gaussian. If you
intentionally retune `scaling.py`, regenerate these expected values.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from platform.sim.scaling import (
    chinchilla_loss, predict_mmlu, predict_humaneval, predict_gsm8k,
)

# (n_params, n_tokens, mmlu, humaneval, gsm8k, loss). Values captured from
# the deterministic predictors at HEAD; tolerances catch curve-shape drift.
SIZES = [
    (1e9,  1e12,  0.493, 0.251, 0.200, 2.223),
    (7e9,  2e12,  0.624, 0.429, 0.375, 2.020),
    (7e10, 5e12,  0.761, 0.662, 0.651, 1.888),
    (4e11, 2e13,  0.839, 0.806, 0.836, 1.814),
]


def _close(actual: float, expected: float, atol: float = 0.005) -> bool:
    return abs(actual - expected) <= atol


def test_eval_predictors_match_locked_numbers():
    for n, d, mmlu, he, gsm, _loss in SIZES:
        m = predict_mmlu(n, d)
        h = predict_humaneval(n, d)
        g = predict_gsm8k(n, d)
        assert _close(m, mmlu), f"MMLU drift @ ({n:.0e},{d:.0e}): {m:.3f} vs {mmlu:.3f}"
        assert _close(h, he),   f"HumanEval drift @ ({n:.0e},{d:.0e}): {h:.3f} vs {he:.3f}"
        assert _close(g, gsm),  f"GSM8K drift @ ({n:.0e},{d:.0e}): {g:.3f} vs {gsm:.3f}"


def test_loss_predictors_match_locked_numbers():
    for n, d, _, _, _, expected_loss in SIZES:
        L = chinchilla_loss(n, d)
        assert _close(L, expected_loss, atol=0.005), \
            f"loss drift @ ({n:.0e},{d:.0e}): {L:.4f} vs {expected_loss:.4f}"


def test_mmlu_floor_and_ceiling():
    """Sanity bounds: MMLU should never drop below random or exceed near-1."""
    assert predict_mmlu(1e3, 1e6) >= 0.25 - 1e-6      # random chance floor
    assert predict_mmlu(1e15, 1e16) <= 0.91           # asymptote
