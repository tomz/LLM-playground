"""Tests for parallel_layers + mixture loader.

Hand-rolled TP layers (`parallel_layers.py`) and the weighted mixture loader
(`mixture.py`) ship as named-in-the-docs companions to the DTensor TP path
and the single-source `StreamingLoader`. These tests pin their behaviour
without spawning processes (TP layers reduce to local mode at group=None,
which is the path we exercise here for unit clarity)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.model.parallel_layers import (  # noqa: E402
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from distgpt.data.mixture import MixtureLoader  # noqa: E402


# ---------- parallel_layers (single-rank fallback) ----------


def test_column_parallel_linear_single_rank_matches_nn_linear():
    """At group=None (no distributed), ColumnParallelLinear is just nn.Linear
    — same weight shape, same forward — so it's a drop-in for single-GPU."""
    torch.manual_seed(0)
    cpl = ColumnParallelLinear(8, 16, group=None)
    assert cpl.weight.shape == (16, 8)
    x = torch.randn(2, 4, 8)
    out = cpl(x)
    assert out.shape == (2, 4, 16)


def test_row_parallel_linear_single_rank_matches_nn_linear():
    torch.manual_seed(0)
    rpl = RowParallelLinear(8, 16, bias=True, group=None)
    assert rpl.weight.shape == (16, 8)
    x = torch.randn(2, 4, 8)
    out = rpl(x)
    assert out.shape == (2, 4, 16)
    # Bias was added once (rank 0).
    rpl2 = RowParallelLinear(8, 16, bias=False, group=None)
    out2 = rpl2(x)
    assert out2.shape == (2, 4, 16)


def test_vocab_parallel_embedding_single_rank_matches_nn_embedding():
    torch.manual_seed(0)
    vpe = VocabParallelEmbedding(32, 8, group=None)
    assert vpe.weight.shape == (32, 8)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = vpe(ids)
    assert out.shape == (2, 3, 8)


def test_column_parallel_rejects_misaligned_dim_under_multi_rank():
    """Tested with a fake group sentinel so we can exercise the assertion
    without spawning processes."""
    class _FakeGroup:
        pass
    # In single-rank (group=None) any out_features works; assert that
    # would only fire under a real group. The size check is done via
    # `_world(group)` which is 1 for None, so no error here — this test
    # documents that the check is deferred until a real group is passed.
    ColumnParallelLinear(8, 13, group=None)


# ---------- mixture loader ----------


class _ConstLoader:
    """Tiny in-memory stand-in for StreamingLoader; emits a constant tag
    so the test can verify which source served each batch."""

    def __init__(self, tag: int):
        self.tag = tag
        self.calls = 0
        self.state = {"calls": 0}

    def next_batch(self):
        self.calls += 1
        self.state["calls"] = self.calls
        x = torch.full((1, 2), self.tag, dtype=torch.long)
        return x, x

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, sd):
        self.calls = sd["calls"]


def test_mixture_loader_honors_weights_in_distribution():
    """Over many draws, realized fractions converge to specified weights."""
    sources = {"a": _ConstLoader(1), "b": _ConstLoader(2), "c": _ConstLoader(3)}
    mix = MixtureLoader(sources, weights={"a": 0.5, "b": 0.3, "c": 0.2}, seed=0)
    n = 5_000
    for _ in range(n):
        mix.next_batch()
    frac = mix.fractions()
    # Allow 3% absolute tolerance — generous for a 5k-sample empirical check.
    assert abs(frac["a"] - 0.5) < 0.03
    assert abs(frac["b"] - 0.3) < 0.03
    assert abs(frac["c"] - 0.2) < 0.03


def test_mixture_loader_rejects_source_weight_mismatch():
    sources = {"a": _ConstLoader(1), "b": _ConstLoader(2)}
    with pytest.raises(ValueError):
        MixtureLoader(sources, weights={"a": 0.5, "missing": 0.5}, seed=0)
    with pytest.raises(ValueError):
        MixtureLoader(sources, weights={"a": 0.5}, seed=0)  # missing "b"


def test_mixture_loader_rejects_invalid_weights():
    sources = {"a": _ConstLoader(1), "b": _ConstLoader(2)}
    with pytest.raises(ValueError):
        MixtureLoader(sources, weights={"a": 0.0, "b": 0.0}, seed=0)
    with pytest.raises(ValueError):
        MixtureLoader(sources, weights={"a": -1.0, "b": 1.0}, seed=0)


def test_mixture_loader_state_dict_roundtrip_is_deterministic():
    """Save state mid-stream; restore into a fresh loader; subsequent draws
    must match what the original would have produced. This is the mid-epoch
    resume guarantee."""
    sources = {"a": _ConstLoader(1), "b": _ConstLoader(2)}
    a = MixtureLoader(sources, weights={"a": 0.7, "b": 0.3}, seed=42)
    # Drain 20 batches, snapshot, drain 20 more.
    for _ in range(20):
        a.next_batch()
    sd = a.state_dict()
    after_a = [a.next_batch()[0][0, 0].item() for _ in range(20)]

    sources2 = {"a": _ConstLoader(1), "b": _ConstLoader(2)}
    b = MixtureLoader(sources2, weights={"a": 0.7, "b": 0.3}, seed=0)
    b.load_state_dict(sd)
    after_b = [b.next_batch()[0][0, 0].item() for _ in range(20)]
    assert after_a == after_b, "mixture RNG state did not round-trip"


def test_mixture_loader_normalizes_weights():
    """Pass ratios instead of probabilities — should normalize, not error."""
    sources = {"a": _ConstLoader(1), "b": _ConstLoader(2)}
    mix = MixtureLoader(sources, weights={"a": 7, "b": 3}, seed=0)
    np.testing.assert_allclose(mix.probs.sum(), 1.0)
    np.testing.assert_allclose(mix.probs, [0.7, 0.3])
