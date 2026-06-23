"""CPU-safe tests for the research harness's pure-Python logic.

The training path (`harness.run_candidate`) is GPU-only by design and is *not*
exercised here — CI is CPU-only. We test everything around it that decides
keep/revert and renders the chart: the gates, ledger I/O + running-best,
candidate introspection, the Pareto frontier, and the scipy-free PCHIP fallback.
These are exactly the parts a broken edit could silently corrupt.
"""
import sys
import pathlib

import numpy as np

RESEARCH = pathlib.Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))

import harness  # noqa: E402
import plot as plotmod  # noqa: E402
import loop as loopmod  # noqa: E402


# ---- gates ---------------------------------------------------------------

def test_gate_rejects_non_finite():
    ok, reason = harness.gate(float("nan"), 1.0, 0.1, {"vocab_size": 65})
    assert not ok and "non-finite" in reason


def test_gate_rejects_no_descent():
    # val_loss at the ~ln(65)=4.17 random ceiling → not learned
    ok, reason = harness.gate(4.17, 4.17, 0.0, {"vocab_size": 65})
    assert not ok and "descend" in reason


def test_gate_rejects_overfit():
    # descended but a huge train↔val gap → overfit guard trips
    ok, reason = harness.gate(2.0, 0.1, 3.0, {"vocab_size": 65, "max_gen_gap": 1.5})
    assert not ok and "overfit" in reason


def test_gate_accepts_good_run():
    ok, reason = harness.gate(2.0, 1.6, 0.4, {"vocab_size": 65, "max_gen_gap": 1.5})
    assert ok and reason == "ok"


# ---- ledger I/O + running best ------------------------------------------

def test_ledger_roundtrip_and_running_best(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.tsv"
    monkeypatch.setattr(loopmod, "LEDGER", ledger)
    loopmod.append_row({"experiment": 1, "val_bpb": 3.0, "status": "keep",
                        "description": "base"})
    loopmod.append_row({"experiment": 2, "val_bpb": 3.5, "status": "discard",
                        "description": "worse"})
    loopmod.append_row({"experiment": 3, "val_bpb": 2.7, "status": "keep",
                        "description": "better"})
    rows = loopmod.read_ledger()
    assert [r["status"] for r in rows] == ["keep", "discard", "keep"]
    # running best only counts kept rows → min(3.0, 2.7)
    assert abs(loopmod.running_best(rows) - 2.7) < 1e-9


def test_candidate_description_parses_without_torch():
    # candidate.py imports nothing heavy at module scope for DESCRIPTION; the
    # loop reads it via AST, so this must work on CPU with no torch import.
    desc = loopmod.candidate_description()
    assert isinstance(desc, str) and len(desc) > 0


# ---- Pareto frontier -----------------------------------------------------

def test_pareto_front_lower_envelope():
    # (throughput, val_bpb): minimise both. (1,3) dominated by (2,2)? No — higher
    # x is better, but pareto_front minimises x; here x is cost so smaller x wins.
    pts = [(1.0, 3.0), (2.0, 2.5), (3.0, 2.5), (2.0, 2.0), (5.0, 1.0)]
    front = plotmod.pareto_front(pts)
    # frontier is sorted by x with strictly-decreasing y
    xs = [p[0] for p in front]
    ys = [p[1] for p in front]
    assert xs == sorted(xs)
    assert all(ys[i] > ys[i + 1] for i in range(len(ys) - 1))
    assert (5.0, 1.0) in front  # the best-quality point is always on the front


# ---- PCHIP fallback (scipy-free) ----------------------------------------

def test_pchip_passes_through_points_and_is_monotone():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([4.0, 3.0, 2.8, 2.5])  # monotone decreasing (a running-best curve)
    xd, yd = plotmod._pchip(x, y, samples=200)
    # endpoints preserved
    assert abs(yd[0] - 4.0) < 1e-6 and abs(yd[-1] - 2.5) < 1e-6
    # monotone non-increasing, no overshoot beyond the data range
    assert yd.max() <= 4.0 + 1e-6 and yd.min() >= 2.5 - 1e-6
    assert all(yd[i] >= yd[i + 1] - 1e-6 for i in range(len(yd) - 1))


def test_pchip_short_input_is_identity():
    x = np.array([1.0])
    y = np.array([2.0])
    xd, yd = plotmod._pchip(x, y)
    assert list(xd) == [1.0] and list(yd) == [2.0]


# ---- require_cuda contract ----------------------------------------------

def test_require_cuda_raises_without_gpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        harness.require_cuda()
    except SystemExit as e:
        assert "GPU-only" in str(e)
        return
    raise AssertionError("require_cuda must SystemExit when CUDA is absent")
