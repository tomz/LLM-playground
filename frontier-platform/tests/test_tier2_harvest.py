"""Tier 2 harvests from the MAI-Thinking-1 deep dive.

  6. Reasoning-trace archetype rubric (eval/reasoning_rubric.py)
  7. Long-context eval adapters (eval/long_context.py)
  8. Pareto-percentile safety threshold (safety/gates.py)

See docs/research/mai-thinking-1-deep-dive.md §§4, 6, 8.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# =====================================================================
# 6. Reasoning-trace archetype rubric
# =====================================================================

from platform.eval.reasoning_rubric import (  # noqa: E402
    DEFAULT_SIGNALS,
    ReasoningRubric,
    ReasoningSignal,
    RubricResult,
)


# The paper's own weak-vs-strong AIME exemplar (§6 / Appendix C), paraphrased.
_WEAK_TRACE = (
    "The roots are probably 704. Let me just pick that. The answer is 704."
)
_STRONG_TRACE = (
    "Let me derive all four candidate roots from the quartic. We get "
    "x in {240, 704, -3, 12}. Wait, I should re-examine which satisfy the "
    "domain condition x > 0 and the original constraint. By symmetry the "
    "valid invariant rules out 704. Let me verify by substituting back: "
    "plugging 240 into the equation checks out. Is that right? Let me test "
    "a small case to confirm. The answer is 240."
)
_STRONG_AGENTIC = (
    "First let me read the repo and trace through where the bug is defined. "
    "I'll grep for the function. Before patching I'll write a unit test: "
    "def test_fix(): assert solve(3) == 9. Now run the tests — they pass."
)


def test_rubric_strong_beats_weak_overall():
    r = ReasoningRubric()
    weak = r.score(_WEAK_TRACE)
    strong = r.score(_STRONG_TRACE)
    assert isinstance(weak, RubricResult)
    assert strong.overall > weak.overall
    # The named archetypes the strong trace exhibits should each fire.
    assert strong["backtracking"] > 0
    assert strong["verification"] > 0
    assert strong["self_skepticism"] > 0
    assert strong["enumerate_then_filter"] > 0


def test_rubric_weak_trace_scores_near_zero():
    r = ReasoningRubric()
    weak = r.score(_WEAK_TRACE)
    # A pure guess exhibits none of the strong-reasoning signals.
    assert weak.overall < 0.15
    assert weak["backtracking"] == 0.0
    assert weak["verification"] == 0.0


def test_rubric_saturating_density():
    # More instances of a behavior -> higher (but bounded) signal. Use phrases
    # with exactly one backtracking marker each so the count is unambiguous.
    r = ReasoningRubric()
    one = r.score("Hold on.")["backtracking"]
    two = r.score("Hold on. Actually no.")["backtracking"]
    assert 0 < one < two < 1.0
    assert one == pytest.approx(0.5)
    assert two == pytest.approx(0.75)


def test_rubric_math_vs_agentic_split():
    r = ReasoningRubric()
    res = r.score(_STRONG_AGENTIC)
    # Agentic behaviors (tests + evidence) fire on the agentic axis.
    assert res.agentic_score > 0
    assert res["unit_testing"] > 0
    assert res["evidence_first"] > 0


def test_rubric_callable_is_trace_judge():
    r = ReasoningRubric()
    # __call__ returns the overall float (TraceJudge protocol).
    assert r(_STRONG_TRACE) == r.score(_STRONG_TRACE).overall


def test_rubric_rank_and_compare():
    r = ReasoningRubric()
    order = r.rank([_WEAK_TRACE, _STRONG_TRACE])
    assert order == [1, 0]  # strong first
    deltas = r.compare(_WEAK_TRACE, _STRONG_TRACE)
    assert deltas["delta:overall"] > 0
    assert deltas["delta:verification"] > 0


def test_rubric_weights_reweight_overall():
    # Zero-weighting every signal except one isolates it.
    only_backtrack = {s.name: (1.0 if s.name == "backtracking" else 0.0)
                      for s in DEFAULT_SIGNALS}
    r = ReasoningRubric(weights=only_backtrack)
    res = r.score(_STRONG_TRACE)
    assert res.overall == pytest.approx(res["backtracking"])


def test_custom_signal_set():
    sig = ReasoningSignal("has_qed", "ends with QED",
                          lambda t: 1.0 if "QED" in t else 0.0)
    r = ReasoningRubric(signals=(sig,))
    assert r.score("therefore done. QED")["has_qed"] == 1.0
    assert r.score("no marker")["has_qed"] == 0.0


# =====================================================================
# 7. Long-context eval adapters
# =====================================================================

from platform.eval.long_context import (  # noqa: E402
    CodeNLLAdapter,
    LongContextQAAdapter,
    RetrievalNLLAdapter,
    bucketize,
    get_long_context_adapter,
    make_needle_record,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_bucketize_partitions_positions():
    spans = bucketize(10, 4)
    assert spans[0][0] == 0
    assert spans[-1][1] == 10
    # Contiguous, non-overlapping.
    for (lo1, hi1), (lo2, hi2) in zip(spans, spans[1:]):
        assert hi1 == lo2
    # Degenerate inputs.
    assert bucketize(0, 4) == []
    assert bucketize(3, 10) == [(0, 1), (1, 2), (2, 3)]


def test_make_needle_record_plants_needle():
    rec = make_needle_record("r1", filler_words=100, needle="The code is 4271.",
                             query="What is the code?", answer="4271",
                             depth_frac=0.5)
    assert rec["answer"] == "4271"
    assert rec["depth_frac"] == 0.5
    assert "code" in rec["needle"]


def test_long_context_qa_accuracy_by_depth(tmp_path):
    adapter = LongContextQAAdapter()
    recs = [
        make_needle_record("d0", filler_words=20, needle="The code is ALPHA.",
                           query="code?", answer="ALPHA", depth_frac=0.0),
        make_needle_record("d1", filler_words=20, needle="The code is BETA.",
                           query="code?", answer="BETA", depth_frac=0.5),
    ]
    fixture = _write_jsonl(tmp_path / "qa.jsonl", recs)
    examples = list(adapter.load(fixture))
    assert len(examples) == 2

    # An oracle generator that echoes the needle answer back -> perfect accuracy.
    def oracle(prompt: str) -> str:
        if "ALPHA" in prompt:
            return "the code is ALPHA"
        return "the code is BETA"

    out = adapter.score(model=None, examples=examples, generate=oracle)
    assert out["accuracy"] == 1.0
    assert out["n_total"] == 2
    # Per-depth buckets reported.
    assert out["acc_depth000"] == 1.0
    assert out["acc_depth050"] == 1.0


def test_long_context_qa_scores_misses(tmp_path):
    adapter = LongContextQAAdapter()
    recs = [make_needle_record("d0", filler_words=10, needle="X is 9.",
                               query="X?", answer="9", depth_frac=0.3)]
    fixture = _write_jsonl(tmp_path / "qa.jsonl", recs)
    examples = list(adapter.load(fixture))
    out = adapter.score(model=None, examples=examples, generate=lambda p: "I don't know")
    assert out["accuracy"] == 0.0


def test_long_context_adapters_require_path():
    for name in ("code_nll", "retrieval_nll", "long_context_qa"):
        with pytest.raises(FileNotFoundError):
            next(get_long_context_adapter(name).load(None))


def test_get_long_context_adapter_unknown_raises():
    with pytest.raises(KeyError):
        get_long_context_adapter("nope")


def test_evaluator_run_long_context_routes_to_qa(tmp_path):
    """The QA adapter is reachable through Evaluator.run_long_context with a
    generate-only decoding (no model needed)."""
    from platform.eval.harness import Evaluator, EvalRequest

    recs = [make_needle_record("d0", filler_words=10, needle="The code is ZED.",
                               query="code?", answer="ZED", depth_frac=0.4)]
    fixture = _write_jsonl(tmp_path / "qa.jsonl", recs)
    ev = Evaluator()
    req = EvalRequest(
        ckpt="lc-model",
        tasks=["long_context_qa"],
        decoding={"generate": lambda _p: "the code is ZED"},
        benchmarks_2026_paths={"long_context_qa": str(fixture)},
    )
    rep = ev.run_long_context(req)
    assert rep.harness_sha == "long_context"
    assert rep.metrics["long_context_qa:accuracy"] == 1.0
    assert rep.metrics["long_context_qa:n_total"] == 1.0


def test_evaluator_run_long_context_no_model_returns_zeros():
    from platform.eval.harness import Evaluator, EvalRequest

    ev = Evaluator()
    req = EvalRequest(ckpt="x", tasks=["code_nll"])
    rep = ev.run_long_context(req)
    assert rep.metrics == {"code_nll:n_total": 0.0}


def test_code_nll_and_retrieval_nll_with_real_engine(tmp_path):
    """Code-NLL and Retrieval-NLL need a teacher-forced logprob stream; exercise
    them against a tiny in-process TorchEngine (CPU)."""
    import torch
    from platform.model.config import ModelConfig
    from platform.model.transformer import Transformer
    from platform.serving.engine import EngineConfig
    from platform.serving.torch_engine import TorchEngine

    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=512, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=128)
    model = Transformer(cfg)
    engine = TorchEngine(EngineConfig(backend="torch", device="cpu"), model=model)

    # Code-NLL
    code_fix = _write_jsonl(tmp_path / "code.jsonl", [
        {"id": "f1", "text": "def add(a, b):\n    return a + b\n" * 4},
    ])
    code_adapter = CodeNLLAdapter(n_buckets=4)
    code_examples = list(code_adapter.load(code_fix))
    code_out = code_adapter.score(engine, code_examples)
    assert code_out["n_docs"] == 1.0
    assert code_out["nll"] > 0.0          # a real NLL
    assert "nll_first" in code_out and "nll_last" in code_out
    assert "nll_bucket0" in code_out

    # Retrieval-NLL
    ret_fix = _write_jsonl(tmp_path / "ret.jsonl", [
        {"id": "r1", "context": "filler " * 20, "needle": "The token is QZX.",
         "query": "token?", "answer": "QZX", "depth_frac": 0.5},
    ])
    ret_adapter = RetrievalNLLAdapter()
    ret_examples = list(ret_adapter.load(ret_fix))
    ret_out = ret_adapter.score(engine, ret_examples)
    assert ret_out["n_total"] == 1.0
    assert ret_out["answer_nll"] > 0.0
    assert any(k.startswith("answer_nll_depth") for k in ret_out)


# =====================================================================
# 8. Pareto-percentile safety threshold
# =====================================================================

from platform.safety.gates import (  # noqa: E402
    ParetoGateConfig,
    ReleaseMetrics,
    pareto_frontier,
    pareto_preflight,
    percentile_threshold,
    preflight,                 # ensure original still importable/usable
    default_thresholds,
    ModelCard,
)


def test_percentile_threshold_interpolates():
    vals = [0.0, 0.5, 1.0]
    assert percentile_threshold(vals, 0) == 0.0
    assert percentile_threshold(vals, 100) == 1.0
    assert percentile_threshold(vals, 50) == pytest.approx(0.5)
    # Empty -> 0 (nothing to clear).
    assert percentile_threshold([], 50) == 0.0
    # Singleton -> that value.
    assert percentile_threshold([0.7], 25) == 0.7


def test_pareto_frontier_identifies_nondominated():
    pts = [
        (0.9, 0.5, 0.5),   # 0
        (0.5, 0.9, 0.5),   # 1
        (0.4, 0.4, 0.4),   # 2 dominated by 0 and 1
        (0.95, 0.95, 0.95),  # 3 dominates all
    ]
    front = pareto_frontier(pts)
    assert 3 in front
    assert 2 not in front


def test_release_metrics_from_rates():
    m = ReleaseMetrics.from_rates(harmful_rate=0.02, benign_refusal_rate=0.1,
                                  quality=0.8)
    assert m.safety == pytest.approx(0.98)
    assert m.non_over_refusal == pytest.approx(0.9)
    assert m.quality == 0.8


def test_pareto_preflight_passes_strong_candidate():
    reference = [
        ReleaseMetrics(0.90, 0.85, 0.80),
        ReleaseMetrics(0.92, 0.80, 0.82),
        ReleaseMetrics(0.88, 0.88, 0.78),
    ]
    # Candidate at/above the fleet on every axis -> clears percentiles and not
    # dominated.
    candidate = ReleaseMetrics(0.95, 0.90, 0.85)
    res = pareto_preflight(candidate, reference)
    assert res.passed
    assert res.failed_axes == []
    assert res.dominated_by is None


def test_pareto_preflight_blocks_under_percentile():
    reference = [
        ReleaseMetrics(0.90, 0.85, 0.80),
        ReleaseMetrics(0.92, 0.80, 0.82),
        ReleaseMetrics(0.88, 0.88, 0.78),
    ]
    # Very unsafe candidate -> fails the safety percentile.
    candidate = ReleaseMetrics(0.10, 0.95, 0.95)
    res = pareto_preflight(candidate, reference)
    assert not res.passed
    assert "safety" in res.failed_axes


def test_pareto_preflight_blocks_dominated_candidate():
    reference = [ReleaseMetrics(0.95, 0.95, 0.95)]
    cfg = ParetoGateConfig(safety_pct=0.0, non_over_refusal_pct=0.0,
                           quality_pct=0.0)  # percentile bar is trivially low
    # Candidate clears the (zero) percentile but is strictly dominated.
    candidate = ReleaseMetrics(0.90, 0.90, 0.90)
    res = pareto_preflight(candidate, reference, cfg)
    assert not res.passed
    assert res.dominated_by == 0
    assert "dominated" in res.notes


def test_pareto_preflight_empty_reference_is_vacuous_pass():
    res = pareto_preflight(ReleaseMetrics(0.1, 0.1, 0.1), [])
    assert res.passed


def test_original_preflight_still_works(tmp_path):
    # Additive change must not disturb the existing JSON-report gate.
    report = tmp_path / "r.json"
    report.write_text(json.dumps({k: 0.0 for k in default_thresholds()}))
    card = ModelCard("ckpt", "eval", str(report), "chat", [])
    assert preflight(card).passed
