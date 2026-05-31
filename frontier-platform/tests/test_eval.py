import json
from pathlib import Path

import numpy as np
import pytest

from platform.eval.harness import Evaluator, EvalRequest
from platform.eval.arena import compute_elo
from platform.eval.contamination import ContaminationIndex, contamination_report
from platform.eval.benchmarks_2026 import (
    ARCAGI2Adapter,
    HLEAdapter,
    LiveCodeBenchAdapter,
    MMMUAdapter,
    REGISTRY,
    SWEBenchVerifiedAdapter,
    get_adapter,
)


class DummyModel:
    def __init__(self, vocab=32, seed=0):
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)
    def forward(self, x):
        x = np.asarray(x)
        return self.rng.standard_normal((*x.shape, self.vocab))


def test_evaluator_run_fast_returns_loss():
    ev = Evaluator()
    m = DummyModel()
    x = np.zeros((2, 8), dtype=np.int64)
    y = np.zeros((2, 8), dtype=np.int64)
    rep = ev.run_fast(m, step=0, batch=(x, y))
    assert "loss" in rep.metrics and rep.metrics["loss"] > 0
    assert rep.metrics["perplexity"] > 1


def test_evaluator_run_returns_report():
    ev = Evaluator()
    req = EvalRequest(ckpt="x", tasks=["hellaswag"])
    rep = ev.run(req)
    assert rep.ckpt == "x"
    assert "hellaswag" in rep.metrics


def test_arena_elo_already_works():
    r = compute_elo([("A", "B", 1.0)] * 10)
    assert r["A"] > r["B"]


# ---------- contamination detection ----------

def test_contamination_index_flags_leaked_example():
    idx = ContaminationIndex(n=4, threshold=0.6)
    idx.add_document(
        "the capital of france is paris and the eiffel tower is famous worldwide"
    )
    # Verbatim leak -> high overlap -> contaminated.
    assert idx.is_contaminated("the capital of france is paris and the eiffel tower")
    # Unrelated text -> ~no overlap -> clean.
    assert not idx.is_contaminated("quantum chromodynamics describes the strong force")


def test_contamination_rate_and_report():
    train = [
        "machine learning models are trained on large corpora of text data",
        "gradient descent optimizes the loss function step by step",
    ]
    eval_tasks = {
        "leaky": [
            "machine learning models are trained on large corpora",  # leaked
            "gradient descent optimizes the loss function step",      # leaked
        ],
        "clean": [
            "the mitochondria is the powerhouse of the cell structure",
            "photosynthesis converts sunlight into chemical energy stores",
        ],
    }
    rep = contamination_report(train, eval_tasks, n=4, threshold=0.6)
    assert rep["leaky"] == 1.0
    assert rep["clean"] == 0.0


def test_evaluator_run_includes_contamination_when_requested():
    ev = Evaluator()
    req = EvalRequest(
        ckpt="x", tasks=["hellaswag"],
        train_texts=["the quick brown fox jumps over the lazy dog every morning"],
        contamination_tasks={"t": ["the quick brown fox jumps over the lazy dog"]},
        contamination_n=4, contamination_threshold=0.6,
    )
    rep = ev.run(req)
    assert rep.contamination.get("t") == 1.0
    assert rep.harness_sha in ("lm_eval", "fast_fallback")


# ---------- 2026 frontier benchmark adapters ----------


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_registry_lists_all_2026_adapters():
    assert set(REGISTRY) == {
        "swe_bench_verified", "arc_agi_2", "hle", "mmmu", "live_code_bench",
    }


def test_get_adapter_raises_on_unknown_name():
    with pytest.raises(KeyError) as e:
        get_adapter("not_a_real_benchmark")
    assert "not_a_real_benchmark" in str(e.value)


def test_swe_bench_adapter_scores_patch_match(tmp_path):
    adapter = SWEBenchVerifiedAdapter()
    gold = (
        "diff --git a/x.py b/x.py\n"
        "index abc..def 100644\n--- a/x.py\n+++ b/x.py\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    fixture = _write_jsonl(tmp_path / "swe.jsonl", [
        {"id": "inst-1", "problem_statement": "fix it", "gold_patch": gold,
         "test_patch": "", "repo": "demo/demo"},
        {"id": "inst-2", "problem_statement": "other", "gold_patch": gold},
    ])
    examples = list(adapter.load(fixture))
    assert len(examples) == 2
    # Model "generates" the gold patch for the first and garbage for the second.
    answers = {"inst-1": gold, "inst-2": "no patch"}
    def gen(prompt: str) -> str:
        for ex in examples:
            if ex.prompt == prompt:
                return answers[ex.id]
        return ""
    out = adapter.score(model=None, examples=examples, generate=gen)
    assert out["resolved_rate"] == 0.5
    assert out["n_total"] == 2


def test_swe_bench_adapter_load_requires_path():
    with pytest.raises(FileNotFoundError):
        next(SWEBenchVerifiedAdapter().load(None))


def test_arc_agi2_adapter_extracts_grid_from_response(tmp_path):
    adapter = ARCAGI2Adapter()
    fixture = _write_jsonl(tmp_path / "arc.jsonl", [
        {"id": "p1", "train": [{"input": [[1]], "output": [[2]]}],
         "test_input": [[1]], "test_output": [[2]]},
        {"id": "p2", "train": [], "test_input": [[0]], "test_output": [[9]]},
    ])
    examples = list(adapter.load(fixture))
    def gen(prompt: str) -> str:
        # p1: correct grid embedded in chatter; p2: wrong grid.
        if "p1" in prompt or "[[1]]" in prompt and "p2" not in prompt:
            return "I think the answer is [[2]] yeah"
        return "definitely [[0]]"
    # Use ids to disambiguate
    def gen2(prompt: str) -> str:
        return "answer is [[2]]" if examples[0].prompt == prompt else "answer is [[0]]"
    out = adapter.score(model=None, examples=examples, generate=gen2)
    assert out["exact_match"] == 0.5
    assert out["n_total"] == 2


def test_hle_adapter_handles_mc_and_free_response(tmp_path):
    adapter = HLEAdapter()
    fixture = _write_jsonl(tmp_path / "hle.jsonl", [
        {"id": "mc1", "question": "2+2?", "choices": ["3", "4", "5", "6"],
         "answer": "B", "category": "math"},
        {"id": "fr1", "question": "Capital of France?",
         "answer": "Paris", "answer_aliases": ["paris"], "category": "geo"},
        {"id": "fr2", "question": "Largest planet?", "answer": "Jupiter",
         "category": "astro"},
    ])
    examples = list(adapter.load(fixture))
    responses = {
        "mc1": "B. 4 is correct",
        "fr1": "the answer is paris obviously",
        "fr2": "I think it is Mars",  # wrong
    }
    def gen(prompt: str) -> str:
        for ex in examples:
            if ex.prompt == prompt:
                return responses[ex.id]
        return ""
    out = adapter.score(model=None, examples=examples, generate=gen)
    assert out["n_total"] == 3
    # 2 of 3 correct
    assert abs(out["accuracy"] - 2 / 3) < 1e-9
    # Per-category accuracies reported
    assert out["acc:math"] == 1.0
    assert out["acc:geo"] == 1.0
    assert out["acc:astro"] == 0.0


def test_mmmu_adapter_skips_image_only_examples(tmp_path):
    adapter = MMMUAdapter()
    fixture = _write_jsonl(tmp_path / "mmmu.jsonl", [
        {"id": "t1", "question": "Q1", "choices": ["a", "b"], "answer": "A",
         "modality": "text"},
        {"id": "i1", "question": "Q2", "choices": ["a", "b"], "answer": "A",
         "modality": "image"},
    ])
    examples = list(adapter.load(fixture))
    def gen(_p: str) -> str:
        return "A"
    out = adapter.score(model=None, examples=examples, generate=gen)
    assert out["n_total"] == 1
    assert out["skipped_image_only"] == 1
    assert out["accuracy"] == 1.0


def test_live_code_bench_adapter_runs_sandboxed_tests(tmp_path):
    adapter = LiveCodeBenchAdapter()
    fixture = _write_jsonl(tmp_path / "lcb.jsonl", [
        {"id": "add", "problem": "add(a,b)", "tests": ["assert add(1, 2) == 3"]},
        {"id": "bad", "problem": "mul(a,b)", "tests": ["assert mul(2, 3) == 6"]},
    ])
    examples = list(adapter.load(fixture))
    responses = {
        "add": "```python\ndef add(a, b):\n    return a + b\n```",
        "bad": "```python\ndef mul(a, b):\n    return a - b  # wrong\n```",
    }
    def gen(prompt: str) -> str:
        for ex in examples:
            if ex.prompt == prompt:
                return responses[ex.id]
        return ""
    out = adapter.score(model=None, examples=examples, generate=gen)
    assert out["n_total"] == 2
    assert out["pass_rate"] == 0.5


def test_evaluator_run_2026_routes_through_adapter(tmp_path):
    fixture = _write_jsonl(tmp_path / "hle.jsonl", [
        {"id": "q1", "question": "Pick A.", "choices": ["x", "y"], "answer": "A",
         "category": "demo"},
    ])
    ev = Evaluator()
    req = EvalRequest(
        ckpt="frontier-2026",
        tasks=["hle"],
        decoding={"generate": lambda _p: "A"},
        benchmarks_2026_paths={"hle": str(fixture)},
    )
    rep = ev.run_2026(req)
    assert rep.harness_sha == "benchmarks_2026"
    assert rep.metrics["hle:accuracy"] == 1.0
    assert rep.metrics["hle:n_total"] == 1.0


def test_evaluator_run_merges_2026_and_legacy_metrics(tmp_path):
    fixture = _write_jsonl(tmp_path / "arc.jsonl", [
        {"id": "p1", "train": [], "test_input": [[1]], "test_output": [[2]]},
    ])
    ev = Evaluator()
    req = EvalRequest(
        ckpt="x",
        tasks=["arc_agi_2", "hellaswag"],
        decoding={"generate": lambda _p: "[[2]]"},
        benchmarks_2026_paths={"arc_agi_2": str(fixture)},
    )
    rep = ev.run(req)
    # 2026 adapter ran and produced its namespaced metrics.
    assert rep.metrics["arc_agi_2:exact_match"] == 1.0
    # Legacy hellaswag fast-fallback key still present so the report shape
    # is back-compat for downstream tooling.
    assert "hellaswag" in rep.metrics


def test_evaluator_run_2026_without_model_returns_zeros():
    """No model wired → soft failure: report shape preserved, all zeros."""
    ev = Evaluator()
    req = EvalRequest(ckpt="x", tasks=["hle"])
    rep = ev.run_2026(req)
    assert rep.metrics == {"hle:n_total": 0.0}
