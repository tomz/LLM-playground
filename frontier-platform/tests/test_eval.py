import numpy as np
from platform.eval.harness import Evaluator, EvalRequest
from platform.eval.arena import compute_elo
from platform.eval.contamination import ContaminationIndex, contamination_report


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
