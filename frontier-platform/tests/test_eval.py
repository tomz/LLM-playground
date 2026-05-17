import numpy as np
from platform.eval.harness import Evaluator, EvalRequest
from platform.eval.arena import compute_elo


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
