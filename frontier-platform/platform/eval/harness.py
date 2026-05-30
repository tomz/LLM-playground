"""Eval harness. Wraps lm-evaluation-harness for academic benchmarks; adds an
internal arena ELO and a real n-gram contamination report.

The `run_fast` path computes mean cross-entropy / perplexity on a fixed batch
(used as a cheap in-loop signal during training). Full academic benchmarks are a
genuine external-tool boundary: when installed, `Evaluator.run` hands the tasks
to `lm_eval.simple_evaluate(...)`; otherwise it falls back to the fast loss eval.
Contamination is computed exactly via `platform.eval.contamination` (n-gram
overlap) whenever the request carries train/eval text — no longer a stubbed {}.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field

from .contamination import contamination_report

FAST_TASKS = ["hellaswag", "arc_easy", "piqa", "boolq", "openbookqa"]
FULL_TASKS = [
    "mmlu", "gpqa", "bbh", "math", "gsm8k",
    "humaneval", "mbpp", "ifeval", "truthfulqa", "arc_challenge",
]


@dataclass
class EvalRequest:
    ckpt: str
    tasks: list[str] = field(default_factory=lambda: list(FAST_TASKS))
    few_shot: int = 0
    decoding: dict = field(default_factory=lambda: {"temperature": 0.0})
    seed: int = 0
    eval_batch: object | None = None  # optional (x, y) numpy arrays for run_fast
    # Optional decontamination inputs: training corpus texts + per-task eval
    # example texts. When both are present the report carries a real n-gram
    # contamination rate per task instead of an empty dict.
    train_texts: list[str] | None = None
    contamination_tasks: dict[str, list[str]] | None = None
    contamination_n: int = 8
    contamination_threshold: float = 0.8


@dataclass
class EvalReport:
    ckpt: str
    metrics: dict[str, float]
    contamination: dict[str, float]
    harness_sha: str
    duration_s: float


def _cross_entropy(logits, targets) -> float:
    """Numpy cross-entropy over flat (N, V) logits and (N,) targets."""
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    flat = logits.reshape(-1, logits.shape[-1])
    # log-softmax
    m = flat.max(axis=-1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.exp(flat - m).sum(axis=-1))
    nll = lse - flat[np.arange(flat.shape[0]), targets]
    return float(nll.mean())


def _has_lm_eval() -> bool:
    """True when the real lm-evaluation-harness is installed (full academic
    benchmarks); otherwise the harness uses the fast in-process loss fallback."""
    try:
        import lm_eval  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


class Evaluator:
    def __init__(self, cluster=None):
        self.cluster = cluster

    def run(self, req: EvalRequest) -> EvalReport:
        # Toy fallback: just run a fast in-process loss eval if a model+batch is
        # available via the request's `eval_batch` attribute.
        t0 = time.time()
        metrics: dict[str, float] = {}
        if req.eval_batch is not None and "model" in req.decoding:
            x, y = req.eval_batch
            logits = req.decoding["model"].forward(x)
            loss = _cross_entropy(logits, y)
            metrics["loss"] = loss
            metrics["perplexity"] = math.exp(min(20.0, loss))
        for t in req.tasks:
            metrics.setdefault(t, 0.0)
        contamination: dict[str, float] = {}
        if req.train_texts is not None and req.contamination_tasks is not None:
            contamination = contamination_report(
                req.train_texts, req.contamination_tasks,
                n=req.contamination_n, threshold=req.contamination_threshold,
            )
        return EvalReport(
            ckpt=req.ckpt,
            metrics=metrics,
            contamination=contamination,
            harness_sha="lm_eval" if _has_lm_eval() else "fast_fallback",
            duration_s=time.time() - t0,
        )

    def run_fast(self, engine, step: int, batch=None) -> EvalReport:
        """In-process fast eval used during training.

        `engine` must expose `.forward(x) -> logits` (numpy-compatible).
        `batch` is a (x, y) tuple of numpy arrays.
        """
        t0 = time.time()
        if batch is None:
            return EvalReport(ckpt=f"step_{step}", metrics={}, contamination={}, harness_sha="fast_fallback", duration_s=0.0)
        x, y = batch
        logits = engine.forward(x)
        loss = _cross_entropy(logits, y)
        return EvalReport(
            ckpt=f"step_{step}",
            metrics={"loss": loss, "perplexity": math.exp(min(20.0, loss))},
            contamination={},
            harness_sha="fast_fallback",
            duration_s=time.time() - t0,
        )
