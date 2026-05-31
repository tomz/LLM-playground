"""Eval harness. Wraps lm-evaluation-harness for academic benchmarks; adds an
internal arena ELO and a real n-gram contamination report.

Three eval modes coexist:

* **fast** (``Evaluator.run_fast``): in-process cross-entropy / perplexity on a
  single batch. Used as a cheap signal *during* training.
* **lm-eval-harness** (``Evaluator.run``): when ``lm-evaluation-harness`` is
  installed, the harness's standard tasks (HellaSwag, ARC, PIQA, GSM8K, MMLU,
  HumanEval, ...) are run via ``lm_eval.simple_evaluate``. The model is
  wrapped via :func:`build_lm_eval_model` so any object exposing a TorchEngine-
  style ``.generate(req)`` works without a separate adapter.
* **2026 frontier adapters** (``Evaluator.run_2026``): runs the SWE-bench /
  ARC-AGI-2 / HLE / MMMU / LiveCodeBench adapters from
  :mod:`platform.eval.benchmarks_2026`. Each adapter ships a deterministic
  CI-friendly scorer + the production fixture format.

Contamination is computed exactly via :mod:`platform.eval.contamination` (n-gram
overlap) whenever the request carries train/eval text — no longer a stubbed
``{}``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable

from .benchmarks_2026 import REGISTRY as _2026_REGISTRY, get_adapter
from .contamination import contamination_report

FAST_TASKS = ["hellaswag", "arc_easy", "piqa", "boolq", "openbookqa"]
FULL_TASKS = [
    "mmlu", "gpqa", "bbh", "math", "gsm8k",
    "humaneval", "mbpp", "ifeval", "truthfulqa", "arc_challenge",
]
# Names that route to platform.eval.benchmarks_2026 instead of lm-eval-harness.
TASKS_2026 = sorted(_2026_REGISTRY.keys())


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
    # 2026 frontier adapters: per-task local JSONL fixtures keyed by task name.
    benchmarks_2026_paths: dict[str, str] | None = None
    # Per-task example caps so a smoke run doesn't process the whole set.
    max_examples_per_task: int | None = None


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


# ---------------------------------------------------------------------------
# lm-evaluation-harness model adapter
# ---------------------------------------------------------------------------


def build_lm_eval_model(engine, tokenizer=None):
    """Wrap a :class:`platform.serving.engine.Engine` for ``lm_eval``.

    ``lm-evaluation-harness`` accepts a ``LM`` subclass; we provide the
    minimum surface (``loglikelihood``, ``loglikelihood_rolling``,
    ``generate_until``) by replaying the engine's logprob stream. Lazy-imports
    ``lm_eval`` so this function is only safe to call when it's installed.
    """
    from lm_eval.api.model import LM  # type: ignore

    class _EngineLM(LM):
        def __init__(self):
            super().__init__()
            self._engine = engine
            self._tok = tokenizer

        # The harness uses these three entry points; everything else is built
        # on top of them.
        def loglikelihood(self, requests):  # noqa: D401
            return [(self._score(prompt, target), True)
                    for prompt, target in [(r.args[0], r.args[1]) for r in requests]]

        def loglikelihood_rolling(self, requests):
            return [self._score(r.args[0], "") for r in requests]

        def generate_until(self, requests):
            outs = []
            for r in requests:
                prompt, until = r.args[0], r.args[1].get("until", []) if len(r.args) > 1 else []
                text = self._generate(prompt, until=until)
                outs.append(text)
            return outs

        # ---- helpers -----------------------------------------------------

        def _score(self, prompt: str, target: str) -> float:
            ids = self._encode(prompt + target)
            return float(sum(self._gen_logprobs(ids)))

        def _generate(self, prompt: str, until: list[str]) -> str:
            from platform.serving.engine import GenRequest
            import asyncio
            req = GenRequest(prompt_ids=self._encode(prompt), max_new_tokens=128,
                             temperature=0.0)
            async def _drain() -> str:
                out_ids = []
                async for chunk in self._engine.generate(req):
                    if not chunk.get("done"):
                        out_ids.append(int(chunk["token_id"]))
                return self._decode(out_ids)
            text = asyncio.run(_drain())
            for sep in until or []:
                if sep in text:
                    text = text.split(sep, 1)[0]
            return text

        def _encode(self, s: str) -> list[int]:
            if self._tok is not None and hasattr(self._tok, "encode"):
                return list(self._tok.encode(s))
            return list(s.encode("utf-8"))

        def _decode(self, ids: list[int]) -> str:
            if self._tok is not None and hasattr(self._tok, "decode"):
                try:
                    return self._tok.decode(ids)
                except Exception:
                    pass
            return bytes(i & 0xFF for i in ids).decode("utf-8", errors="replace")

        def _gen_logprobs(self, ids: list[int]) -> list[float]:
            """One-shot generation that returns per-token logprobs."""
            from platform.serving.engine import GenRequest
            import asyncio
            req = GenRequest(prompt_ids=ids, max_new_tokens=0, temperature=0.0)
            async def _drain() -> list[float]:
                lps = []
                async for chunk in self._engine.generate(req):
                    if not chunk.get("done") and "logprob" in chunk:
                        lps.append(float(chunk["logprob"]))
                return lps
            return asyncio.run(_drain())

    return _EngineLM()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    def __init__(self, cluster=None):
        self.cluster = cluster

    def run(self, req: EvalRequest) -> EvalReport:
        """Score ``req.tasks`` via lm-eval-harness when available; otherwise
        the existing fast fallback. Tasks named in :data:`TASKS_2026` are
        always routed to the 2026 adapters in :func:`run_2026`.
        """
        t0 = time.time()
        metrics: dict[str, float] = {}

        # 1. 2026 adapters
        adapter_tasks = [t for t in req.tasks if t in _2026_REGISTRY]
        if adapter_tasks:
            sub = self._run_2026_internal(req, only=adapter_tasks)
            metrics.update(sub)

        # 2. lm-evaluation-harness for the rest
        other_tasks = [t for t in req.tasks if t not in _2026_REGISTRY]
        if other_tasks:
            if _has_lm_eval() and "engine" in req.decoding:
                from lm_eval import simple_evaluate  # type: ignore
                lm = build_lm_eval_model(
                    req.decoding["engine"], req.decoding.get("tokenizer"),
                )
                lim = req.max_examples_per_task
                out = simple_evaluate(model=lm, tasks=other_tasks, num_fewshot=req.few_shot,
                                      limit=lim)
                # simple_evaluate returns {'results': {task: {metric: value}}}
                for task, m in (out.get("results") or {}).items():
                    for k, v in m.items():
                        try:
                            metrics[f"{task}:{k}"] = float(v)
                        except (TypeError, ValueError):
                            continue
            else:
                # Fast fallback so the report shape is still populated.
                for t in other_tasks:
                    metrics.setdefault(t, 0.0)

        if req.eval_batch is not None and "model" in req.decoding:
            x, y = req.eval_batch
            logits = req.decoding["model"].forward(x)
            loss = _cross_entropy(logits, y)
            metrics["loss"] = loss
            metrics["perplexity"] = math.exp(min(20.0, loss))

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

    def run_2026(self, req: EvalRequest) -> EvalReport:
        """Run only the 2026 frontier adapters (SWE-bench/ARC-AGI-2/HLE/...)."""
        t0 = time.time()
        metrics = self._run_2026_internal(req, only=req.tasks)
        return EvalReport(
            ckpt=req.ckpt,
            metrics=metrics,
            contamination={},
            harness_sha="benchmarks_2026",
            duration_s=time.time() - t0,
        )

    # ---- internals --------------------------------------------------------

    def _run_2026_internal(self, req: EvalRequest, *, only: Iterable[str]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        paths = req.benchmarks_2026_paths or {}
        model = req.decoding.get("model") or req.decoding.get("engine")
        generate = req.decoding.get("generate")
        if model is None and generate is None:
            # No model wired — fail soft so an eval-only call still returns a
            # populated report shape (every metric is 0.0).
            for t in only:
                metrics[f"{t}:n_total"] = 0.0
            return metrics
        for task in only:
            if task not in _2026_REGISTRY:
                continue
            adapter = get_adapter(task)
            path = paths.get(task)
            if path is None:
                metrics[f"{task}:n_total"] = 0.0
                continue
            examples = list(adapter.load(path))
            if req.max_examples_per_task is not None:
                examples = examples[: req.max_examples_per_task]
            sub = adapter.score(model, examples, generate=generate)
            for k, v in sub.items():
                metrics[f"{task}:{k}"] = float(v)
        return metrics

    def run_fast(self, engine, step: int, batch=None) -> EvalReport:
        """In-process fast eval used during training.

        ``engine`` must expose ``.forward(x) -> logits`` (numpy-compatible).
        ``batch`` is a (x, y) tuple of numpy arrays.
        """
        t0 = time.time()
        if batch is None:
            return EvalReport(ckpt=f"step_{step}", metrics={}, contamination={},
                              harness_sha="fast_fallback", duration_s=0.0)
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
