"""Long-context evaluation adapters (harvest of MAI-Thinking-1 Appendix B).

The paper extends context to 256K via a cheap staged recipe (pre-train 16K →
mid-train 64K → short 256K extension) and measures it with three families
(docs/research/mai-thinking-1-deep-dive.md §4):

* **Code-NLL** — negative log-likelihood on long code files. A clean, label-free
  signal that long-range modelling improved: NLL on tokens *deep* into a long
  document should approach NLL on tokens near the start once the model uses the
  context. We report NLL bucketed by position so you can see the curve flatten.
* **Retrieval-NLL** — NLL on a planted answer span that can only be predicted by
  attending to a fact placed earlier in the context (the LM-loss analogue of
  needle-in-a-haystack). Low NLL on the needle ⇒ the model retrieved it.
* **Answer-accuracy-by-position** — generative QA where the supporting fact is
  placed at a controlled depth; accuracy as a function of needle depth is the
  "lost in the middle" curve.

These adapters follow the :class:`platform.eval.benchmarks_2026.BenchmarkAdapter`
shape (``name`` / ``load`` / ``score``) and run on CPU. ``Code-NLL`` and
``Retrieval-NLL`` need teacher-forced per-token logprobs, computed with a single
forward pass via :func:`platform.alignment._common.compute_token_logps` (the same
numerics the GRPO/SFT path uses) — they accept either a bare ``Transformer`` or a
``TorchEngine`` wrapping one. The accuracy adapter only needs ``str -> str``
generation and uses the shared engine-backed generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .benchmarks_2026 import Example, _default_generate, _iter_jsonl


# ---------------------------------------------------------------------------
# Position bucketing
# ---------------------------------------------------------------------------


def bucketize(n_positions: int, n_buckets: int) -> list[tuple[int, int]]:
    """Split ``range(n_positions)`` into ``n_buckets`` contiguous [lo, hi) spans.

    Used to report NLL as a function of *depth into the document* so the
    long-range curve (NLL high→low as the model learns to use context) is
    visible rather than averaged away."""
    if n_positions <= 0 or n_buckets <= 0:
        return []
    n_buckets = min(n_buckets, n_positions)
    step = n_positions / n_buckets
    spans = []
    for b in range(n_buckets):
        lo = int(round(b * step))
        hi = int(round((b + 1) * step))
        if hi > lo:
            spans.append((lo, hi))
    return spans


# ---------------------------------------------------------------------------
# Teacher-forced per-token logprob (single forward pass over the model)
# ---------------------------------------------------------------------------


def _resolve_model(model):
    """Return a raw torch model exposing ``__call__ -> (logits, _)``.

    Accepts either a bare :class:`platform.model.transformer.Transformer` or a
    serving :class:`~platform.serving.torch_engine.TorchEngine` wrapping one
    (unwrapped via ``.model``)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "forward"):
        return inner
    return model


def _token_logprobs(model, token_ids: list[int]) -> list[float]:
    """Teacher-forced per-token logprobs for ``token_ids`` under ``model``.

    Runs a single forward pass and gathers ``log P(token_ids[t] | token_ids[:t])``
    for ``t = 1..len-1`` (the first position has no prediction), so the returned
    list has ``len(token_ids) - 1`` entries aligned to the *predicted* tokens.
    This is the standard NLL/perplexity computation — not the serving engine's
    sampling stream — so it works for Code-NLL / Retrieval-NLL on CPU without
    decoding. Reuses :func:`platform.alignment._common.compute_token_logps` so the
    numerics match the GRPO/SFT logprob path exactly.
    """
    import torch

    from platform.alignment._common import compute_token_logps

    m = _resolve_model(model)
    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    x, y = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        lp = compute_token_logps(m, x, y)  # [1, T-1]
    return lp[0].tolist()


def _encode(text: str, tokenizer=None) -> list[int]:
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text))
    return list(text.encode("utf-8"))


def _nll_by_bucket(logprobs: list[float], n_buckets: int) -> dict[str, float]:
    """Aggregate a per-token logprob list into mean NLL per position bucket
    plus an overall mean. Keys: ``nll`` (overall), ``nll_bucket{i}``,
    ``nll_first``/``nll_last`` shorthands."""
    nlls = [-lp for lp in logprobs]
    if not nlls:
        return {"nll": 0.0, "n_tokens": 0.0}
    out: dict[str, float] = {"nll": sum(nlls) / len(nlls), "n_tokens": float(len(nlls))}
    spans = bucketize(len(nlls), n_buckets)
    for i, (lo, hi) in enumerate(spans):
        chunk = nlls[lo:hi]
        out[f"nll_bucket{i}"] = sum(chunk) / len(chunk)
    if spans:
        out["nll_first"] = out["nll_bucket0"]
        out["nll_last"] = out[f"nll_bucket{len(spans) - 1}"]
    return out


# ---------------------------------------------------------------------------
# Code-NLL
# ---------------------------------------------------------------------------


@dataclass
class CodeNLLAdapter:
    """Mean per-token NLL on long code documents, bucketed by position.

    Fixture JSONL: ``{"id", "text"}`` where ``text`` is a long code file. The
    score is averaged over documents; ``nll_first``/``nll_last`` expose the
    near-start vs. deep-in-document NLL so the long-range improvement is legible
    (a long-context model drives ``nll_last`` toward ``nll_first``).
    """

    name: str = "code_nll"
    n_buckets: int = 8

    def load(self, path: str | Path | None = None) -> Iterator[Example]:
        if path is None:
            raise FileNotFoundError(
                "Code-NLL needs a JSONL of long code files; pass `path=` "
                "(records: id, text)."
            )
        for rec in _iter_jsonl(path):
            yield Example(id=rec["id"], prompt=rec["text"], answer="",
                          meta={"chars": len(rec["text"])})

    def score(self, model, examples: Iterable[Example], *,
              generate: Callable[[str], str] | None = None,
              tokenizer=None) -> dict[str, float]:
        agg: dict[str, list[float]] = {}
        n_docs = 0
        for ex in examples:
            ids = _encode(ex.prompt, tokenizer)
            if len(ids) < 2:
                continue
            n_docs += 1
            lps = _token_logprobs(model, ids)
            sub = _nll_by_bucket(lps, self.n_buckets)
            for k, v in sub.items():
                agg.setdefault(k, []).append(v)
        out = {k: (sum(v) / len(v)) for k, v in agg.items() if v}
        out["n_docs"] = float(n_docs)
        return out


# ---------------------------------------------------------------------------
# Retrieval-NLL (needle, scored by LM loss on the planted answer)
# ---------------------------------------------------------------------------


@dataclass
class RetrievalNLLAdapter:
    """NLL on a planted answer span that requires attending to an earlier fact.

    Fixture JSONL per record: ``context`` (long filler), ``needle`` (a sentence
    stating the fact, inserted at ``depth_frac`` into the context), ``query`` and
    ``answer`` (the span whose NLL we measure). Low ``answer_nll`` ⇒ the model
    used the planted fact. ``answer_nll_by_depth`` lets a caller chart the
    lost-in-the-middle curve when records vary ``depth_frac``.
    """

    name: str = "retrieval_nll"

    def load(self, path: str | Path | None = None) -> Iterator[Example]:
        if path is None:
            raise FileNotFoundError(
                "Retrieval-NLL needs a JSONL; pass `path=` (records: id, "
                "context, needle, query, answer, depth_frac)."
            )
        for rec in _iter_jsonl(path):
            yield Example(
                id=rec["id"],
                prompt=_assemble_needle_prompt(rec),
                answer=str(rec["answer"]),
                meta={"depth_frac": float(rec.get("depth_frac", 0.5))},
            )

    def score(self, model, examples: Iterable[Example], *,
              generate: Callable[[str], str] | None = None,
              tokenizer=None) -> dict[str, float]:
        nlls: list[float] = []
        by_depth: dict[str, list[float]] = {}
        n = 0
        for ex in examples:
            full_ids = _encode(ex.prompt + ex.answer, tokenizer)
            ans_ids = _encode(ex.answer, tokenizer)
            if len(ans_ids) < 1 or len(full_ids) <= len(ans_ids):
                continue
            n += 1
            lps = _token_logprobs(model, full_ids)
            # The answer occupies the final len(ans_ids) prediction positions.
            ans_lps = lps[-len(ans_ids):] if len(lps) >= len(ans_ids) else lps
            nll = (-sum(ans_lps) / len(ans_lps)) if ans_lps else 0.0
            nlls.append(nll)
            bucket = f"{round(ex.meta['depth_frac'] * 100):03d}"
            by_depth.setdefault(bucket, []).append(nll)
        out: dict[str, float] = {
            "answer_nll": (sum(nlls) / len(nlls)) if nlls else 0.0,
            "n_total": float(n),
        }
        for depth, vals in sorted(by_depth.items()):
            out[f"answer_nll_depth{depth}"] = sum(vals) / len(vals)
        return out


# ---------------------------------------------------------------------------
# Answer-accuracy-by-position (generative needle-in-haystack)
# ---------------------------------------------------------------------------


@dataclass
class LongContextQAAdapter:
    """Generative QA with the supporting fact planted at a controlled depth.

    Fixture JSONL per record: ``context``, ``needle``, ``query``, ``answer``,
    ``depth_frac``. We assemble ``context`` with ``needle`` inserted at
    ``depth_frac``, ask ``query``, and check the generation contains ``answer``
    (case-insensitive). Accuracy is reported overall and **bucketed by depth** so
    the lost-in-the-middle dip is measurable.
    """

    name: str = "long_context_qa"

    def load(self, path: str | Path | None = None) -> Iterator[Example]:
        if path is None:
            raise FileNotFoundError(
                "Long-context QA needs a JSONL; pass `path=` (records: id, "
                "context, needle, query, answer, depth_frac)."
            )
        for rec in _iter_jsonl(path):
            yield Example(
                id=rec["id"],
                prompt=_assemble_needle_prompt(rec),
                answer=str(rec["answer"]),
                meta={"depth_frac": float(rec.get("depth_frac", 0.5))},
            )

    def score(self, model, examples: Iterable[Example], *,
              generate: Callable[[str], str] | None = None,
              tokenizer=None) -> dict[str, float]:
        gen = generate or _default_generate(model)
        n = 0
        n_pass = 0
        by_depth: dict[str, list[int]] = {}
        for ex in examples:
            n += 1
            resp = gen(ex.prompt) or ""
            ok = int(ex.answer.lower() in resp.lower())
            n_pass += ok
            bucket = f"{round(ex.meta['depth_frac'] * 100):03d}"
            by_depth.setdefault(bucket, []).append(ok)
        out: dict[str, float] = {
            "accuracy": (n_pass / n) if n else 0.0,
            "n_total": float(n),
        }
        for depth, hits in sorted(by_depth.items()):
            out[f"acc_depth{depth}"] = sum(hits) / len(hits)
        return out


# ---------------------------------------------------------------------------
# Shared needle assembly
# ---------------------------------------------------------------------------


def _assemble_needle_prompt(rec: dict) -> str:
    """Insert ``needle`` into ``context`` at ``depth_frac`` and append the query.

    If the record already provides a fully-assembled ``prompt`` we honour it;
    otherwise we build a haystack so fixtures can be specified compactly."""
    if rec.get("prompt"):
        return rec["prompt"]
    context = rec.get("context", "")
    needle = rec.get("needle", "")
    query = rec.get("query", "")
    depth = float(rec.get("depth_frac", 0.5))
    if needle:
        words = context.split()
        at = max(0, min(len(words), int(round(len(words) * depth))))
        words[at:at] = [needle]
        context = " ".join(words)
    return f"{context}\n\nQuestion: {query}\nAnswer:"


def make_needle_record(rid: str, *, filler_words: int, needle: str, query: str,
                       answer: str, depth_frac: float, filler_token: str = "the") -> dict:
    """Build a synthetic needle-in-haystack record (handy for tests + smoke runs).

    Produces a haystack of ``filler_words`` repeated tokens with ``needle``
    planted at ``depth_frac``; the model must surface ``answer`` for ``query``."""
    context = " ".join([filler_token] * filler_words)
    return {
        "id": rid,
        "context": context,
        "needle": needle,
        "query": query,
        "answer": answer,
        "depth_frac": depth_frac,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


LONG_CONTEXT_REGISTRY: dict[str, type] = {
    "code_nll": CodeNLLAdapter,
    "retrieval_nll": RetrievalNLLAdapter,
    "long_context_qa": LongContextQAAdapter,
}


def get_long_context_adapter(name: str):
    cls = LONG_CONTEXT_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"unknown long-context adapter: {name!r}; "
            f"known: {sorted(LONG_CONTEXT_REGISTRY)}"
        )
    return cls()


__all__ = [
    "CodeNLLAdapter",
    "RetrievalNLLAdapter",
    "LongContextQAAdapter",
    "bucketize",
    "make_needle_record",
    "LONG_CONTEXT_REGISTRY",
    "get_long_context_adapter",
]
