"""Adapters for 2026-class frontier benchmarks.

The simulator (``platform.sim.scaling``) ships closed-form sigmoid predictors
for the 2026 reasoning / coding / agentic benchmarks (SWE-bench-Verified,
ARC-AGI-2, HLE, MMMU, LiveCodeBench). This module turns those predictors into
**adapters that can be wired into the real :class:`Evaluator`** so that, when
a model is actually trainable, the same code path that produced the predicted
numbers runs against real eval data.

Every adapter implements the same minimal :class:`BenchmarkAdapter` protocol:

* ``name`` — task name as the harness reports it
* ``load(...)`` — return an iterable of :class:`Example`s. Reads a local JSONL
  fixture by default; the production path overrides this with HF Datasets /
  the upstream loader.
* ``score(model, examples)`` — score the model on the examples and return a
  ``{metric_name: float}`` dict that drops straight into ``EvalReport.metrics``.

For the dependency-free path, ``score`` operates via the
:class:`platform.serving.engine.Engine` (which works against the in-process
TorchEngine without a GPU), so the whole adapter is unit-testable in CI.

These adapters are intentionally faithful to the *shape* of the upstream
benchmarks (their I/O format, their scoring function) but they are not
substitutes for the real upstream loaders. See ``platform.eval.harness`` for
how an installed ``lm-evaluation-harness`` is preferred when present.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol


@dataclass(frozen=True)
class Example:
    """One benchmark example. ``meta`` carries adapter-specific fields."""

    id: str
    prompt: str
    answer: str = ""
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------


class BenchmarkAdapter(Protocol):
    name: str

    def load(self, path: str | Path | None = None) -> Iterator[Example]: ...
    def score(self, model, examples: Iterable[Example],
              *, generate: Callable[[str], str] | None = None) -> dict[str, float]: ...


def _iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------------------------------------------------------
# SWE-bench Verified
# ----------------------------------------------------------------------------


@dataclass
class SWEBenchVerifiedAdapter:
    """Score: fraction of instances whose generated patch passes the upstream
    test suite. This adapter uses a sandboxed-execution scorer (the real
    pipeline plugs in :func:`platform.rl.jail.run_in_jailed_sandbox`); the
    in-CI scorer is a deterministic patch-equality check against a known-good
    patch shipped in the fixture."""

    name: str = "swe_bench_verified"

    def load(self, path: str | Path | None = None) -> Iterator[Example]:
        if path is None:
            raise FileNotFoundError(
                "SWE-bench Verified ships via HF Datasets; pass `path=` to a "
                "JSONL with one record per instance (id, problem_statement, "
                "test_patch, gold_patch)."
            )
        for rec in _iter_jsonl(path):
            yield Example(
                id=rec["id"],
                prompt=rec["problem_statement"],
                answer=rec.get("gold_patch", ""),
                meta={"test_patch": rec.get("test_patch", ""),
                       "repo": rec.get("repo", "")},
            )

    def score(self, model, examples, *, generate=None) -> dict[str, float]:
        gen = generate or _default_generate(model)
        n_total = 0
        n_pass = 0
        for ex in examples:
            n_total += 1
            patch = gen(ex.prompt)
            if _patch_equivalent(patch, ex.answer):
                n_pass += 1
        return {"resolved_rate": (n_pass / n_total) if n_total else 0.0,
                "n_total": float(n_total)}


def _patch_equivalent(a: str, b: str) -> bool:
    """Normalise + compare two unified-diff patches.

    Strips diff headers (``index``, ``---``, ``+++``, hunk ranges) so a patch
    generated against a slightly different parent hash still matches. The real
    SWE-bench scorer *applies* the patch and runs the test suite; this is a
    deterministic stand-in for CI."""
    def _norm(s: str) -> list[str]:
        out = []
        for line in s.splitlines():
            if not line.strip():
                continue
            if line.startswith(("diff --git", "index ", "---", "+++", "@@")):
                continue
            out.append(line.rstrip())
        return out
    return _norm(a) == _norm(b)


# ----------------------------------------------------------------------------
# ARC-AGI-2
# ----------------------------------------------------------------------------


@dataclass
class ARCAGI2Adapter:
    """ARC-AGI-2: grid → grid puzzles. Score is exact-match per task.

    Examples carry a list of ``train`` (input/output) pairs and a single
    ``test_input``; the model must emit a grid in the canonical JSON
    representation that equals ``test_output``."""

    name: str = "arc_agi_2"

    def load(self, path: str | Path | None = None) -> Iterator[Example]:
        if path is None:
            raise FileNotFoundError(
                "ARC-AGI-2 ships as per-task JSON files; pass `path=` to a "
                "JSONL aggregating them (id, train, test_input, test_output)."
            )
        for rec in _iter_jsonl(path):
            yield Example(
                id=rec["id"],
                prompt=_format_arc_prompt(rec),
                answer=json.dumps(rec["test_output"], separators=(",", ":")),
                meta={"train": rec.get("train", []),
                       "test_input": rec["test_input"],
                       "test_output": rec["test_output"]},
            )

    def score(self, model, examples, *, generate=None) -> dict[str, float]:
        gen = generate or _default_generate(model)
        n_total = 0
        n_pass = 0
        for ex in examples:
            n_total += 1
            raw = gen(ex.prompt)
            if _arc_grid_equal(raw, ex.meta["test_output"]):
                n_pass += 1
        return {"exact_match": (n_pass / n_total) if n_total else 0.0,
                "n_total": float(n_total)}


def _format_arc_prompt(rec: dict) -> str:
    train_block = "\n".join(
        f"input={json.dumps(t['input'])}\noutput={json.dumps(t['output'])}"
        for t in rec.get("train", [])
    )
    return (
        "ARC puzzle: induce the rule from the training pairs and apply it to the test input.\n"
        f"{train_block}\n"
        f"test_input={json.dumps(rec['test_input'])}\n"
        "Reply with ONLY a JSON 2-D array for test_output."
    )


def _arc_grid_equal(generated_text: str, expected_grid: list[list[int]]) -> bool:
    """Pull the first JSON array out of ``generated_text`` and compare."""
    m = re.search(r"\[\s*\[.*?\]\s*\]", generated_text, flags=re.DOTALL)
    if not m:
        return False
    try:
        got = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    return got == expected_grid


# ----------------------------------------------------------------------------
# HLE (Humanity's Last Exam)
# ----------------------------------------------------------------------------


@dataclass
class HLEAdapter:
    """HLE: ~3,000 graduate-level questions across STEM + humanities.

    Mix of multiple-choice and free-response. We score multiple-choice by
    exact letter match (A/B/C/D) and free-response by case-insensitive
    string-contains against a list of acceptable answers, mirroring the
    upstream grader's tolerance.
    """

    name: str = "hle"

    def load(self, path: str | Path | None = None) -> Iterator[Example]:
        if path is None:
            raise FileNotFoundError(
                "HLE ships via HF Datasets; pass `path=` to a JSONL "
                "(id, question, choices?, answer, answer_aliases?)."
            )
        for rec in _iter_jsonl(path):
            yield Example(
                id=rec["id"],
                prompt=_format_hle_prompt(rec),
                answer=str(rec["answer"]),
                meta={
                    "choices": rec.get("choices"),
                    "answer_aliases": rec.get("answer_aliases", []),
                    "category": rec.get("category", ""),
                },
            )

    def score(self, model, examples, *, generate=None) -> dict[str, float]:
        gen = generate or _default_generate(model)
        n_total = 0
        n_pass = 0
        per_cat: dict[str, list[int]] = {}
        for ex in examples:
            n_total += 1
            response = gen(ex.prompt)
            passed = _hle_match(response, ex)
            n_pass += int(passed)
            cat = ex.meta.get("category", "_uncat")
            per_cat.setdefault(cat, []).append(int(passed))
        out = {"accuracy": (n_pass / n_total) if n_total else 0.0,
               "n_total": float(n_total)}
        for cat, hits in per_cat.items():
            out[f"acc:{cat}"] = sum(hits) / len(hits)
        return out


def _format_hle_prompt(rec: dict) -> str:
    if rec.get("choices"):
        choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(rec["choices"]))
        return (
            f"{rec['question']}\n\n{choices}\n\n"
            "Reply with the single letter of the correct answer."
        )
    return f"{rec['question']}\n\nProvide your final answer on the last line."


def _hle_match(response: str, ex: Example) -> bool:
    answer = ex.answer.strip()
    response = (response or "").strip()
    if ex.meta.get("choices"):
        # MC: extract the first A-Z that appears at the start of any line.
        for line in response.splitlines():
            line = line.strip().lstrip("(").rstrip(").").strip()
            if line and line[0].upper() in "ABCDEFGHIJ":
                return line[0].upper() == answer.upper()
        return False
    # Free response: case-insensitive contains, accept any alias.
    candidates = [answer, *ex.meta.get("answer_aliases", [])]
    low = response.lower()
    return any(c.lower() in low for c in candidates if c)


# ----------------------------------------------------------------------------
# MMMU (Massive Multi-discipline Multimodal Understanding)
# ----------------------------------------------------------------------------


@dataclass
class MMMUAdapter:
    """MMMU is multimodal; in this text-only harness we score the language
    half (questions whose modality is text-only). Scoring is MC exact-match.

    The adapter records ``skipped_image_only`` so reports can flag the gap
    when a text-only model is run against the full benchmark."""

    name: str = "mmmu"

    def load(self, path: str | Path | None = None) -> Iterator[Example]:
        if path is None:
            raise FileNotFoundError(
                "MMMU ships via HF Datasets; pass `path=` to a JSONL "
                "(id, question, choices, answer, modality)."
            )
        for rec in _iter_jsonl(path):
            yield Example(
                id=rec["id"],
                prompt=_format_hle_prompt({  # MC formatter is shared
                    "question": rec["question"], "choices": rec["choices"],
                }),
                answer=str(rec["answer"]),
                meta={"choices": rec["choices"],
                       "modality": rec.get("modality", "text"),
                       "subject": rec.get("subject", "")},
            )

    def score(self, model, examples, *, generate=None) -> dict[str, float]:
        gen = generate or _default_generate(model)
        n_total = 0
        n_pass = 0
        n_skipped = 0
        for ex in examples:
            if ex.meta.get("modality") != "text":
                n_skipped += 1
                continue
            n_total += 1
            response = gen(ex.prompt)
            if _hle_match(response, ex):
                n_pass += 1
        return {
            "accuracy": (n_pass / n_total) if n_total else 0.0,
            "n_total": float(n_total),
            "skipped_image_only": float(n_skipped),
        }


# ----------------------------------------------------------------------------
# LiveCodeBench
# ----------------------------------------------------------------------------


@dataclass
class LiveCodeBenchAdapter:
    """Generate Python that passes the problem's unit tests in a sandbox.

    The fixture format is JSONL with ``id``, ``problem``, ``tests``
    (a list of ``assert ...`` strings), and an optional ``starter_code``.
    We reuse :class:`platform.rl.verifiers.CodeUnitTestVerifier` so the
    sandbox + jail story is consistent with the RLVR code-execution path.
    """

    name: str = "live_code_bench"

    def load(self, path: str | Path | None = None) -> Iterator[Example]:
        if path is None:
            raise FileNotFoundError(
                "LiveCodeBench ships via the upstream evaluator; pass `path=` "
                "to a JSONL (id, problem, tests, starter_code?)."
            )
        for rec in _iter_jsonl(path):
            yield Example(
                id=rec["id"],
                prompt=_format_lcb_prompt(rec),
                answer="",  # no canonical solution stored
                meta={"tests": rec["tests"], "starter_code": rec.get("starter_code", "")},
            )

    def score(self, model, examples, *, generate=None) -> dict[str, float]:
        from platform.rl.verifiers import CodeUnitTestVerifier
        gen = generate or _default_generate(model)
        n_total = 0
        n_pass = 0
        for ex in examples:
            n_total += 1
            response = gen(ex.prompt)
            v = CodeUnitTestVerifier(tests=ex.meta["tests"], all_or_nothing=True)
            if v(ex.prompt, response) == 1.0:
                n_pass += 1
        return {"pass_rate": (n_pass / n_total) if n_total else 0.0,
                "n_total": float(n_total)}


def _format_lcb_prompt(rec: dict) -> str:
    starter = rec.get("starter_code") or ""
    if starter:
        starter = f"\n\nStarter:\n```python\n{starter}\n```"
    return (
        f"Solve the problem. Reply with a complete Python solution inside a "
        f"```python ... ``` block.\n\n{rec['problem']}{starter}"
    )


# ----------------------------------------------------------------------------
# Engine-driven default generator
# ----------------------------------------------------------------------------


def _default_generate(model) -> Callable[[str], str]:
    """Return a ``str -> str`` generation closure backed by the in-process Engine.

    When ``model`` is already a callable, use it directly. When it's a
    :class:`platform.serving.engine.Engine`, run the async generate stream to
    completion. When it's a :class:`platform.model.transformer.Transformer`,
    wrap it in a TorchEngine on the fly. This is what makes the adapters
    unit-testable without a tokenizer pin or a fixed generator interface.
    """
    if callable(model) and not hasattr(model, "generate"):
        return model

    import asyncio

    def _gen(prompt: str) -> str:
        # Lazy imports so this module stays light at import time.
        from platform.serving.engine import EngineConfig, GenRequest

        engine = model
        if not hasattr(engine, "generate"):
            from platform.serving.torch_engine import TorchEngine
            engine = TorchEngine(EngineConfig(backend="torch", device="cpu"),
                                  model=model)

        # Fall back to a simple byte tokenizer when none is wired so the
        # adapter remains usable in CI.
        prompt_ids = list(prompt.encode("utf-8"))[:512]
        req = GenRequest(prompt_ids=prompt_ids, max_new_tokens=64,
                          temperature=0.0)

        async def _drain() -> list[int]:
            out: list[int] = []
            async for chunk in engine.generate(req):
                if not chunk.get("done"):
                    out.append(int(chunk["token_id"]))
            return out
        ids = asyncio.run(_drain())
        try:
            return bytes(i & 0xFF for i in ids).decode("utf-8", errors="replace")
        except Exception:
            return ""
    return _gen


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


REGISTRY: dict[str, type] = {
    "swe_bench_verified": SWEBenchVerifiedAdapter,
    "arc_agi_2": ARCAGI2Adapter,
    "hle": HLEAdapter,
    "mmmu": MMMUAdapter,
    "live_code_bench": LiveCodeBenchAdapter,
}


def get_adapter(name: str) -> BenchmarkAdapter:
    cls = REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"unknown 2026 benchmark adapter: {name!r}; known: {sorted(REGISTRY)}")
    return cls()


__all__ = [
    "Example",
    "BenchmarkAdapter",
    "SWEBenchVerifiedAdapter",
    "ARCAGI2Adapter",
    "HLEAdapter",
    "MMMUAdapter",
    "LiveCodeBenchAdapter",
    "REGISTRY",
    "get_adapter",
]
