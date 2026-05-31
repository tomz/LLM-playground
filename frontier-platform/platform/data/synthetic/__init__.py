"""Synthetic data factory for SFT / RLVR / reasoning-trace generation.

This package replaces what used to be a 16-line word-bag corpus generator with
a real factory: a pluggable :class:`Teacher` produces candidate samples for a
pluggable :class:`GenerationPolicy`, candidates are filtered by a verifier
(reuses :mod:`platform.rl.verifiers`), deduplicated with MinHash-LSH (reuses
:mod:`platform.data.dedup`), and checked against an evaluation contamination
index (reuses :mod:`platform.data.decontaminate`). Every accepted *and* rejected
sample is recorded in a :class:`SampleRecord` so the pipeline has data lineage.

Back-compatibility: ``write_corpus`` is still importable from this module and
produces the same word-bag corpus the test suite relies on.

Design goals:

* Teachers, policies, verifiers, dedupers, and decontaminators are all
  *injected*, so we can unit-test the factory with a deterministic teacher and
  drive a real engine (vLLM / TorchEngine) in production.
* The factory is an *iterator* of :class:`SampleRecord` so it can stream
  millions of samples without holding them all in memory.
* JSONL lineage output is one record per line for trivially-parallel
  downstream processing.
* Zero hard dependencies beyond stdlib + (optional) sympy via the verifier.

See ``docs/17a-frontier-model-gap-research-v2.md`` §2.3 and
``docs/18-implementation-roadmap.md`` ★ 1 for the motivation.
"""
from __future__ import annotations

import random
from pathlib import Path

from .factory import FactoryStats, SyntheticFactory
from .lineage import SampleRecord, read_lineage_jsonl, write_lineage_jsonl
from .policies import (
    GenerationPolicy,
    MathProblemPolicy,
    QAPolicy,
    ReasoningTracePolicy,
    RephrasePolicy,
    TemplatePolicy,
    TextbookPolicy,
)
from .teacher import (
    CallableTeacher,
    EchoTeacher,
    EngineTeacher,
    Teacher,
    TemplateTeacher,
)

_WORDS = (
    "the quick brown fox jumps over the lazy dog "
    "alpha beta gamma delta epsilon zeta eta theta iota "
    "model training data pipeline gradient descent loss curve"
).split()


def write_corpus(
    root: str | Path,
    n_files: int = 20,
    words_per_file: int = 200,
    seed: int = 0,
) -> Path:
    """Write a tiny deterministic word-bag corpus for tests.

    Kept for back-compat with ``tests/conftest.py`` and
    ``scripts/smoke_pipeline.py``; new code should reach for
    :class:`SyntheticFactory` instead.
    """
    rng = random.Random(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        words = [rng.choice(_WORDS) for _ in range(words_per_file)]
        (root / f"doc_{i:04d}.txt").write_text(" ".join(words) + ".\n", encoding="utf-8")
    return root


__all__ = [
    # back-compat
    "write_corpus",
    # teachers
    "Teacher",
    "EchoTeacher",
    "TemplateTeacher",
    "CallableTeacher",
    "EngineTeacher",
    # policies
    "GenerationPolicy",
    "TemplatePolicy",
    "RephrasePolicy",
    "TextbookPolicy",
    "MathProblemPolicy",
    "QAPolicy",
    "ReasoningTracePolicy",
    # factory + records
    "SyntheticFactory",
    "FactoryStats",
    "SampleRecord",
    "write_lineage_jsonl",
    "read_lineage_jsonl",
]
