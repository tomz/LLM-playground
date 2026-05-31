"""Weighted multi-source mixture sampler.

Production frontier runs train on a *mixture* of corpora — web (CommonCrawl /
FineWeb), code (GitHub), math/STEM (arXiv), wiki/books, instruction data —
weighted by per-source importance. This module wraps any number of
``StreamingLoader`` instances and returns batches in a Dirichlet-stable
weighted-sampling order.

The mixture weights are *per-batch probabilities*, not per-step ratios, so
small-fraction sources still get sampled (every few batches) even when their
weight is < 1/n. State (per-loader cursors + the mixture RNG state) is
serializable so resume is deterministic across topology changes.

Usage::

    from distgpt.data.streaming import StreamingLoader
    from distgpt.data.mixture import MixtureLoader

    sources = {
        "web":  StreamingLoader("data/fineweb",   ...),
        "code": StreamingLoader("data/github",    ...),
        "math": StreamingLoader("data/arxiv",     ...),
    }
    mix = MixtureLoader(sources, weights={"web": 0.6, "code": 0.3, "math": 0.1}, seed=0)
    for _ in range(steps):
        x, y = mix.next_batch()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass
class MixtureState:
    rng_state: list = field(default_factory=list)        # numpy RNG state
    per_source_steps: dict[str, int] = field(default_factory=dict)


class MixtureLoader:
    """Weighted sampler over named ``StreamingLoader`` instances.

    Each ``next_batch()`` call samples a source name from the categorical
    distribution defined by ``weights`` and forwards the call. Per-source
    sampling counts are tracked for diagnostics ('did the math corpus
    actually get its 10%?').
    """

    def __init__(
        self,
        sources: Mapping[str, object],
        weights: Mapping[str, float],
        seed: int = 0,
    ):
        if not sources:
            raise ValueError("MixtureLoader needs at least one source")
        # Strict check that weights and sources match — silent missing
        # weights would skip a source entirely, which is the kind of bug
        # that costs a frontier run a week of wrong data ratios.
        missing = set(sources) - set(weights)
        extra = set(weights) - set(sources)
        if missing or extra:
            raise ValueError(
                f"sources/weights mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        # Normalize so we accept ratios or probabilities equally; raise on
        # all-zero / negative weights.
        w = np.array([float(weights[k]) for k in sources], dtype=np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"weights must be non-negative and sum > 0; got {w}")
        self.names = list(sources.keys())
        self.sources = dict(sources)
        self.probs = w / w.sum()
        self.rng = np.random.default_rng(seed)
        self.state = MixtureState(per_source_steps={k: 0 for k in self.names})

    def next_batch(self):
        # Choose a source by the categorical, then forward.
        idx = int(self.rng.choice(len(self.names), p=self.probs))
        name = self.names[idx]
        self.state.per_source_steps[name] += 1
        return self.sources[name].next_batch()

    def state_dict(self) -> dict:
        # numpy RNG state is non-trivial to serialize via plain dict; use
        # its bit_generator's state which round-trips cleanly through JSON.
        bg = self.rng.bit_generator.state
        return {
            "bit_generator_state": bg,
            "per_source_steps": dict(self.state.per_source_steps),
            "sources": {k: v.state_dict() for k, v in self.sources.items()
                         if hasattr(v, "state_dict")},
        }

    def load_state_dict(self, sd: dict) -> None:
        if "bit_generator_state" in sd:
            self.rng.bit_generator.state = sd["bit_generator_state"]
        self.state.per_source_steps.update(sd.get("per_source_steps", {}))
        for name, child_state in (sd.get("sources") or {}).items():
            if name in self.sources and hasattr(self.sources[name], "load_state_dict"):
                self.sources[name].load_state_dict(child_state)

    def fractions(self) -> dict[str, float]:
        """Actual realized sampling fractions so far. Use this in the
        training log to verify the mixture is being honored at scale."""
        total = sum(self.state.per_source_steps.values())
        if total == 0:
            return {k: 0.0 for k in self.names}
        return {k: v / total for k, v in self.state.per_source_steps.items()}


__all__ = ["MixtureLoader", "MixtureState"]
