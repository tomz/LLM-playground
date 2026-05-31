"""The synthetic data factory itself.

Given a :class:`Teacher` and a :class:`GenerationPolicy`, ``SyntheticFactory``
streams prompts → teacher responses → acceptance verifier → dedup →
contamination check → :class:`SampleRecord`. Every step is optional so a unit
test can isolate exactly the part it wants to exercise.

The factory is an iterator. It will keep asking the policy for prompts and
the teacher for completions until ``n`` *accepted* samples have been produced
or ``max_attempts`` has been hit. Stats (acceptance rate, rejection reasons)
are exposed on ``.stats`` for end-of-run logging.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .lineage import SampleRecord, write_lineage_jsonl
from .policies import GenerationPolicy
from .teacher import Teacher


@dataclass
class FactoryStats:
    """Aggregate counters for a factory run, suitable for end-of-run logging."""

    requested: int = 0
    attempted: int = 0
    accepted: int = 0
    rejected_verifier: int = 0
    rejected_duplicate: int = 0
    rejected_contaminated: int = 0
    rejected_empty: int = 0
    teacher_errors: int = 0

    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.attempted)

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected_verifier": self.rejected_verifier,
            "rejected_duplicate": self.rejected_duplicate,
            "rejected_contaminated": self.rejected_contaminated,
            "rejected_empty": self.rejected_empty,
            "teacher_errors": self.teacher_errors,
            "acceptance_rate": self.acceptance_rate(),
        }


@dataclass
class SyntheticFactory:
    """Stream :class:`SampleRecord` s by composing a teacher + policy + filters.

    Wiring:

    * ``teacher``                : :class:`Teacher` (echo / template / engine)
    * ``policy``                 : :class:`GenerationPolicy` (math / textbook / …)
    * ``accept_threshold``       : minimum verifier score to accept (if the policy
                                   ships one).
    * ``deduper``                : optional :class:`MinHashDeduper`; near-duplicate
                                   teacher *responses* are rejected.
    * ``decontaminator``         : optional :class:`Decontaminator`; responses that
                                   overlap an eval set are rejected.
    * ``max_attempts``           : safety belt against an infinite generation loop
                                   when the verifier rejects everything.
    * ``seed``                   : controls the policy's RNG and the sample IDs so
                                   runs are reproducible bit-for-bit (given a
                                   deterministic teacher).
    * ``emit_rejections``        : when True, the iterator also yields rejected
                                   records (with ``accepted=False`` and a
                                   ``rejection_reason``) for full lineage. The
                                   ``n`` budget always counts *accepted* samples.
    """

    teacher: Teacher
    policy: GenerationPolicy
    accept_threshold: float = 0.5
    deduper: object | None = None              # platform.data.dedup.MinHashDeduper
    decontaminator: object | None = None       # platform.data.decontaminate.Decontaminator
    max_attempts_per_sample: int = 8
    seed: int = 0
    emit_rejections: bool = False
    teacher_kwargs: dict = field(default_factory=dict)
    stats: FactoryStats = field(default_factory=FactoryStats)

    def generate(self, n: int) -> Iterator[SampleRecord]:
        """Yield up to ``n`` accepted samples (and rejections if enabled)."""
        self.stats = FactoryStats(requested=n)
        rng = random.Random(self.seed)
        verifier = self.policy.acceptance_verifier()
        # Pull prompts lazily: the policy is allowed to be an infinite generator,
        # we just stop once we have ``n`` accepted samples.
        budget = n * self.max_attempts_per_sample
        prompt_iter = iter(self.policy.prompts(budget, rng=rng))
        counter = itertools.count()
        produced = 0
        for prompt in prompt_iter:
            if produced >= n:
                break
            self.stats.attempted += 1
            idx = next(counter)
            try:
                response = self.teacher.generate(prompt, **self.teacher_kwargs)
            except Exception as exc:  # pragma: no cover - depends on teacher
                self.stats.teacher_errors += 1
                if self.emit_rejections:
                    yield self._mk_record(
                        idx, prompt, "", accepted=False, score=0.0,
                        reason=f"teacher_error:{type(exc).__name__}",
                    )
                continue

            response = str(response or "")
            if not response.strip():
                self.stats.rejected_empty += 1
                if self.emit_rejections:
                    yield self._mk_record(
                        idx, prompt, response, accepted=False, score=0.0,
                        reason="empty_response",
                    )
                continue

            # Acceptance verifier
            score = 1.0
            if verifier is not None:
                score = float(verifier(prompt, response))
                if score < self.accept_threshold:
                    self.stats.rejected_verifier += 1
                    if self.emit_rejections:
                        yield self._mk_record(
                            idx, prompt, response, accepted=False, score=score,
                            reason="verifier",
                        )
                    continue

            # Near-duplicate dedup over responses (so two policies that produce
            # the same answer text don't ship both).
            if self.deduper is not None:
                doc_id = f"{self.seed}:{idx}"
                novel = self.deduper.add(doc_id, response)
                if not novel:
                    self.stats.rejected_duplicate += 1
                    if self.emit_rejections:
                        yield self._mk_record(
                            idx, prompt, response, accepted=False, score=score,
                            reason="duplicate",
                        )
                    continue

            # Contamination check vs. eval index
            if self.decontaminator is not None and self.decontaminator.is_contaminated(response):
                self.stats.rejected_contaminated += 1
                if self.emit_rejections:
                    yield self._mk_record(
                        idx, prompt, response, accepted=False, score=score,
                        reason="contaminated",
                    )
                continue

            self.stats.accepted += 1
            produced += 1
            yield self._mk_record(
                idx, prompt, response, accepted=True, score=score, reason=None,
            )

    def write_jsonl(
        self, n: int, path: str | Path, *, only_accepted: bool = True,
    ) -> Path:
        """Convenience: generate ``n`` samples and stream them to a JSONL file."""
        # Always emit rejections if we're writing the full lineage, even when
        # the caller didn't set emit_rejections explicitly.
        prior = self.emit_rejections
        self.emit_rejections = self.emit_rejections or (not only_accepted)
        try:
            return write_lineage_jsonl(
                self.generate(n), path, only_accepted=only_accepted,
            )
        finally:
            self.emit_rejections = prior

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mk_record(
        self,
        idx: int,
        prompt: str,
        response: str,
        *,
        accepted: bool,
        score: float,
        reason: str | None,
    ) -> SampleRecord:
        sample_id = self._sample_id(idx, prompt)
        return SampleRecord(
            sample_id=sample_id,
            teacher=getattr(self.teacher, "name", type(self.teacher).__name__),
            policy=getattr(self.policy, "name", type(self.policy).__name__),
            seed=self.seed,
            prompt=prompt,
            response=response,
            accepted=accepted,
            verifier_score=score,
            rejection_reason=reason,
        )

    def _sample_id(self, idx: int, prompt: str) -> str:
        h = hashlib.blake2b(
            f"{self.seed}|{idx}|{prompt}".encode("utf-8"), digest_size=8
        ).hexdigest()
        return f"syn-{h}"
