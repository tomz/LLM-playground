"""Train/eval contamination detection via n-gram overlap.

Benchmark contamination — eval examples (or close paraphrases) leaking into the
training corpus — silently inflates scores and is one of the most common ways a
"frontier" number turns out to be fake. The standard cheap detector (used by
GPT-4, Llama, and the lm-evaluation-harness decontamination tooling) is
**n-gram overlap**: build a set of token n-grams from the training corpus, then
flag any eval example whose n-grams overlap the training set above a threshold.

This module is dependency-free and exact (not a learned model): it builds a
hashed n-gram index over the training shards and reports, per eval task, the
fraction of examples that are contaminated. For web-scale corpora the same index
is sharded across the cluster and backed by a Bloom filter / RocksDB; the API
(`ContaminationIndex.add_document` / `.contamination_rate`) is unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_RE = re.compile(r"\w+")


def _ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    toks = _WORD_RE.findall(text.lower())
    if len(toks) < n:
        return [tuple(toks)] if toks else []
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


@dataclass
class ContaminationIndex:
    """Hashed n-gram index over a training corpus for overlap detection.

    ``n`` is the n-gram order (13 is the GPT-3/Llama convention for token-level;
    8 is reasonable for word-level). An eval example is *contaminated* when the
    fraction of its n-grams seen in the training corpus meets ``threshold``.
    """

    n: int = 8
    threshold: float = 0.8
    _seen: set[int] = field(default_factory=set)

    def add_document(self, text: str) -> None:
        for g in _ngrams(text, self.n):
            self._seen.add(hash(g))

    def add_documents(self, texts) -> None:
        for t in texts:
            self.add_document(t)

    def overlap(self, text: str) -> float:
        """Fraction of ``text``'s n-grams that appear in the training corpus."""
        grams = _ngrams(text, self.n)
        if not grams:
            return 0.0
        hit = sum(1 for g in grams if hash(g) in self._seen)
        return hit / len(grams)

    def is_contaminated(self, text: str) -> bool:
        return self.overlap(text) >= self.threshold

    def contamination_rate(self, eval_examples) -> float:
        """Fraction of ``eval_examples`` (iterable of str) that are contaminated."""
        examples = list(eval_examples)
        if not examples:
            return 0.0
        n_bad = sum(1 for ex in examples if self.is_contaminated(ex))
        return n_bad / len(examples)


def contamination_report(
    train_texts,
    eval_tasks: dict[str, list[str]],
    *,
    n: int = 8,
    threshold: float = 0.8,
) -> dict[str, float]:
    """Build an index over ``train_texts`` and return ``{task: contamination_rate}``.

    ``eval_tasks`` maps a task name to its list of eval example strings (the
    prompt, or prompt+answer). Returned rates are in ``[0, 1]``.
    """
    idx = ContaminationIndex(n=n, threshold=threshold)
    idx.add_documents(train_texts)
    return {task: idx.contamination_rate(examples) for task, examples in eval_tasks.items()}
