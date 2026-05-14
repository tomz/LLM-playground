"""Remove training docs that overlap eval sets via 13-gram bloom filter."""
from __future__ import annotations


class Decontaminator:
    def __init__(self, eval_paths: list[str], n: int = 13, fpr: float = 1e-6):
        self.n = n
        # real: build pybloomfilter.BloomFilter over n-gram hashes of every eval doc

    def is_contaminated(self, text: str, max_overlap_frac: float = 0.5) -> bool:
        raise NotImplementedError

    def report(self) -> dict:
        """Per-eval-set contamination stats for the audit log."""
        raise NotImplementedError
