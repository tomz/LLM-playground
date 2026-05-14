"""Exact, near, and substring deduplication.

Real impls: `text-dedup` for MinHash-LSH; `google-research/deduplicate-text-datasets`
(suffix-array, Rust) for substring dedup.
"""
from __future__ import annotations
import hashlib
from typing import Iterable, Iterator


def sha1_normalized(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode()).hexdigest()


class MinHashDeduper:
    """Streaming MinHash-LSH. Documents within Jaccard >= threshold are dropped."""
    def __init__(self, num_perm: int = 128, threshold: float = 0.8, ngram: int = 5):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram = ngram
        # real: datasketch.MinHashLSH backed by Redis or RocksDB

    def add(self, doc_id: str, text: str) -> bool:
        """Return True if the doc is novel (kept)."""
        raise NotImplementedError


def substring_dedup(corpus_path: str, min_match_len: int = 50) -> None:
    """Build suffix array, mask any substring >= min_match_len appearing >1 times."""
    raise NotImplementedError


def stream_exact_dedup(docs: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    seen: set[str] = set()
    for doc_id, text in docs:
        h = sha1_normalized(text)
        if h in seen:
            continue
        seen.add(h)
        yield doc_id, text
