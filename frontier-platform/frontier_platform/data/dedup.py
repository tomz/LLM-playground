"""Exact, near, and substring deduplication.

Real impls: `text-dedup` for MinHash-LSH; `google-research/deduplicate-text-datasets`
(suffix-array, Rust) for substring dedup.
"""
from __future__ import annotations
import hashlib
import re
import struct
from typing import Iterable, Iterator

_TOKEN_RE = re.compile(r"\w+")


def sha1_normalized(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode()).hexdigest()


def _shingles(text: str, n: int) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < n:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def _hash_int(seed: int, s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8, salt=struct.pack("<Q", seed & 0xFFFFFFFFFFFFFFFF))
    return int.from_bytes(h.digest(), "little")


class MinHashDeduper:
    """Streaming MinHash-LSH. Documents within Jaccard >= threshold are dropped.

    Pure Python. For real-scale (billions of docs) use datasketch + Redis/RocksDB.
    """
    def __init__(self, num_perm: int = 128, threshold: float = 0.8, ngram: int = 5, bands: int | None = None):
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram = ngram
        # Heuristic: choose bands so that band-count * rows = num_perm and the
        # LSH probability curve roughly matches the threshold.
        if bands is None:
            bands = max(1, num_perm // 4)
        while num_perm % bands != 0:
            bands -= 1
        self.bands = bands
        self.rows = num_perm // bands
        self._buckets: list[dict[bytes, set[str]]] = [dict() for _ in range(bands)]
        self._signatures: dict[str, list[int]] = {}

    def _signature(self, text: str) -> list[int]:
        shingles = _shingles(text, self.ngram)
        if not shingles:
            return [0] * self.num_perm
        sig = [min(_hash_int(p, sh) for sh in shingles) for p in range(self.num_perm)]
        return sig

    def add(self, doc_id: str, text: str) -> bool:
        sig = self._signature(text)
        candidates: set[str] = set()
        band_keys = []
        for b in range(self.bands):
            chunk = tuple(sig[b * self.rows : (b + 1) * self.rows])
            key = hashlib.blake2b(repr(chunk).encode(), digest_size=16).digest()
            band_keys.append(key)
            for cid in self._buckets[b].get(key, ()):
                candidates.add(cid)
        for cid in candidates:
            other = self._signatures[cid]
            same = sum(1 for a, c in zip(sig, other) if a == c)
            if same / self.num_perm >= self.threshold:
                return False
        # Novel — register.
        self._signatures[doc_id] = sig
        for b, key in enumerate(band_keys):
            self._buckets[b].setdefault(key, set()).add(doc_id)
        return True


def substring_dedup(corpus_path: str, min_match_len: int = 50) -> None:
    """Toy in-memory substring dedup over a text file.

    For each line, drop it if any contiguous run of >= min_match_len characters
    has been seen in a previous (kept) line. Rewrites the file in place.

    Scaling note: this is O(N^2) in characters; use the Rust suffix-array impl
    from google-research/deduplicate-text-datasets for production scale.
    """
    from pathlib import Path
    p = Path(corpus_path)
    seen: set[str] = set()
    kept: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line
        if len(s) >= min_match_len and any(
            s[i : i + min_match_len] in seen for i in range(0, len(s) - min_match_len + 1)
        ):
            continue
        kept.append(line)
        for i in range(0, max(0, len(s) - min_match_len + 1)):
            seen.add(s[i : i + min_match_len])
    p.write_text("\n".join(kept), encoding="utf-8")


def stream_exact_dedup(docs: Iterable[tuple[str, str]]) -> Iterator[tuple[str, str]]:
    seen: set[str] = set()
    for doc_id, text in docs:
        h = sha1_normalized(text)
        if h in seen:
            continue
        seen.add(h)
        yield doc_id, text
