"""Remove training docs that overlap eval sets via 13-gram digest set.

Real impl uses a Bloom filter sized for the full eval corpus; we use a Python
`set[int]` which is fine up to a few tens of millions of n-grams.
"""
from __future__ import annotations
import hashlib
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"\w+")


def _ngram_hashes(text: str, n: int) -> list[int]:
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < n:
        return []
    out = []
    for i in range(len(toks) - n + 1):
        gram = " ".join(toks[i : i + n])
        h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        out.append(int.from_bytes(h, "little"))
    return out


class Decontaminator:
    def __init__(self, eval_paths: list[str], n: int = 13, fpr: float = 1e-6):
        self.n = n
        self.fpr = fpr
        self._index: set[int] = set()
        self._per_set_size: dict[str, int] = {}
        self._hits_per_set: dict[str, int] = {}
        self._per_set_grams: dict[str, set[int]] = {}
        for path in eval_paths:
            p = Path(path)
            text = p.read_text(encoding="utf-8", errors="replace")
            grams: set[int] = set()
            for line in text.splitlines():
                for h in _ngram_hashes(line, n):
                    grams.add(h)
            self._per_set_grams[str(p)] = grams
            self._per_set_size[str(p)] = len(grams)
            self._hits_per_set[str(p)] = 0
            self._index |= grams

    def is_contaminated(self, text: str, max_overlap_frac: float = 0.5) -> bool:
        grams = _ngram_hashes(text, self.n)
        if not grams:
            return False
        hits = sum(1 for g in grams if g in self._index)
        frac = hits / len(grams)
        if frac >= max_overlap_frac:
            # Attribute to the eval set with the largest contribution.
            doc_grams = set(grams)
            best, best_overlap = None, -1
            for name, gset in self._per_set_grams.items():
                ov = len(doc_grams & gset)
                if ov > best_overlap:
                    best, best_overlap = name, ov
            if best is not None:
                self._hits_per_set[best] = self._hits_per_set.get(best, 0) + 1
            return True
        return False

    def report(self) -> dict:
        return {
            name: {
                "ngrams": self._per_set_size[name],
                "flagged_train_docs": self._hits_per_set.get(name, 0),
            }
            for name in self._per_set_size
        }
