"""Quality, language, and toxicity filters."""
from __future__ import annotations
import re
from dataclasses import dataclass

# Tiny English stopword list — enough for the heuristic LID + Gopher rules.
_EN_STOPWORDS = frozenset(
    "the be to of and a in that have i it for not on with he as you do at "
    "this but his by from they we say her she or an will my one all would "
    "there their what so up out if about who get which go me when make can".split()
)
_BAD_WORDS = {
    "hate": ["hate", "racist", "bigot"],
    "violence": ["kill", "murder", "attack", "shoot"],
    "sexual": ["porn", "explicit"],
    "selfharm": ["suicide", "selfharm", "self-harm"],
}
_BULLET_RE = re.compile(r"^\s*([-*•‣◦⁃]|\d+[.)])\s+")


@dataclass
class FilterVerdict:
    keep: bool
    reason: str = ""
    scores: dict | None = None


def detect_language(text: str) -> tuple[str, float]:
    """Return (iso639 code, confidence). Real: fasttext lid.176."""
    from importlib.util import find_spec
    if find_spec("fasttext") is not None:
        # Real impl would lazy-load lid.176.bin here. We don't ship the model,
        # so fall through to the stopword heuristic.
        pass
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    if not words:
        return ("unk", 0.0)
    hits = sum(1 for w in words if w in _EN_STOPWORDS)
    ratio = hits / len(words)
    if ratio >= 0.05:
        # confidence saturates around 0.99
        conf = min(0.99, 0.65 + ratio * 2.0)
        return ("en", conf)
    return ("unk", 0.0)


def gopher_rules(text: str) -> FilterVerdict:
    """Heuristic quality filter from Rae et al. 2021 (Gopher paper)."""
    words = text.split()
    n = len(words)
    if n < 50:
        return FilterVerdict(False, "too-short")
    if n > 100_000:
        return FilterVerdict(False, "too-long")
    mean_wlen = sum(len(w) for w in words) / n
    if not (3.0 <= mean_wlen <= 10.0):
        return FilterVerdict(False, "mean-word-len")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        bad = sum(1 for ln in lines if _BULLET_RE.match(ln) or ln.rstrip().endswith("…") or ln.rstrip().endswith("..."))
        if bad / len(lines) > 0.10:
            return FilterVerdict(False, "bullets-or-ellipsis")
    alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha / n < 0.80:
        return FilterVerdict(False, "non-alpha")
    stops = sum(1 for w in words if w.lower() in _EN_STOPWORDS)
    if stops < 2:
        return FilterVerdict(False, "no-stopwords")
    return FilterVerdict(True, "ok")


def quality_classifier(text: str) -> float:
    """FineWeb-Edu style score in [0, 5]. Real impl is a fastText classifier."""
    words = text.split()
    n = len(words)
    if n < 20:
        return 0.0
    length_score = min(1.0, n / 200.0)
    punct = sum(1 for c in text if c in ".,;:?!")
    punct_ratio = punct / max(1, len(text))
    punct_score = 1.0 - min(1.0, abs(punct_ratio - 0.03) / 0.05)
    stops = sum(1 for w in words if w.lower() in _EN_STOPWORDS)
    stop_score = min(1.0, stops / max(1, n // 20))
    composite = (length_score + punct_score + stop_score) / 3.0
    return round(5.0 * composite, 4)


def toxicity_score(text: str) -> dict:
    from importlib.util import find_spec
    if find_spec("detoxify") is not None:
        # Real impl: cache Detoxify model on first call.
        pass
    low = text.lower()
    tokens = re.findall(r"[a-z']+", low)
    total = max(1, len(tokens))
    out: dict[str, float] = {}
    for cat, words in _BAD_WORDS.items():
        hits = sum(1 for t in tokens if t in words)
        out[cat] = min(1.0, hits / total * 20.0)
    return out


def pipeline(text: str, lang_allowlist: set[str]) -> FilterVerdict:
    lang, conf = detect_language(text)
    if lang not in lang_allowlist or conf < 0.65:
        return FilterVerdict(False, "lang")
    v = gopher_rules(text)
    if not v.keep:
        return v
    if quality_classifier(text) < 3.0:
        return FilterVerdict(False, "low-quality")
    tox = toxicity_score(text)
    if max(tox.values()) > 0.9:
        return FilterVerdict(False, "toxic", scores=tox)
    return FilterVerdict(True, "ok")
