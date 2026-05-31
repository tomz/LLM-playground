"""Input/output content classifiers (the 'classifier sandwich').

A classifier scores text along harm categories (hate, violence, sexual,
selfharm, ...) on ``[0, 1]`` where 0 is safe. The serving stack puts one on
the *input* path (block unsafe prompts before the model runs) and one on the
*output* path (block unsafe completions before they reach the user) — hence
the "sandwich".

This module replaces what used to be a single 30-line keyword counter with a
pluggable :class:`Classifier` protocol plus three concrete implementations:

* :class:`KeywordClassifier`  — the original dependency-free heuristic; kept
  as the fallback so CI / GPU-less laptops still exercise the safety path.
* :class:`LlamaGuardClassifier` — adapter for a real model (Meta Llama-Guard,
  ShieldGemma, fine-tuned RoBERTa, etc.). Loads lazily so the import doesn't
  pull in ``transformers`` for installs that don't need it.
* :class:`ClassifierEnsemble` — combines multiple classifiers by max-per-
  category (the conservative production choice — strictest wins).

Back-compat: :class:`InputClassifier` and :class:`OutputClassifier` still
exist with the same ``.score(...)`` signature. Internally they now delegate
to a configurable backing :class:`Classifier` (default
:class:`KeywordClassifier`), so existing tests and callers see no behaviour
change unless they opt in.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Protocol

# ----------------------------------------------------------------------------
# Category vocabulary
# ----------------------------------------------------------------------------

# Canonical category list used everywhere in this module. Matches the keys the
# existing OutputClassifier returned, and the existing
# ``platform.safety.redteam.judges.ClassifierJudge`` key_map already maps these
# onto the gate categories (``hate→bias``, ``violence→cbrn``, etc.).
CATEGORIES: tuple[str, ...] = ("hate", "violence", "sexual", "selfharm")


def _empty_scores() -> dict[str, float]:
    return {c: 0.0 for c in CATEGORIES}


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------


class Classifier(Protocol):
    """Score a single ``(prompt, completion)`` pair along :data:`CATEGORIES`.

    Implementations may ignore ``completion`` when only the prompt is being
    scored (input path); :class:`InputClassifier` calls with
    ``completion=None``. Returned dicts must carry every key in
    :data:`CATEGORIES` (missing keys are treated as 0 by callers, but
    explicit zeros keep downstream code branch-free).
    """

    name: str

    def score(self, prompt: str, completion: str | None = None) -> dict[str, float]:
        ...


# ----------------------------------------------------------------------------
# Keyword classifier (the original fallback)
# ----------------------------------------------------------------------------

# Category → bad-word lexicon. Identical to the original module so the
# existing tests' exact match on "kill" / "attack" / "bomb" still passes.
_DEFAULT_LEXICON: dict[str, list[str]] = {
    "hate": ["hate", "slur", "racist", "bigot"],
    "violence": ["kill", "murder", "attack", "shoot", "bomb"],
    "sexual": ["porn", "explicit", "nsfw"],
    "selfharm": ["suicide", "selfharm", "self-harm"],
}


def _kw_score(text: str, lexicon: dict[str, list[str]]) -> dict[str, float]:
    low = text.lower()
    tokens = re.findall(r"[a-z-]+", low)
    n = max(1, len(tokens))
    out = _empty_scores()
    for cat, words in lexicon.items():
        hits = sum(1 for t in tokens if t in words)
        out[cat] = min(1.0, hits / n * 10.0)
    return out


@dataclass
class KeywordClassifier:
    """Dependency-free fallback classifier.

    Score per category = ``min(1, 10 * hits / token_count)`` where ``hits`` is
    the number of tokens (case-insensitive, ``[a-z-]+``) matching the
    category's lexicon. The ``× 10`` is intentional: a single bad token in a
    short prompt should saturate to 1.0, not 1/10.

    A real deployment swaps this out for :class:`LlamaGuardClassifier`. This
    one stays as the test fallback so CI doesn't need to load model weights.
    """

    lexicon: dict[str, list[str]] = field(default_factory=lambda: dict(_DEFAULT_LEXICON))
    name: str = "keyword"

    def score(self, prompt: str, completion: str | None = None) -> dict[str, float]:
        if completion is None:
            return _kw_score(prompt, self.lexicon)
        a = _kw_score(prompt, self.lexicon)
        b = _kw_score(completion, self.lexicon)
        # Output classifier sandwich semantics: max over prompt and completion.
        return {k: max(a[k], b[k]) for k in CATEGORIES}


# ----------------------------------------------------------------------------
# Llama-Guard / trained-model adapter
# ----------------------------------------------------------------------------


class LlamaGuardClassifier:
    """Adapter for a real safety classifier model (Llama-Guard, ShieldGemma,
    fine-tuned RoBERTa, etc.).

    Constructor parameters:

    * ``model_id``      — HF model id or local path. Loaded lazily on first
                          :meth:`score` call so import time stays cheap.
    * ``categories``    — which categories the model emits, in the order its
                          logits / outputs are aligned to. Defaults to the
                          standard 4 we use everywhere.
    * ``backend``       — pluggable inference path. ``"transformers"`` is the
                          default; ``"callable"`` lets you wire any
                          ``fn(prompt: str, completion: str|None) ->
                          dict[str, float]`` for tests or RPC bridges.

    On model load failure we *warn and fall back* to :class:`KeywordClassifier`
    rather than raising — the serving stack must never go un-classified on a
    missing-weights condition. The warning is once-per-model-id so logs aren't
    spammed.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Guard-3-8B",
        *,
        categories: tuple[str, ...] = CATEGORIES,
        backend: str = "transformers",
        callable_fn=None,
        name: str | None = None,
    ):
        self.model_id = model_id
        self.categories = tuple(categories)
        self.backend = backend
        self._callable_fn = callable_fn
        self.name = name or f"llama_guard:{model_id}"
        self._impl: Classifier | None = None
        self._load_failed = False
        if backend == "callable":
            if callable_fn is None:
                raise ValueError("backend='callable' requires callable_fn=…")

    def _ensure_loaded(self) -> Classifier:
        if self._impl is not None:
            return self._impl
        if self._load_failed:
            return self._impl  # KeywordClassifier from the previous attempt

        if self.backend == "callable":
            fn = self._callable_fn

            @dataclass
            class _Wrap:
                name: str = self.name

                def score(self_inner, prompt, completion=None):
                    raw = dict(fn(prompt, completion))
                    out = _empty_scores()
                    for k, v in raw.items():
                        if k in CATEGORIES:
                            out[k] = float(v)
                    return out

            self._impl = _Wrap()
            return self._impl

        if self.backend == "transformers":
            try:
                # Lazy import keeps the dep optional.
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
                self._impl = _TransformersLlamaGuardImpl(
                    self._tokenizer, self._model, self.categories, self.name,
                )
                return self._impl
            except Exception as e:
                warnings.warn(
                    f"LlamaGuardClassifier could not load {self.model_id!r} "
                    f"({type(e).__name__}: {e}); falling back to KeywordClassifier. "
                    "This is fine for CI; for production install `transformers` "
                    "and the model weights.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._impl = KeywordClassifier()
                self._load_failed = True
                return self._impl

        raise ValueError(f"unknown LlamaGuardClassifier backend: {self.backend!r}")

    def score(self, prompt: str, completion: str | None = None) -> dict[str, float]:
        impl = self._ensure_loaded()
        return impl.score(prompt, completion)


@dataclass
class _TransformersLlamaGuardImpl:
    """Internal worker that calls a loaded HF Llama-Guard-style model.

    Llama-Guard models are CausalLMs trained to emit either a single ``safe``
    token or a structured ``unsafe\\nS1,S2,...`` token sequence. This adapter
    runs a short conditional generation and parses the response into the
    canonical category-score dict. The :class:`LlamaGuardClassifier` outer
    object holds the lazy-load and the fallback policy; we keep this worker
    tiny and unit-testable.
    """

    tokenizer: object
    model: object
    categories: tuple[str, ...]
    name: str

    def score(self, prompt: str, completion: str | None = None) -> dict[str, float]:
        text = prompt if completion is None else f"{prompt}\n\n{completion}"
        inputs = self.tokenizer(text, return_tensors="pt")
        out = self.model.generate(  # type: ignore[attr-defined]
            **inputs, max_new_tokens=32, do_sample=False,
        )
        decoded = self.tokenizer.decode(  # type: ignore[attr-defined]
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )
        return _parse_llama_guard_response(decoded, self.categories)


# Llama-Guard's S-codes map onto our category names. Conservative coverage —
# unmapped codes are silently dropped. Real deployments would extend this for
# the full Llama-Guard 3 taxonomy.
_LLAMA_GUARD_CATEGORY_MAP: dict[str, str] = {
    "S1": "violence",   # Violent Crimes
    "S2": "violence",   # Non-Violent Crimes (closest fit in our 4-cat schema)
    "S3": "sexual",     # Sex-Related Crimes
    "S4": "sexual",     # Child Exploitation
    "S5": "hate",       # Defamation
    "S6": "violence",   # Specialized Advice (treated as violence-adjacent)
    "S10": "hate",      # Hate
    "S11": "selfharm",  # Suicide & Self-Harm
    "S12": "sexual",    # Sexual Content
}


def _parse_llama_guard_response(text: str, categories: tuple[str, ...]) -> dict[str, float]:
    """Parse a Llama-Guard generation into a category-score dict.

    Llama-Guard 3 emits either ``safe`` (all-zero scores) or
    ``unsafe\\nS1,S5,S11``. Anything we recognise maps to 1.0; the rest stay 0.
    """
    out = {c: 0.0 for c in categories}
    head = text.strip().splitlines()[0].strip().lower() if text.strip() else ""
    if head.startswith("safe"):
        return out
    # 'unsafe' on line 0, codes on line 1.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    codes_line = lines[1] if len(lines) > 1 else ""
    for tok in re.split(r"[\s,]+", codes_line):
        cat = _LLAMA_GUARD_CATEGORY_MAP.get(tok.upper())
        if cat and cat in out:
            out[cat] = 1.0
    return out


# ----------------------------------------------------------------------------
# Ensemble
# ----------------------------------------------------------------------------


class ClassifierEnsemble:
    """Combine multiple classifiers. Default reduction: max-per-category.

    Conservative semantics: the strictest signal wins. Swap ``reduce='mean'``
    for averaging when you have many calibrated classifiers, or
    ``reduce='min'`` if you want unanimous agreement (rare; mostly useful for
    research ablations).
    """

    def __init__(self, classifiers: list[Classifier], *,
                 reduce: str = "max", name: str = "ensemble"):
        if not classifiers:
            raise ValueError("ensemble needs at least one classifier")
        if reduce not in ("max", "mean", "min"):
            raise ValueError(f"reduce must be 'max', 'mean', or 'min'; got {reduce!r}")
        self.classifiers = list(classifiers)
        self.reduce = reduce
        self.name = name

    def score(self, prompt: str, completion: str | None = None) -> dict[str, float]:
        per = [c.score(prompt, completion) for c in self.classifiers]
        out = _empty_scores()
        for c in CATEGORIES:
            vs = [float(s.get(c, 0.0)) for s in per]
            if self.reduce == "max":
                out[c] = max(vs)
            elif self.reduce == "min":
                out[c] = min(vs)
            else:  # mean
                out[c] = sum(vs) / len(vs)
        return out


# ----------------------------------------------------------------------------
# Back-compat: InputClassifier / OutputClassifier
# ----------------------------------------------------------------------------


class InputClassifier:
    """Score a prompt only. Back-compatible with the original module.

    Accepts an optional ``backing`` :class:`Classifier`; defaults to
    :class:`KeywordClassifier` so the existing test exact-value asserts pass."""

    def __init__(self, backing: Classifier | None = None):
        self.backing = backing or KeywordClassifier()

    def score(self, prompt: str) -> dict[str, float]:
        return self.backing.score(prompt, None)


class OutputClassifier:
    """Score a (prompt, completion) pair with the sandwich semantics.

    Defaults to :class:`KeywordClassifier`, which combines the two by max
    per-category — exactly what the previous module did, so existing tests
    keep their exact-value expectations."""

    def __init__(self, backing: Classifier | None = None):
        self.backing = backing or KeywordClassifier()

    def score(self, prompt: str, completion: str) -> dict[str, float]:
        return self.backing.score(prompt, completion)


__all__ = [
    "CATEGORIES",
    "Classifier",
    "KeywordClassifier",
    "LlamaGuardClassifier",
    "ClassifierEnsemble",
    "InputClassifier",
    "OutputClassifier",
]
