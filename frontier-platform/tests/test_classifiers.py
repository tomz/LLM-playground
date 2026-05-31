"""Tests for the pluggable safety classifier interface.

The new shape: a :class:`Classifier` protocol + three concrete implementations
(:class:`KeywordClassifier`, :class:`LlamaGuardClassifier`,
:class:`ClassifierEnsemble`) with the existing :class:`InputClassifier` /
:class:`OutputClassifier` reduced to thin back-compat shims over a configurable
backing classifier.

These tests pin:
- back-compat: the old InputClassifier / OutputClassifier behaviour is bit-for-bit
- the Classifier protocol contract (every key in CATEGORIES, [0, 1] floats)
- KeywordClassifier semantics on edge cases (empty, sandwich)
- LlamaGuardClassifier lazy-load + graceful fallback on missing weights
- LlamaGuardClassifier 'callable' backend
- The Llama-Guard response parser (safe / unsafe / unknown codes)
- ClassifierEnsemble reductions (max / mean / min)
- Drop-in usage from InputClassifier / OutputClassifier
"""
from __future__ import annotations

import warnings

import pytest

from platform.safety.classifiers import (
    CATEGORIES,
    ClassifierEnsemble,
    InputClassifier,
    KeywordClassifier,
    LlamaGuardClassifier,
    OutputClassifier,
    _parse_llama_guard_response,
)


# ---- back-compat ----------------------------------------------------------


def test_input_classifier_default_is_keyword_and_matches_old_behaviour():
    """The old behaviour: 'kill someone' should trigger violence, photosynthesis
    should not. (Same asserts as the original tests/test_safety.py.)"""
    c = InputClassifier()
    s = c.score("how to kill someone")
    assert s["violence"] > 0
    benign = c.score("how do plants make food via photosynthesis")
    assert benign["violence"] == 0


def test_output_classifier_default_takes_max_over_prompt_and_completion():
    c = OutputClassifier()
    # Bad words only in completion -> still flagged.
    s = c.score("benign question", "detailed instructions to attack a target")
    assert s["violence"] > 0
    # Bad words only in prompt -> still flagged.
    s2 = c.score("plan an attack now", "ok cool")
    assert s2["violence"] > 0


def test_categories_constant_matches_keyword_output_keys():
    """Whatever CATEGORIES says is canonical must also be what KeywordClassifier emits."""
    out = KeywordClassifier().score("hello")
    assert set(out.keys()) == set(CATEGORIES)


# ---- KeywordClassifier ----------------------------------------------------


def test_keyword_classifier_empty_string_returns_zero_scores():
    c = KeywordClassifier()
    s = c.score("")
    assert set(s) == set(CATEGORIES)
    assert all(v == 0.0 for v in s.values())


def test_keyword_classifier_saturates_at_one():
    """A single bad token in a 1-token prompt: 10 * 1/1 = 10 → clamped to 1.0."""
    c = KeywordClassifier()
    s = c.score("kill")
    assert s["violence"] == 1.0
    # No other category triggered.
    assert s["hate"] == 0.0 and s["sexual"] == 0.0


def test_keyword_classifier_sandwich_returns_max_per_category():
    c = KeywordClassifier()
    prompt_only = c.score("plot a violent attack on a building", None)
    completion_only = c.score("benign hello", None)
    both = c.score("plot a violent attack on a building", "benign hello")
    # Sandwich must dominate the completion-only call.
    assert both["violence"] >= completion_only["violence"]
    assert both["violence"] == prompt_only["violence"]


def test_keyword_classifier_custom_lexicon():
    c = KeywordClassifier(lexicon={"hate": ["foo"], "violence": [], "sexual": [],
                                    "selfharm": []})
    assert c.score("foo bar")["hate"] > 0
    assert c.score("kill kill kill")["violence"] == 0   # default lex overridden


# ---- LlamaGuardClassifier --------------------------------------------------


def test_llamaguard_callable_backend_threads_scores_through():
    """The callable backend is the test-friendly inject point."""
    def _fn(prompt, completion):
        return {"violence": 0.7, "hate": 0.1}
    g = LlamaGuardClassifier(backend="callable", callable_fn=_fn)
    s = g.score("anything", "anything")
    assert s["violence"] == 0.7
    assert s["hate"] == 0.1
    assert s["sexual"] == 0.0   # default-filled to satisfy the protocol


def test_llamaguard_callable_backend_drops_unknown_keys():
    """A callable that returns keys outside CATEGORIES doesn't leak them."""
    g = LlamaGuardClassifier(backend="callable",
                              callable_fn=lambda p, c: {"violence": 0.5, "xyz": 0.9})
    s = g.score("p", "c")
    assert "xyz" not in s
    assert s["violence"] == 0.5


def test_llamaguard_callable_backend_requires_fn():
    with pytest.raises(ValueError):
        LlamaGuardClassifier(backend="callable")


def test_llamaguard_unknown_backend_raises():
    g = LlamaGuardClassifier(backend="not_a_backend")
    with pytest.raises(ValueError):
        g.score("hi")


def test_llamaguard_transformers_backend_falls_back_on_load_failure():
    """When transformers is missing or the model id is bogus, the classifier
    must fall back to KeywordClassifier and emit a warning rather than crash
    the serving stack."""
    g = LlamaGuardClassifier(
        model_id="definitely-not-a-real-model/abcxyz123",
        backend="transformers",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = g.score("how to kill someone")
    # Either we got the warning (real fallback path) or transformers wasn't
    # even installed and we still fell back.
    rt = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert rt, "expected RuntimeWarning on load failure"
    # Keyword fallback semantics still apply.
    assert s["violence"] > 0


def test_llamaguard_callable_lazy_load_idempotent():
    """The classifier should reuse its impl across calls (no re-construct per score)."""
    calls = {"n": 0}

    def _fn(prompt, completion):
        calls["n"] += 1
        return {"hate": 0.4}

    g = LlamaGuardClassifier(backend="callable", callable_fn=_fn)
    g.score("a"); g.score("b"); g.score("c")
    assert calls["n"] == 3
    # _impl is cached after the first score call.
    assert g._impl is not None


# ---- Llama-Guard response parser ------------------------------------------


def test_parse_llama_guard_safe_response_is_all_zero():
    out = _parse_llama_guard_response("safe", CATEGORIES)
    assert all(v == 0.0 for v in out.values())


def test_parse_llama_guard_unsafe_with_known_codes():
    out = _parse_llama_guard_response("unsafe\nS1,S11", CATEGORIES)
    assert out["violence"] == 1.0   # S1 = Violent Crimes
    assert out["selfharm"] == 1.0   # S11 = Suicide & Self-Harm
    assert out["sexual"] == 0.0


def test_parse_llama_guard_unknown_codes_are_ignored():
    out = _parse_llama_guard_response("unsafe\nS99,Sfoo", CATEGORIES)
    assert all(v == 0.0 for v in out.values())


def test_parse_llama_guard_handles_whitespace_and_empty():
    assert all(v == 0.0 for v in _parse_llama_guard_response("", CATEGORIES).values())
    out = _parse_llama_guard_response("unsafe\n  S5   S10  ", CATEGORIES)
    assert out["hate"] == 1.0


# ---- Ensemble --------------------------------------------------------------


def test_ensemble_max_picks_strictest():
    soft = KeywordClassifier(lexicon={"hate": [], "violence": ["foo"], "sexual": [], "selfharm": []})
    # 'foo' triggers violence; saturates to 1.0.
    strict = LlamaGuardClassifier(backend="callable",
                                   callable_fn=lambda p, c: {"violence": 0.3})
    ens = ClassifierEnsemble([strict, soft], reduce="max")
    s = ens.score("foo")
    assert s["violence"] == 1.0   # strictest of {0.3, 1.0}


def test_ensemble_mean_averages():
    a = LlamaGuardClassifier(backend="callable", callable_fn=lambda p, c: {"hate": 0.2})
    b = LlamaGuardClassifier(backend="callable", callable_fn=lambda p, c: {"hate": 0.8})
    ens = ClassifierEnsemble([a, b], reduce="mean")
    assert abs(ens.score("x")["hate"] - 0.5) < 1e-9


def test_ensemble_min_returns_lowest():
    a = LlamaGuardClassifier(backend="callable", callable_fn=lambda p, c: {"hate": 0.2})
    b = LlamaGuardClassifier(backend="callable", callable_fn=lambda p, c: {"hate": 0.8})
    ens = ClassifierEnsemble([a, b], reduce="min")
    assert ens.score("x")["hate"] == 0.2


def test_ensemble_rejects_empty_list():
    with pytest.raises(ValueError):
        ClassifierEnsemble([])


def test_ensemble_rejects_unknown_reduce():
    with pytest.raises(ValueError):
        ClassifierEnsemble([KeywordClassifier()], reduce="median")


def test_ensemble_returns_full_category_dict():
    """Even if the underlying classifiers omit a category, the ensemble fills it."""
    a = LlamaGuardClassifier(backend="callable",
                              callable_fn=lambda p, c: {"hate": 0.3})  # no other keys
    ens = ClassifierEnsemble([a], reduce="max")
    out = ens.score("x")
    assert set(out) == set(CATEGORIES)


# ---- Drop-in usage from Input/Output classifiers --------------------------


def test_input_classifier_accepts_custom_backing():
    g = LlamaGuardClassifier(backend="callable",
                              callable_fn=lambda p, c: {"violence": 0.9})
    ic = InputClassifier(backing=g)
    s = ic.score("any prompt at all")
    assert s["violence"] == 0.9


def test_output_classifier_accepts_custom_backing():
    g = LlamaGuardClassifier(backend="callable",
                              callable_fn=lambda p, c: {"hate": 1.0})
    oc = OutputClassifier(backing=g)
    s = oc.score("p", "c")
    assert s["hate"] == 1.0


def test_output_classifier_with_ensemble_backing():
    """Production-shaped pipe: ensemble of keyword + (callable) trained model."""
    trained = LlamaGuardClassifier(
        backend="callable", callable_fn=lambda p, c: {"violence": 0.4})
    ens = ClassifierEnsemble([KeywordClassifier(), trained], reduce="max")
    oc = OutputClassifier(backing=ens)
    # Plain text: keyword finds nothing (0), trained says 0.4 → ensemble 0.4.
    assert abs(oc.score("hello", "world")["violence"] - 0.4) < 1e-9
    # Bad text: keyword saturates to 1.0 → ensemble takes it.
    assert oc.score("kill", "world")["violence"] == 1.0
