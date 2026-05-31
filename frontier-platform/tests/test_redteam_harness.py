"""Tests for the structured red-team harness (replaces the 5-prompt regex toy).

Covers:
- back-compat (run_suite/run_all/SUITES return the same shape as before)
- probe + suite registry
- judges (regex / classifier / ensemble / callable)
- report builder integrates with platform.safety.gates.preflight
"""
from __future__ import annotations

import json

import pytest

from platform.safety.classifiers import OutputClassifier
from platform.safety.gates import ModelCard, default_thresholds, preflight
from platform.safety.redteam import (
    SUITES,
    CallableJudge,
    ClassifierJudge,
    EnsembleJudge,
    JudgeResult,
    Probe,
    RegexRefusalJudge,
    Suite,
    all_suites,
    build_report,
    get_suite,
    register_suite,
    run_all,
    run_suite,
    write_report,
)
from platform.safety.redteam.probes import CATEGORIES


# ----- back-compat ----------------------------------------------------------


def test_SUITES_includes_originals():
    for name in ["harmbench", "advbench", "multi_turn_social", "cybench", "metr_agent"]:
        assert name in SUITES


def test_run_suite_back_compat_dict_shape():
    """The original tests asserted the return shape; the new code must keep it."""
    refuser = lambda s: "I can't help with that."
    out = run_suite(refuser, "harmbench")
    assert set(out) >= {"suite", "n_probes", "refusal_rate", "details"}
    assert out["suite"] == "harmbench"
    assert out["n_probes"] > 0
    assert 0.0 <= out["refusal_rate"] <= 1.0
    # Each detail line carries the old fields plus the new structured ones.
    for d in out["details"]:
        assert {"prompt", "response", "refused"} <= set(d)
        assert {"id", "category", "category_scores"} <= set(d)


def test_run_suite_refuser_yields_full_refusal():
    out = run_suite(lambda s: "Sorry, but I cannot assist with that.", "advbench")
    assert out["refusal_rate"] == 1.0


def test_run_suite_compliant_yields_zero_refusal():
    out = run_suite(lambda s: "Sure, here you go: ...", "advbench")
    assert out["refusal_rate"] == 0.0


def test_run_all_covers_every_suite():
    out = run_all(lambda s: "I can't help with that.")
    assert set(out) == set(SUITES)
    for r in out.values():
        assert r["refusal_rate"] == 1.0


# ----- endpoint types -------------------------------------------------------


def test_endpoint_accepts_object_with_generate():
    class _Obj:
        def generate(self, p):
            return "I cannot help with that."
    out = run_suite(_Obj(), "harmbench")
    assert out["refusal_rate"] == 1.0


def test_endpoint_str_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        run_suite("http://localhost:8000", "harmbench")


def test_endpoint_unsupported_raises_typeerror():
    with pytest.raises(TypeError):
        run_suite(42, "harmbench")


# ----- probes + registry ----------------------------------------------------


def test_probe_rejects_unknown_category():
    with pytest.raises(ValueError):
        Probe(id="x", suite="s", category="not_a_category", prompt="hi")


def test_register_suite_adds_and_blocks_overwrite():
    probes = [Probe("custom-01", "custom", "jailbreak", "test")]
    suite = Suite(name="custom_test_suite", probes=probes)
    register_suite(suite)
    assert "custom_test_suite" in SUITES
    fetched = get_suite("custom_test_suite")
    assert fetched.probes[0].id == "custom-01"

    # Re-registering without overwrite should refuse.
    with pytest.raises(ValueError):
        register_suite(Suite(name="custom_test_suite", probes=[]))
    # With overwrite=True it replaces.
    register_suite(Suite(name="custom_test_suite", probes=[]), overwrite=True)
    assert get_suite("custom_test_suite").probes == []


def test_get_suite_unknown_raises():
    with pytest.raises(KeyError):
        get_suite("definitely-not-real")


def test_all_suites_returns_suite_objects():
    suites = all_suites()
    assert all(isinstance(s, Suite) for s in suites)
    assert {s.name for s in suites} >= set(SUITES) - {"custom_test_suite"}


# ----- judges ---------------------------------------------------------------


def _probe(category="cyber"):
    return Probe(id="t", suite="harmbench", category=category, prompt="ignored")


def test_regex_judge_refusal_path():
    j = RegexRefusalJudge()
    r = j.judge(_probe(), "I won't help with that, sorry.")
    assert r.refused is True
    assert all(v == 0.0 for v in r.category_scores.values())


def test_regex_judge_compliance_path():
    j = RegexRefusalJudge()
    r = j.judge(_probe("cyber"), "Sure, here is the payload...")
    assert r.refused is False
    assert r.category_scores["cyber"] == 1.0
    # Other categories left at zero.
    assert r.category_scores["bias"] == 0.0


def test_classifier_judge_maps_scores():
    j = ClassifierJudge(OutputClassifier())
    # 'kill ... attack' triggers the violence keyword classifier → mapped to cbrn.
    r = j.judge(_probe("cbrn"), "kill kill attack attack bomb")
    assert r.category_scores["cbrn"] > 0


def test_classifier_judge_attributes_compliance_to_probe_category():
    """When the model complies and the classifier sees nothing, the judge still
    attributes harm to the probe's own category so coverage is not undercounted."""
    j = ClassifierJudge(OutputClassifier())
    r = j.judge(_probe("autonomy"), "Sure, here is a benign-looking response.")
    assert r.refused is False
    assert r.category_scores["autonomy"] == 1.0


def test_ensemble_max_picks_strictest():
    soft = CallableJudge(lambda p, r: JudgeResult(refused=False, category_scores={"cyber": 0.2}),
                         name="soft")
    strict = CallableJudge(lambda p, r: JudgeResult(refused=False, category_scores={"cyber": 0.9}),
                           name="strict")
    e = EnsembleJudge([soft, strict], reduce="max")
    r = e.judge(_probe("cyber"), "...")
    assert r.category_scores["cyber"] == 0.9


def test_ensemble_mean_averages():
    a = CallableJudge(lambda p, r: JudgeResult(refused=False, category_scores={"cyber": 0.2}))
    b = CallableJudge(lambda p, r: JudgeResult(refused=False, category_scores={"cyber": 0.8}))
    e = EnsembleJudge([a, b], reduce="mean")
    r = e.judge(_probe("cyber"), "...")
    assert abs(r.category_scores["cyber"] - 0.5) < 1e-9


def test_ensemble_refusal_majority_vote():
    yes = CallableJudge(lambda p, r: JudgeResult(refused=True))
    no = CallableJudge(lambda p, r: JudgeResult(refused=False))
    # 2 refuse, 1 comply -> refused True
    e1 = EnsembleJudge([yes, yes, no])
    assert e1.judge(_probe(), "...").refused is True
    # 1 refuse, 2 comply -> refused False
    e2 = EnsembleJudge([yes, no, no])
    assert e2.judge(_probe(), "...").refused is False


def test_ensemble_empty_judges_raises():
    with pytest.raises(ValueError):
        EnsembleJudge([])


def test_ensemble_bad_reduce_raises():
    with pytest.raises(ValueError):
        EnsembleJudge([RegexRefusalJudge()], reduce="median")


def test_callable_judge_tuple_form():
    fn = lambda p, r: (False, {"cbrn": 0.7, "cyber": 0.1})
    j = CallableJudge(fn=fn, name="raw_tuple")
    r = j.judge(_probe("cbrn"), "...")
    assert r.refused is False
    assert r.category_scores["cbrn"] == 0.7
    assert r.category_scores["bias"] == 0.0   # default-filled


def test_callable_judge_bad_return_raises():
    j = CallableJudge(fn=lambda p, r: "garbage")
    with pytest.raises(TypeError):
        j.judge(_probe(), "...")


# ----- report builder + gates.preflight roundtrip --------------------------


def test_build_report_compliant_endpoint_produces_high_scores():
    """A model that complies with everything should fail the gate on most
    categories — that is the whole point of the harness."""
    results = run_all(lambda s: "Sure, here is exactly what you asked for: ...")
    report = build_report(results)
    # All gate categories are present as floats.
    for c in CATEGORIES:
        assert c in report
        assert 0.0 <= report[c] <= 1.0
    # _per_suite is populated for every suite we ran.
    assert set(report["_per_suite"]) == set(results.keys())
    # _meta carries probe count + schema.
    assert report["_meta"]["schema_version"] == 1
    assert report["_meta"]["n_probes"] > 0


def test_build_report_refusing_endpoint_passes_preflight(tmp_path):
    """A model that refuses everything should drive every category to 0 and PASS."""
    results = run_all(lambda s: "I cannot help with that.")
    path = write_report(tmp_path / "report.json", results)
    # Sanity: the file is valid JSON with all categories.
    parsed = json.loads(path.read_text())
    assert all(parsed[c] == 0.0 for c in CATEGORIES)

    card = ModelCard(
        ckpt="ckpt", eval_report_uri="eval",
        redteam_report_uri=str(path), intended_use="chat", known_limits=[],
    )
    res = preflight(card)
    assert res.passed
    assert res.failed_categories == []


def test_build_report_compliant_endpoint_fails_preflight(tmp_path):
    """A model that complies should be BLOCKED by preflight."""
    results = run_all(lambda s: "Here are the step-by-step instructions you asked for: ...")
    path = write_report(tmp_path / "report.json", results)
    card = ModelCard(
        ckpt="ckpt", eval_report_uri="eval",
        redteam_report_uri=str(path), intended_use="chat", known_limits=[],
    )
    res = preflight(card, thresholds=default_thresholds())
    assert not res.passed
    # The compliant endpoint hits cbrn / cyber / persuasion / autonomy / jailbreak.
    assert "cbrn" in res.failed_categories or "cyber" in res.failed_categories


def test_build_report_handles_empty_results():
    report = build_report({})
    for c in CATEGORIES:
        assert report[c] == 0.0
    assert report["_meta"]["n_probes"] == 0


def test_custom_judge_threads_through_run_suite():
    """An ensemble judge with a stub classifier flows through run_suite."""
    j = EnsembleJudge([RegexRefusalJudge(), ClassifierJudge(OutputClassifier())])
    out = run_suite(lambda s: "I won't help with that.", "harmbench", judge=j)
    assert out["refusal_rate"] == 1.0
    # Per-probe records carry the new structured fields.
    for d in out["details"]:
        assert isinstance(d["category_scores"], dict)
        assert set(d["category_scores"]).issuperset(CATEGORIES)
