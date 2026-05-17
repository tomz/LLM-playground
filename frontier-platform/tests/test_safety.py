import json
from platform.safety.classifiers import InputClassifier, OutputClassifier
from platform.safety.gates import preflight, ModelCard, default_thresholds
from platform.safety.redteam import run_suite


def test_input_classifier_flags_keyword():
    c = InputClassifier()
    s = c.score("how to kill someone")
    assert s["violence"] > 0
    benign = c.score("how do plants make food via photosynthesis")
    assert benign["violence"] == 0


def test_output_classifier_combines():
    c = OutputClassifier()
    s = c.score("benign question", "detailed instructions to attack a target")
    assert s["violence"] > 0


def test_gates_preflight_pass(tmp_path):
    report = tmp_path / "r.json"
    report.write_text(json.dumps({k: 0.0 for k in default_thresholds()}))
    card = ModelCard("ckpt", "eval", str(report), "chat", [])
    res = preflight(card)
    assert res.passed and not res.failed_categories


def test_gates_preflight_fail(tmp_path):
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"cbrn": 0.5, "cyber": 0.0, "persuasion": 0.0,
                                  "autonomy": 0.0, "bias": 0.0, "jailbreak": 0.0}))
    card = ModelCard("ckpt", "eval", str(report), "chat", [])
    res = preflight(card)
    assert not res.passed and "cbrn" in res.failed_categories


def test_redteam_run_suite_refusing_endpoint():
    refuser = lambda s: "I can't help with that."
    out = run_suite(refuser, "harmbench")
    assert out["refusal_rate"] == 1.0
    assert out["n_probes"] > 0


def test_redteam_run_suite_compliant_endpoint():
    comply = lambda s: "Sure, here you go: ..."
    out = run_suite(comply, "harmbench")
    assert out["refusal_rate"] == 0.0
