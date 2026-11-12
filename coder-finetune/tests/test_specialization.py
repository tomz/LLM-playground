"""CPU-only checks of specialization budget, split and evidence contracts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
from eval import specialization as sp


@pytest.mark.parametrize("overrides", [
    {"train_steps": 201}, {"base_steps": 0}, {"batch_size": 33}, {"batch_size": True},
    {"seeds": ()}, {"seeds": (1, 1)}, {"seeds": tuple(range(11))},
    {"branch_seconds": 31}, {"branch_seconds": float("nan")},
    {"learning_rate": float("inf")},
])
def test_budget_rejects_unbounded_requests(overrides):
    with pytest.raises(ValueError):
        sp.Budget(**overrides)


@pytest.mark.parametrize("collision", ["id", "prompt"])
def test_leakage_rejected_before_output(tmp_path, collision):
    splits = sp.dataset()
    original = splits["train"][0]
    splits["heldout"][0] = sp.Example(original.task_id if collision == "id" else "new",
                                     original.prompt if collision == "prompt" else "even:999", 0)
    with pytest.raises(ValueError, match="leakage"):
        sp.run_comparison(tmp_path / "run", sp.Budget(seeds=(0,)), splits=splits)
    assert not (tmp_path / "run").exists()


def test_real_training_baseline_lora_and_checkpoint_roundtrip(tmp_path):
    budget = sp.Budget(seeds=(0,), base_steps=5, train_steps=8)
    result = sp.run_comparison(tmp_path / "run", budget)
    base, sft, lora = result["branches"]
    assert all(b["status"] == "ok" for b in result["branches"])
    assert len({b["initial_sha256"] for b in result["branches"]}) == 1
    assert base["initial_sha256"] == base["final_sha256"]
    assert sft["initial_sha256"] != sft["final_sha256"]
    assert lora["initial_sha256"] != lora["final_sha256"]
    assert lora["trainable_parameters"] < sft["trainable_parameters"]
    for branch in result["branches"]:
        assert len(branch["attempts"]) == 40
        assert branch["cost"]["completed_steps"] == (0 if branch["arm"] == "baseline" else 8)
        assert branch["cost"]["api_calls"] == 0
        for split in ("heldout", "regression"):
            m = branch["metrics"][split]
            assert m["n"] == m["calibration_n"] == 20
            assert 0 <= m["accuracy"] <= 1 and 0 <= m["brier"] <= 1
    assert len((tmp_path / "run/branches.jsonl").read_text().splitlines()) == 3
    model_dir = tmp_path / "run/models/seed-0/baseline"
    loaded = sp.load_model(model_dir)
    assert sp.state_hash(loaded) == base["final_sha256"]
    second = sp.run_comparison(tmp_path / "local", budget, model_dir=model_dir)
    assert second["base_preparation"][0]["training"] == []
    assert second["branches"][0]["initial_sha256"] == base["final_sha256"]
    merged_lora = sp.load_model(tmp_path / "run/models/seed-0/lora")
    frozen_base = loaded.state_dict()
    for key, tensor in merged_lora.state_dict().items():
        if key != "head.weight":
            assert torch.equal(tensor, frozen_base[key]), key


def test_all_failed_training_branches_are_retained(tmp_path, monkeypatch):
    real_train = sp.train
    def failing_train(model, examples, steps, budget, seed, deadline, ledger):
        if examples[0].task_id.startswith("even/"):
            ledger["training"].append({"status": "started", "step": 0, "examples": budget.batch_size})
            raise RuntimeError("injected optimizer failure")
        return real_train(model, examples, steps, budget, seed, deadline, ledger)
    monkeypatch.setattr(sp, "train", failing_train)
    result = sp.run_comparison(tmp_path / "run", sp.Budget(seeds=(0, 1), base_steps=1, train_steps=1))
    assert len(result["branches"]) == 6
    for branch in result["branches"]:
        if branch["arm"] == "baseline":
            assert branch["status"] == "ok"
        else:
            assert branch["status"] == "failed"
            assert len(branch["attempts"]) == 40
            assert all(a["status"] == "failed" for a in branch["attempts"])
            assert branch["cost"]["completed_steps"] == 0
            assert branch["cost"]["attempted_steps"] == 1
            assert branch["metrics"]["heldout"]["accuracy"] == 0
            assert branch["metrics"]["heldout"]["calibration_n"] == 0
            assert branch["metrics"]["heldout"]["brier"] is None
    assert result["aggregate"]["sft"]["heldout"]["failed_branches"] == 2


def test_timeout_accounts_for_every_planned_attempt():
    attempts = sp.evaluate(sp.TinyClassifier(), sp.dataset(), time.monotonic() - 1)
    assert len(attempts) == 40 and all(a["status"] == "failed" for a in attempts)
    assert all("TimeoutError" in a["error"] for a in attempts)
    ledger = {"training": []}
    with pytest.raises(TimeoutError):
        sp.train(sp.TinyClassifier(), sp.dataset()["train"], 1, sp.Budget(), 0, 0, ledger)
    assert not ledger["training"]


def test_calibration_failed_denominator_is_explicit():
    attempts = [dict(status="ok", probability=.8, label=1, confidence=.8, correct=True, seconds=.01),
                dict(status="failed", correct=False)]
    measured = sp.metrics(attempts)
    assert measured["accuracy"] == .5 and measured["failed"] == 1
    assert measured["calibration_n"] == 1
    assert measured["brier"] == pytest.approx(.04)
    assert measured["ece_10_bins"] == pytest.approx(.2)


def test_local_only_checkpoint_integrity(tmp_path):
    with pytest.raises(ValueError, match="existing local"):
        sp.load_model(tmp_path / "missing-hub-model")
    sp.save_model(sp.TinyClassifier(), tmp_path / "model")
    weights = tmp_path / "model/model.pt"
    weights.write_bytes(weights.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        sp.load_model(tmp_path / "model")


def test_no_existing_artifact_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        sp.run_comparison(tmp_path, sp.Budget(seeds=(0,), base_steps=1, train_steps=1))
