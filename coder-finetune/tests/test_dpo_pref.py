"""Preference data and the strict, sigmoid-only TRL DPO training contract."""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cf_pref import pairs as pref


# ---------- preference dataset ----------

def test_builtin_pairs_have_prompt_chosen_rejected_text():
    ds = pref.load_builtin(repeat=1, use_chat=False)
    assert len(ds) == len(pref.BUILTIN_PREFERENCES)
    ex = ds[0]
    assert {"prompt", "chosen", "rejected"} <= set(ex)
    assert isinstance(ex["prompt"], str)
    assert isinstance(ex["chosen"], str)
    assert "```python" in ex["chosen"]
    assert ex["chosen"] != ex["rejected"]


def test_builtin_pairs_chatml_shape():
    ds = pref.load_builtin(repeat=1, use_chat=True)
    ex = ds[0]
    assert isinstance(ex["prompt"], list)
    assert ex["prompt"][0]["role"] == "system"
    assert isinstance(ex["chosen"], list)
    assert ex["chosen"][0]["role"] == "assistant"


def test_repeat_scales_dataset():
    ds = pref.load_builtin(repeat=10, use_chat=False)
    assert len(ds) == 10 * len(pref.BUILTIN_PREFERENCES)


def test_load_dispatch_builtin():
    ds = pref.load({"source": "builtin", "repeat": 2, "format": "text"})
    assert len(ds) == 2 * len(pref.BUILTIN_PREFERENCES)


def test_load_unknown_source_raises():
    with pytest.raises(ValueError):
        pref.load({"source": "nope"})


def test_binary_feedback_adapter_derives_kto_rows():
    from cf_pref import binary

    ds = binary.load({"source": "builtin", "repeat": 1, "format": "text"})
    assert len(ds) == 2 * len(pref.BUILTIN_PREFERENCES)
    assert {"prompt", "completion", "label"} <= set(ds[0])
    assert {ex["label"] for ex in ds} == {True, False}


def test_trusted_preference_fixtures_have_correct_labels():
    """Host execution is limited to these reviewed, static repository fixtures."""
    from cf_rl.prompts import BUILTIN_TASKS
    from eval.run_humaneval import build_program
    from eval.trusted_subprocess import trusted_run_one

    tests_by_entry = {entry: (stub, test) for _, entry, stub, test in BUILTIN_TASKS}
    checked = set()
    for _, chosen, rejected in pref.BUILTIN_PREFERENCES:
        entry = chosen.split("def ", 1)[1].split("(", 1)[0].strip()
        if entry not in tests_by_entry:
            continue
        stub, test = tests_by_entry[entry]
        passed, message = trusted_run_one(build_program(stub, chosen, test, entry), mp_mode="spawn")
        assert passed, f"trusted chosen fixture for {entry} failed: {message}"
        passed, message = trusted_run_one(build_program(stub, rejected, test, entry), mp_mode="spawn")
        assert not passed and "AssertionError" in message, (entry, message)
        checked.add(entry)
    assert checked == set(tests_by_entry), "every built-in verifier needs checked preference labels"


# ---------- trainer config plumbing (no model load) ----------

def test_make_config_rejects_unknown_fields():
    from cf_pref.dpo_train import _make_config
    from trl import DPOConfig

    with pytest.raises(ValueError, match="this_field_does_not_exist"):
        _make_config(DPOConfig, {"output_dir": "out/x", "beta": 0.2, "bf16": False,
                                 "this_field_does_not_exist": 123})


@pytest.mark.parametrize("loss_type", ["simpo", "hinge", ["simpo"]])
def test_make_config_rejects_untested_loss_types(loss_type):
    from cf_pref.dpo_train import _make_config
    from trl import DPOConfig

    with pytest.raises(ValueError, match="only sigmoid"):
        _make_config(DPOConfig, {"output_dir": "out/x", "bf16": False, "loss_type": loss_type})


def _min_pref_cfg(objective: str) -> dict:
    return {
        "out_dir": "out/x", "seed": 0, "method": "lora",
        "model": {"dtype": "float32"},
        "train": {"epochs": 1, "batch_size": 1, "grad_accum": 1, "lr": 1e-6,
                  "warmup_ratio": 0.0, "weight_decay": 0.0, "grad_clip": 1.0,
                  "log_every": 1, "save_every": 1, "max_seq_len": 64, "use_cpu": True},
        "pref": {"objective": objective},
    }


@pytest.mark.parametrize("objective", ["simpo", "orpo", "kto", "typo"])
def test_unsupported_objectives_fail_before_trainer_construction(monkeypatch, objective):
    import trl
    from cf_pref import dpo_train

    def forbidden_trainer(**kwargs):
        pytest.fail("unsupported objective reached DPOTrainer")

    monkeypatch.setattr(trl, "DPOTrainer", forbidden_trainer)
    with pytest.raises(ValueError, match="unsupported.*use dpo"):
        dpo_train.build_pref_trainer(object(), object(), [], _min_pref_cfg(objective))


@pytest.mark.parametrize("option", ["gamma", "reference_free", "ref_model_sync_steps", "beta_typo"])
def test_preference_options_are_not_silently_discarded(option):
    from cf_pref.dpo_train import build_pref_config

    cfg = _min_pref_cfg("dpo")
    cfg["pref"][option] = 0.1
    with pytest.raises(ValueError, match=option):
        build_pref_config(cfg)


def test_dpo_trainer_preserves_inputs_and_sigmoid_objective(monkeypatch):
    import trl
    from cf_pref import dpo_train

    captured = {}
    monkeypatch.setattr(trl, "DPOTrainer", lambda **kwargs: captured.update(kwargs) or "TRAINER")
    model, tokenizer, dataset = object(), object(), object()
    cfg = _min_pref_cfg("dpo")
    cfg["pref"]["beta"] = 0.2
    result = dpo_train.build_pref_trainer(model, tokenizer, dataset, cfg)
    assert result == "TRAINER"
    assert captured["model"] is model and captured["processing_class"] is tokenizer
    assert captured["train_dataset"] is dataset
    # TRL clones the reference or disables PEFT adapters; None is NOT reference-free DPO.
    assert captured["ref_model"] is None
    args = captured["args"]
    loss_types = args.loss_type if isinstance(args.loss_type, (list, tuple)) else [args.loss_type]
    assert loss_types == ["sigmoid"]
    assert args.beta == 0.2 and args.max_length == 64
    assert args.bf16 is False and args.fp16 is False


# ---------- educational losses, not trainer integrations ----------

def test_simpo_loss_decreases_as_normalized_margin_grows():
    from cf_pref.objectives import simpo_loss
    import torch

    lengths = torch.tensor([10.0])
    low_margin = simpo_loss(torch.tensor([-8.0]), torch.tensor([-9.0]), lengths, lengths, beta=2.0, gamma=0.0)
    high_margin = simpo_loss(torch.tensor([-6.0]), torch.tensor([-9.0]), lengths, lengths, beta=2.0, gamma=0.0)
    assert high_margin < low_margin


def test_kto_loss_rewards_desirable_and_undesirable_directions():
    from cf_pref.objectives import kto_loss
    import torch

    ref = torch.zeros(1)
    desirable_bad = kto_loss(torch.tensor([-1.0]), ref, torch.tensor([True]), beta=1.0)
    desirable_good = kto_loss(torch.tensor([1.0]), ref, torch.tensor([True]), beta=1.0)
    undesirable_bad = kto_loss(torch.tensor([1.0]), ref, torch.tensor([False]), beta=1.0)
    undesirable_good = kto_loss(torch.tensor([-1.0]), ref, torch.tensor([False]), beta=1.0)
    assert desirable_good < desirable_bad
    assert undesirable_good < undesirable_bad
