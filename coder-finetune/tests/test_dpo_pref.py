import sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cf_pref import pairs as pref


# ---------- preference dataset ----------

def test_builtin_pairs_have_prompt_chosen_rejected_text():
    ds = pref.load_builtin(repeat=1, use_chat=False)
    assert len(ds) == len(pref.BUILTIN_PREFERENCES)
    ex = ds[0]
    assert {"prompt", "chosen", "rejected"} <= set(ex)
    # text format -> strings, chosen carries a fenced python block
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


def test_chosen_solutions_actually_pass_their_verifier():
    # Cross-check against the RLVR verifier: every 'chosen' answer must pass the
    # matching unit tests, and every 'rejected' must fail — otherwise the
    # preference label is wrong and DPO would learn the wrong margin.
    from cf_rl.reward import code_unit_test_reward
    from cf_rl.prompts import BUILTIN_TASKS

    tests_by_entry = {entry: (stub, test) for _, entry, stub, test in BUILTIN_TASKS}
    for instr, chosen, rejected in pref.BUILTIN_PREFERENCES:
        # derive entry point from "def <name>("
        entry = chosen.split("def ", 1)[1].split("(", 1)[0].strip()
        if entry not in tests_by_entry:
            continue
        stub, test = tests_by_entry[entry]
        rc = code_unit_test_reward(
            completions=["```python\n" + chosen + "```"],
            test=[test], entry_point=[entry], prompt_code=[stub],
        )
        rr = code_unit_test_reward(
            completions=["```python\n" + rejected + "```"],
            test=[test], entry_point=[entry], prompt_code=[stub],
        )
        assert rc == [1.0], f"chosen for {entry} should pass: {chosen!r}"
        assert rr == [0.0], f"rejected for {entry} should fail: {rejected!r}"


# ---------- trainer config plumbing (no model load) ----------

def test_make_config_filters_unknown_kwargs():
    from cf_pref.dpo_train import _make_config
    from trl import DPOConfig

    cfg = _make_config(DPOConfig, {
        "output_dir": "out/x", "beta": 0.2,
        # Explicit bf16=False: TRL's DPOConfig auto-enables bf16 when left at
        # its None default, which transformers then rejects on a CPU-only host
        # ("Your setup doesn't support bf16/gpu"). The real builder
        # (build_pref_trainer) already passes an explicit hardware-gated bool;
        # mirror that here so the unit test is hermetic on GPU-less CI.
        "bf16": False,
        "this_field_does_not_exist": 123,   # must be dropped, not crash
    })
    assert cfg.beta == 0.2


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


def test_simpo_objective_selects_simpo_loss_type_if_supported():
    from cf_pref.dpo_train import _make_config
    from trl import DPOConfig

    cfg = _make_config(DPOConfig, {
        "output_dir": "out/x", "beta": 0.2, "bf16": False, "loss_type": "simpo",
    })
    # Don't silently pass if the field vanishes — a missing loss_type means our
    # simpo wiring assumption is broken and the test must say so.
    assert hasattr(cfg, "loss_type"), "DPOConfig lost loss_type; simpo wiring needs review"
    lt = cfg.loss_type if isinstance(cfg.loss_type, (list, tuple)) else [cfg.loss_type]
    assert "simpo" in lt


def _min_pref_cfg(objective: str) -> dict:
    # Minimal config exercising build_pref_trainer's dispatch without a GPU.
    # dtype is float32 so neither bf16 nor fp16 is requested on CPU-only CI.
    return {
        "out_dir": "out/x", "seed": 0, "method": "lora",
        "model": {"dtype": "float32"},
        "train": {"epochs": 1, "batch_size": 1, "grad_accum": 1, "lr": 1e-6,
                  "warmup_ratio": 0.0, "weight_decay": 0.0, "grad_clip": 1.0,
                  "log_every": 1, "save_every": 1, "max_seq_len": 64},
        "pref": {"objective": objective},
    }


def test_simpo_objective_routes_through_dpo_trainer_reference_free(monkeypatch):
    # End-to-end through build_pref_trainer's simpo branch: capture the args
    # handed to DPOTrainer so we verify the *wiring*, not just a standalone
    # config object. A real model load is unnecessary — the branch logic
    # (loss_type default, ref-free flags, ref_model=None) is what can regress.
    import trl
    from cf_pref import dpo_train

    captured = {}

    def fake_trainer(*, model, ref_model, args, train_dataset, processing_class):
        captured.update(model=model, ref_model=ref_model, args=args,
                        train_dataset=train_dataset, processing_class=processing_class)
        return "TRAINER"

    monkeypatch.setattr(trl, "DPOTrainer", fake_trainer)

    sentinel_model, sentinel_tok, sentinel_ds = object(), object(), object()
    out = dpo_train.build_pref_trainer(
        sentinel_model, sentinel_tok, sentinel_ds, _min_pref_cfg("simpo"),
    )

    assert out == "TRAINER"
    # Reference-free: no separate reference model is loaded.
    assert captured["ref_model"] is None
    assert captured["model"] is sentinel_model
    assert captured["processing_class"] is sentinel_tok
    # SimPO loss is selected by default for the simpo objective...
    args = captured["args"]
    assert "simpo" in (args.loss_type if isinstance(args.loss_type, (list, tuple)) else [args.loss_type])
    # ...and the reference-sync schedule is disabled (no ref model to sync to).
    assert getattr(args, "ref_model_sync_steps", None) is None
    assert args.beta == 0.1


def test_dpo_objective_keeps_default_sigmoid_loss(monkeypatch):
    # Guardrail: the default 'dpo' objective must NOT inherit simpo settings.
    import trl
    from cf_pref import dpo_train

    captured = {}
    monkeypatch.setattr(trl, "DPOTrainer",
                        lambda **kw: captured.update(kw) or "TRAINER")

    dpo_train.build_pref_trainer(object(), object(), object(), _min_pref_cfg("dpo"))
    args = captured["args"]
    lt = args.loss_type if isinstance(args.loss_type, (list, tuple)) else [args.loss_type]
    assert "sigmoid" in lt
    assert "simpo" not in lt


def test_orpo_missing_raises_actionable_error():
    # On TRL 1.x (no ORPOTrainer) the orpo path must raise a clear SystemExit,
    # never an opaque ImportError. Only exercises the dispatch branch.
    from cf_pref import dpo_train

    has_orpo = True
    try:
        from trl import ORPOTrainer  # noqa: F401
    except ImportError:
        has_orpo = False

    cfg = {
        "out_dir": "out/x", "seed": 0, "method": "full",
        "model": {"dtype": "bfloat16"},
        "train": {"epochs": 1, "batch_size": 1, "grad_accum": 1, "lr": 1e-6,
                  "warmup_ratio": 0.0, "weight_decay": 0.0, "grad_clip": 1.0,
                  "log_every": 1, "save_every": 1, "max_seq_len": 64,
                  "gradient_checkpointing": False},
        "pref": {"objective": "orpo"},
    }
    if not has_orpo:
        with pytest.raises(SystemExit):
            dpo_train.build_pref_trainer(object(), object(), [], cfg)
    else:
        pytest.skip("this TRL ships ORPOTrainer; missing-path not exercised")
