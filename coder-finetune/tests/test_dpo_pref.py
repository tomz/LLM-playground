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
        "this_field_does_not_exist": 123,   # must be dropped, not crash
    })
    assert cfg.beta == 0.2


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
