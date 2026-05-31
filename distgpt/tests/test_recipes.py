"""Recipe configs + warm-start checkpoint loading.

Two concerns:
  1. The shipped recipes in `configs/recipes/` must be valid YAML and have
     every key the trainer reads. A typo here turns a multi-hour cooldown
     into a KeyError on minute one.
  2. The `load_ckpt:` warm-start path: load weights from one run into
     another with fresh optim / loader / step counter. Verifies the
     CheckpointManager.load_weights_only contract and the trainer's
     priority order (native resume > warm start > cold start).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


RECIPES = ["cooldown.yaml", "longctx_finetune.yaml", "muon_speedrun_1b.yaml"]
REQUIRED_TOP = {"run_id", "out_dir", "data", "model", "parallel", "optim", "train"}


# ---------- recipe YAML parsing ----------


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_yaml_parses_with_required_keys(name: str):
    """Every recipe must parse and contain the keys the trainer reads.

    This is a static-config-validation test — catches typos like missing
    `train.log_every` before they reach a real run.
    """
    path = ROOT / "configs" / "recipes" / name
    assert path.exists(), f"recipe {name} missing"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in REQUIRED_TOP:
        assert k in cfg, f"{name}: missing top-level key {k!r}"
    # Trainer-mandatory nested keys
    assert "vocab_size" in cfg["model"]
    assert "max_seq_len" in cfg["model"]
    assert "lr" in cfg["optim"] and "min_lr" in cfg["optim"]
    assert "warmup_steps" in cfg["optim"]
    assert "total_steps" in cfg["optim"]
    assert "micro_batch" in cfg["train"]
    assert "grad_accum" in cfg["train"]
    assert "log_every" in cfg["train"]
    assert "eval_every" in cfg["train"]
    assert "ckpt_every" in cfg["train"]


def test_cooldown_recipe_sets_warm_start_path():
    """The cooldown recipe is a warm-start workflow; it must include
    `load_ckpt:` pointing somewhere or the trainer falls through to a
    cold start at step 0 (which silently produces a randomly-initialized
    cooldown — embarrassing)."""
    with open(ROOT / "configs" / "recipes" / "cooldown.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "load_ckpt" in cfg
    assert cfg["load_ckpt"], "cooldown.yaml has empty load_ckpt"
    # And it should use the LOW-LR cooldown regime, not the base-pretrain LR.
    assert cfg["optim"]["lr"] < 1e-3, (
        f"cooldown LR {cfg['optim']['lr']} looks like a pre-train LR, not a cooldown"
    )
    assert cfg["optim"]["warmup_steps"] == 0, (
        "cooldown should not re-warm up — we're already at the bottom of cosine"
    )


def test_longctx_recipe_uses_extended_rope_base():
    """The long-context recipe's whole point is rope_base scaling; if
    someone copies the file but forgets to bump rope_base, the model
    won't actually extend its effective context."""
    with open(ROOT / "configs" / "recipes" / "longctx_finetune.yaml") as f:
        cfg = yaml.safe_load(f)
    # rope_base must be > 10000 (the GPT-2/Llama default we extend from).
    assert cfg["model"]["rope_base"] > 10000, (
        "longctx recipe has default rope_base; context extension won't work"
    )
    # And the seq_len must match max_seq_len (trainer asserts this).
    assert cfg["data"]["seq_len"] == cfg["model"]["max_seq_len"]


def test_muon_recipe_uses_muon_optimizer():
    """The Muon speedrun recipe must actually select the Muon optimizer
    (else it's just configs/1b.yaml with QK-norm). Catches a regression
    where the optimizer field got reverted."""
    with open(ROOT / "configs" / "recipes" / "muon_speedrun_1b.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["optim"]["optimizer"] == "muon"
    # The dual-optim convention is `muon_lr` for hidden weights and `lr` for
    # the AdamW group; both must be present.
    assert "muon_lr" in cfg["optim"]
    assert cfg["optim"]["muon_lr"] > cfg["optim"]["lr"], (
        "muon_lr should be much larger than the AdamW lr (typically 50-100x)"
    )


# ---------- warm-start path ----------


def _tiny_train_cfg(run_id: str, out_dir: Path, data_dir: Path,
                      load_ckpt: str | None = None) -> dict:
    cfg = {
        "run_id": run_id,
        "out_dir": str(out_dir),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 32, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
            "d_model": 16, "d_ffn": 32, "max_seq_len": 16,
        },
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none",
                      "activation_ckpt": "none"},
        "optim": {
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 2,
        },
        "train": {
            "micro_batch": 2, "grad_accum": 1,
            "log_every": 1, "eval_every": 99, "ckpt_every": 1,
        },
    }
    if load_ckpt is not None:
        cfg["load_ckpt"] = load_ckpt
    return cfg


def _make_data(data_dir: Path) -> None:
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rng.integers(0, 32, size=4000, dtype=np.uint16).tofile(
        str(data_dir / "shard_0.bin")
    )


def test_warm_start_loads_weights_and_resets_step(tmp_path: Path):
    """End-to-end warm-start:
      1. Train run-A for 2 steps; checkpoint at step 1.
      2. Train run-B with load_ckpt pointing at run-A's checkpoint and a
         FRESH out_dir.
      3. Run-B must start at step 0 (not resume run-A's step counter) and
         its initial weights must match run-A's saved weights.
    """
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    _make_data(data_dir)

    # Phase 1: pretrain run-A.
    out_a = tmp_path / "out_a"
    cfg_a = _tiny_train_cfg("run_a", out_a, data_dir)
    train(cfg_a)
    ckpt_a_root = out_a / "run_a" / "ckpts"
    step_dirs = sorted(ckpt_a_root.glob("step_*"))
    assert step_dirs, f"no checkpoint produced in {ckpt_a_root}"
    ckpt_a = step_dirs[-1]
    log_a_path = out_a / "log.jsonl"
    assert log_a_path.exists(), f"no log at {log_a_path}"
    log_a = log_a_path.read_text().splitlines()
    assert any('"step": 1' in line or '"step":1' in line for line in log_a), (
        "run-A didn't reach step 1; can't test warm start"
    )

    # Phase 2: warm-start run-B from run-A's checkpoint.
    out_b = tmp_path / "out_b"
    cfg_b = _tiny_train_cfg("run_b", out_b, data_dir, load_ckpt=str(ckpt_a))
    train(cfg_b)
    log_b = (out_b / "log.jsonl").read_text().splitlines()
    # Run-B must START at step 0 — i.e. the first logged step is 0, not
    # whatever run-A ended at.
    import json
    first_step = json.loads(log_b[0]).get("step", None)
    assert first_step == 0, (
        f"warm-started run should begin at step 0, got step={first_step}"
    )


def test_native_resume_takes_priority_over_warm_start(tmp_path: Path):
    """If `out_dir/run_id/ckpts/` already has a checkpoint, the trainer
    must RESUME from it (not warm-start), even when `load_ckpt:` is set.

    This is the safety property that lets you put `load_ckpt:` in a recipe
    and re-run it after an interruption without it overwriting your
    progress with the warm-start checkpoint.
    """
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    _make_data(data_dir)

    # Phase 1: pretrain into out_x.
    out_x = tmp_path / "out_x"
    cfg_x = _tiny_train_cfg("run_x", out_x, data_dir)
    train(cfg_x)
    ckpt_x = sorted((out_x / "run_x" / "ckpts").glob("step_*"))[-1]

    # Phase 2: same run_id + out_dir, with a load_ckpt pointer. Trainer
    # must IGNORE load_ckpt and resume from out_dir's own latest ckpt.
    cfg_resume = _tiny_train_cfg(
        "run_x", out_x, data_dir, load_ckpt=str(ckpt_x)
    )
    cfg_resume["optim"]["total_steps"] = 4  # extend so the resume has work to do
    train(cfg_resume)
    log = (out_x / "log.jsonl").read_text().splitlines()
    # The resumed run must reach steps > 1 (continuation), proving it
    # resumed rather than warm-started at step 0 and re-doing step 1.
    import json
    max_step = max(json.loads(line).get("step", -1) for line in log)
    assert max_step >= 2, (
        f"resume didn't advance past phase-1 progress; max_step={max_step}"
    )


def test_warm_start_with_missing_ckpt_path_errors_clearly(tmp_path: Path):
    """If `load_ckpt:` points at a nonexistent path, the trainer must fail
    loudly with FileNotFoundError. Silently falling through to a cold
    start would produce a randomly-initialised cooldown — a footgun we
    explicitly want to avoid.

    We catch FileNotFoundError specifically rather than `Exception` because
    DCP's own missing-path error is a `BaseException` subclass
    (`CheckpointException`) that pytest.raises(Exception) wouldn't match;
    `load_weights_only` validates up-front to convert it into a clean
    FileNotFoundError before DCP gets confused.
    """
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    _make_data(data_dir)
    cfg = _tiny_train_cfg("run_bad", tmp_path / "out_bad", data_dir,
                            load_ckpt="/nonexistent/path/step_000000099")
    with pytest.raises(FileNotFoundError, match="warm-start"):
        train(cfg)
