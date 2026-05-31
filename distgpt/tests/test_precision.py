"""FP8 / Transformer Engine precision-path tests.

TE is an OPTIONAL dep — most CI machines won't have it. These tests cover
the BEHAVIOUR around TE's absence and the HW-capability gating:

  * resolve_fp8_recipe returns None for 'off' (always safe)
  * resolve_fp8_recipe raises on unknown setting strings
  * resolve_fp8_recipe falls back to None (with warning) on non-bf16 dtype
  * resolve_fp8_recipe falls back to None (with warning) on CPU
  * device_supports_fp8 returns False on CPU regardless of CUDA
  * autocast_fp8_context returns nullcontext when recipe is None
  * autocast_fp8_context raises ImportError when recipe set but TE missing
  * trainer runs to completion with train.fp8: off (no-op path)
  * trainer falls back gracefully when train.fp8 set on CPU (no-FP8 warning)

The actual numerics test (FP8 loss vs bf16) is gated on a Hopper+TE setup
and skipped here.
"""
from __future__ import annotations
import contextlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.training.precision import (  # noqa: E402
    autocast_fp8_context, device_supports_fp8, log_fp8_choice,
    resolve_fp8_recipe,
)


# ---------- resolve_fp8_recipe ----------


def test_resolve_off_returns_none():
    """`train.fp8: off` (or omitted) is always a no-op — even on CPU, even
    without TE, even with fp32 dtype. None means 'use the dense path'."""
    assert resolve_fp8_recipe("off", "cuda", torch.bfloat16) is None
    assert resolve_fp8_recipe(None, "cuda", torch.bfloat16) is None
    assert resolve_fp8_recipe(False, "cuda", torch.bfloat16) is None


def test_resolve_rejects_unknown_setting():
    """Typos in the YAML must raise loudly, not silently disable FP8."""
    with pytest.raises(ValueError, match="unknown"):
        resolve_fp8_recipe("fp8_e4m3_v2", "cuda", torch.bfloat16)
    with pytest.raises(ValueError, match="off.*e4m3.*hybrid"):
        resolve_fp8_recipe("hybird", "cuda", torch.bfloat16)  # typo


def test_resolve_warns_and_disables_on_fp32_dtype():
    """FP8 requires bf16 master dtype; anything else falls back with a
    warning so the user notices the misconfiguration but the run still
    starts."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = resolve_fp8_recipe("hybrid", "cuda", torch.float32)
    assert out is None
    assert any("bfloat16" in str(w.message) for w in caught)


def test_resolve_warns_and_disables_on_cpu():
    """FP8 on CPU is meaningless — fall back to no-FP8 with a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = resolve_fp8_recipe("hybrid", "cpu", torch.bfloat16)
    assert out is None
    assert any("does not support" in str(w.message).lower()
                or "fp8" in str(w.message).lower() for w in caught)


def test_resolve_passes_through_when_hw_and_dtype_ok():
    """On an FP8-capable GPU + bf16, the resolver returns the recipe string
    unchanged so the autocast context can build the TE recipe."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 9:
        pytest.skip(f"GPU sm_{major}x does not support FP8")
    out = resolve_fp8_recipe("hybrid", "cuda", torch.bfloat16)
    assert out == "hybrid"
    out = resolve_fp8_recipe("e4m3", "cuda", torch.bfloat16)
    assert out == "e4m3"


# ---------- device_supports_fp8 ----------


def test_device_supports_fp8_false_on_cpu():
    assert device_supports_fp8("cpu") is False


def test_device_supports_fp8_matches_capability():
    """Sanity-check the capability table against the actual device."""
    if not torch.cuda.is_available():
        assert device_supports_fp8("cuda") is False
        return
    major, _ = torch.cuda.get_device_capability(0)
    expected = major in {9, 10, 12}  # Hopper, Blackwell DC, Blackwell consumer
    assert device_supports_fp8("cuda") is expected


# ---------- autocast_fp8_context ----------


def test_autocast_with_none_recipe_is_nullcontext():
    """recipe=None → nullcontext, so the trainer's `with autocast():` block
    is unconditional even when FP8 is off."""
    ctx = autocast_fp8_context(None)
    # nullcontext is a contextmanager — it must accept enter/exit cleanly.
    assert isinstance(ctx, contextlib.nullcontext)
    with ctx:
        pass


def test_autocast_with_recipe_but_no_te_raises_importerror():
    """If TE isn't installed but a real recipe is requested, build the
    context manager must raise a clear ImportError. Silent fallback would
    hide the throughput win the user expected."""
    try:
        import transformer_engine  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="transformer.engine|transformer-engine"):
            autocast_fp8_context("hybrid")
    else:
        pytest.skip("transformer_engine IS installed; can't test missing-dep path")


# ---------- log_fp8_choice ----------


def test_log_fp8_choice_does_not_raise(capsys):
    """The startup banner is print-only; verify both branches produce output
    rather than blowing up on a missing attribute."""
    log_fp8_choice(None, torch.bfloat16)
    out = capsys.readouterr().out
    assert "disabled" in out
    log_fp8_choice("hybrid", torch.bfloat16)
    out = capsys.readouterr().out
    assert "hybrid" in out and "enabled" in out


# ---------- trainer end-to-end ----------


def test_trainer_runs_with_fp8_off(tmp_path: Path):
    """`train.fp8: off` is the default and must continue to work in
    single-process bf16 mode (covered by other tests too, but this one is
    explicit about the off-path)."""
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    rng.integers(0, 32, size=4000, dtype=np.uint16).tofile(
        str(data_dir / "shard_0.bin")
    )
    cfg = {
        "run_id": "fp8_off",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",  # CPU path, no autocast either
        "log": {"jsonl": True, "wandb_project": None},
        "model": {"vocab_size": 32, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
                   "d_model": 16, "d_ffn": 32, "max_seq_len": 16},
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none",
                      "activation_ckpt": "none"},
        "optim": {
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 2,
        },
        "train": {
            "micro_batch": 2, "grad_accum": 1,
            "log_every": 1, "eval_every": 99, "ckpt_every": 99,
            "fp8": "off",
        },
    }
    train(cfg)


def test_trainer_fp8_on_cpu_falls_back_with_warning(tmp_path: Path):
    """Setting `train.fp8: hybrid` on a CPU run must NOT crash — the
    resolver warns and falls back to no-FP8. This is the user-friendly
    behaviour we want; a misconfigured cluster launch shouldn't lose the
    run."""
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.zeros(4000, dtype=np.uint16).tofile(str(data_dir / "shard_0.bin"))
    cfg = {
        "run_id": "fp8_fallback",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": False, "wandb_project": None},
        "model": {"vocab_size": 32, "n_layer": 1, "n_head": 2, "n_kv_head": 2,
                   "d_model": 16, "d_ffn": 32, "max_seq_len": 16},
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none",
                      "activation_ckpt": "none"},
        "optim": {
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 1,
        },
        "train": {
            "micro_batch": 1, "grad_accum": 1,
            "log_every": 1, "eval_every": 99, "ckpt_every": 99,
            "fp8": "hybrid",  # asks for FP8 on CPU → must warn + disable
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # we already test the warning above
        train(cfg)  # must not raise
