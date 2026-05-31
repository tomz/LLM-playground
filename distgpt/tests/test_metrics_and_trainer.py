"""Tests for the observability helpers in distgpt/utils/metrics.py and
their integration into the trainer's log."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.model.config import ModelConfig  # noqa: E402
from distgpt.model.transformer import GPT  # noqa: E402
from distgpt.utils.metrics import (  # noqa: E402
    compute_grad_norm, compute_param_norm, estimate_mfu, model_flops_per_token,
    peak_tflops_for_device,
)


def _tiny_model() -> GPT:
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                       d_model=32, d_ffn=64, max_seq_len=16)
    return GPT(cfg)


def test_compute_grad_norm_zero_when_no_grads():
    """A fresh model has p.grad is None — must return 0.0, not crash."""
    m = _tiny_model()
    assert compute_grad_norm(m) == 0.0


def test_compute_grad_norm_matches_clip_grad_norm_total():
    """The reported norm must equal what clip_grad_norm_ would compute on
    the same grads — that's what makes the log value comparable across
    runs that do/don't clip."""
    m = _tiny_model().train()
    x = torch.randint(0, 64, (2, 8))
    _, loss = m(x, x)
    loss.backward()
    expected = float(torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf")))
    # Re-run forward/backward so the (clipped) grads from the last call
    # don't pollute the second measurement. clip_grad_norm_(... inf) returns
    # the pre-clip norm without modifying.
    got = compute_grad_norm(m)
    assert got == pytest.approx(expected, rel=1e-5)


def test_compute_param_norm_is_finite_and_positive():
    m = _tiny_model()
    n = compute_param_norm(m)
    assert n > 0
    assert np.isfinite(n)


def test_peak_tflops_none_on_cpu():
    """No CUDA → no peak → trainer must omit MFU rather than report nonsense."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this test pins the CPU fallback")
    assert peak_tflops_for_device(torch.bfloat16) is None


def test_model_flops_per_token_is_6n():
    """Sanity-pin the 6N approximation (Chinchilla / PaLM convention)."""
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                       d_model=32, d_ffn=64, max_seq_len=16)
    expected = 6.0 * cfg.param_count()
    assert model_flops_per_token(cfg) == expected


def test_estimate_mfu_returns_none_for_bad_inputs():
    cfg = ModelConfig(n_layer=1, n_head=2, n_kv_head=2, d_model=32, d_ffn=64)
    assert estimate_mfu(cfg, 1, 0, 100, 1) is None      # dt=0
    assert estimate_mfu(cfg, 1, 1, 0, 1) is None        # peak=0
    assert estimate_mfu(cfg, 1, 1, 100, 0) is None      # world=0


def test_estimate_mfu_in_unit_interval_for_realistic_inputs():
    cfg = ModelConfig(n_layer=24, n_head=16, n_kv_head=8,
                       d_model=2048, d_ffn=5632)
    # 1B model, 32k tok/step, 1s/step on an A100 peak (312 TF). MFU = (6×1e9
    # × 32k) / 1s / 312 TF = ~0.61. Don't assert the exact value, just that
    # it's in a plausible range.
    mfu = estimate_mfu(cfg, tokens_per_step=32_768, dt_seconds=1.0,
                        peak_tflops=312.0, world_size=1)
    assert 0.3 < mfu < 1.5  # bigger than 1 means the model's peak is wrong
                            # — caller's job to notice that, not ours.


def test_trainer_log_includes_new_metrics(tmp_path: Path):
    """The trainer's JSONL log now carries grad_norm, mfu (when known),
    tok_per_s_per_gpu. This is the user-visible contract; pin it."""
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        rng.integers(0, 256, size=8_000, dtype=np.uint16).tofile(
            str(data_dir / f"shard_{i:06d}.bin")
        )

    cfg = {
        "run_id": "smoke_metrics",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 256, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
            "d_model": 32, "d_ffn": 64, "max_seq_len": 16,
            "rope_base": 10000.0, "tie_embeddings": True,
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
            "log_every": 1, "eval_every": 1, "ckpt_every": 1,
            "compile": False,
        },
    }
    train(cfg)
    log = Path(cfg["out_dir"]) / "log.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()
             if line.strip()]
    train_rows = [r for r in rows if "loss" in r and "lr" in r]
    assert train_rows, "no train rows in log"
    sample = train_rows[0]
    for key in ("loss", "lr", "ms", "tok_per_s", "tok_per_s_per_gpu",
                "grad_norm"):
        assert key in sample, f"missing log key: {key}; got {list(sample)}"
    # Grad norm must be finite and positive after a real backward pass.
    assert sample["grad_norm"] > 0 and np.isfinite(sample["grad_norm"])
    # On the very first log step (step // log_every == 0) we also sample
    # param_norm.
    first = next(r for r in train_rows if r["step"] == 0)
    assert "param_norm" in first


def test_trainer_compile_knob_does_not_break_run(tmp_path: Path):
    """`train.compile: true` must not crash the loop. The compiled path
    is a no-op fallback on backends that don't support it; the metric we
    pin is 'training still completes', not 'speedup happens'."""
    from distgpt.training.trainer import train

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        rng.integers(0, 256, size=8_000, dtype=np.uint16).tofile(
            str(data_dir / f"shard_{i:06d}.bin")
        )

    cfg = {
        "run_id": "smoke_compile",
        "out_dir": str(tmp_path / "out"),
        "data": {"dir": str(data_dir), "seq_len": 16},
        "seed": 0,
        "dtype": "float32",
        "log": {"jsonl": True, "wandb_project": None},
        "model": {
            "vocab_size": 256, "n_layer": 2, "n_head": 2, "n_kv_head": 2,
            "d_model": 32, "d_ffn": 64, "max_seq_len": 16,
            "rope_base": 10000.0, "tie_embeddings": True,
        },
        "parallel": {"dp": 1, "tp": 1, "pp": 1, "zero": "none",
                      "activation_ckpt": "none"},
        "optim": {
            "lr": 1e-3, "min_lr": 1e-4, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "grad_clip": 1.0,
            "warmup_steps": 1, "total_steps": 2,
        },
        "train": {
            "micro_batch": 1, "grad_accum": 1,
            "log_every": 1, "eval_every": 99, "ckpt_every": 99,
            "compile": True,   # the bit we're testing
        },
    }
    train(cfg)
    log = Path(cfg["out_dir"]) / "log.jsonl"
    assert log.exists()
