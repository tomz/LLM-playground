"""Precision policy tests (platform/training/precision.py)."""
from __future__ import annotations

import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.training.precision import (
    PrecisionPolicy,
    resolve_precision,
    cuda_supports_bf16,
)


def test_resolve_precision_cpu_falls_back_to_fp32():
    # On a machine without a bf16 CUDA device, fp8/bf16 fall back to fp32.
    dt, backend = resolve_precision("fp8")
    if not cuda_supports_bf16():
        assert dt == "fp32"
        assert backend in ("fp32_fallback",)
    else:
        assert dt == "bf16"


def test_resolve_precision_validates():
    try:
        resolve_precision("int3")
        assert False
    except ValueError:
        pass


def test_fp32_policy_autocast_is_noop_and_forward_runs():
    pol = PrecisionPolicy.create("fp32")
    assert pol.autocast_dtype == "fp32"
    assert not pol.uses_fp8
    m = Transformer(ModelConfig(vocab_size=128, n_layer=2, n_head=2, n_kv_head=1,
                                d_model=32, d_ffn=64, max_seq_len=32))
    x = torch.randint(0, 128, (2, 16)); y = torch.randint(0, 128, (2, 16))
    with pol.autocast():
        _, loss = m(x, targets=y)
    assert torch.isfinite(loss)


def test_fp8_requested_runs_through_fallback_forward():
    """Requesting fp8 must not crash on CPU — it resolves to a runnable policy
    and the forward still produces finite loss (so the code path is GPU-ready)."""
    pol = PrecisionPolicy.create("fp8")
    m = Transformer(ModelConfig(vocab_size=128, n_layer=2, n_head=2, n_kv_head=1,
                                d_model=32, d_ffn=64, max_seq_len=32))
    x = torch.randint(0, 128, (2, 16)); y = torch.randint(0, 128, (2, 16))
    with pol.autocast():
        _, loss = m(x, targets=y)
    assert torch.isfinite(loss)


def test_parallel_engine_uses_precision_policy():
    from platform.training.parallel import ParallelConfig, ParallelEngine
    m = Transformer(ModelConfig(vocab_size=128, n_layer=2, n_head=2, n_kv_head=1,
                                d_model=32, d_ffn=64, max_seq_len=32))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    eng = ParallelEngine(m, opt, ParallelConfig(precision="fp8"))
    x = torch.randint(0, 128, (2, 16)); y = torch.randint(0, 128, (2, 16))
    metrics = eng.forward_backward((x, y))
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert "precision_backend" in metrics
