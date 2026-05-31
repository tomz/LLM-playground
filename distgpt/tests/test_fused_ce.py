"""Fused linear-cross-entropy (Liger) integration tests.

Liger is an OPTIONAL dep — if the kernel package isn't installed (or runs
on a non-Triton GPU), `model.fused_ce=True` must raise a clear `ImportError`
at construction time, not silently disable itself or crash mid-step.

When the package IS available, the fused path must produce the same loss
(within bf16 rounding) as the dense `lm_head @ x + cross_entropy` path. We
guard the numeric-equivalence test with `pytest.importorskip` so CI works
without the package.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distgpt.model.config import ModelConfig  # noqa: E402
from distgpt.model.transformer import GPT  # noqa: E402


def test_default_model_does_not_import_liger():
    """The default build path must not even attempt to import liger-kernel —
    `fused_ce` defaults to False so the user pays nothing if they don't
    opt in. We verify that by building a model fresh and checking the
    private slot stays None.
    """
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                      d_model=32, d_ffn=64, max_seq_len=16)
    m = GPT(cfg)
    assert cfg.fused_ce is False
    assert m._liger_ce is None


def test_fused_ce_true_raises_importerror_when_liger_missing():
    """If liger-kernel isn't installed, asking for fused_ce=True must raise
    `ImportError` at GPT.__init__ time with a message pointing the user at
    the install command. Silent disable would be a footgun — the user would
    get the dense path at the same VRAM cost they thought they'd cut.
    """
    try:
        import liger_kernel  # noqa: F401
    except ImportError:
        cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                          d_model=32, d_ffn=64, max_seq_len=16, fused_ce=True)
        with pytest.raises(ImportError, match="liger-kernel"):
            GPT(cfg)
    else:
        pytest.skip("liger-kernel is installed; can't test the missing-dep path")


def test_fused_ce_matches_dense_loss_when_available():
    """When liger is available, the fused-CE loss must match the dense
    `lm_head @ x + cross_entropy` loss to within bf16 rounding tolerance.

    This is the only test that exercises the actual kernel; gated on the
    optional dep so CI passes without it.
    """
    pytest.importorskip("liger_kernel")
    if not torch.cuda.is_available():
        pytest.skip("liger-kernel needs CUDA + Triton")
    torch.manual_seed(0)
    cfg_dense = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                            d_model=32, d_ffn=64, max_seq_len=16,
                            fused_ce=False)
    cfg_fused = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                            d_model=32, d_ffn=64, max_seq_len=16,
                            fused_ce=True)
    m_dense = GPT(cfg_dense).cuda()
    m_fused = GPT(cfg_fused).cuda()
    m_fused.load_state_dict(m_dense.state_dict())

    x = torch.randint(0, 64, (2, 8), device="cuda")
    _, loss_dense = m_dense(x, x)
    _, loss_fused = m_fused(x, x)
    # bf16 inside the kernel — tolerance taken from midgpt's measurement.
    rel = (loss_dense - loss_fused).abs() / loss_dense.abs().clamp_min(1e-6)
    assert rel.item() < 5e-3, (
        f"fused vs dense CE diverged: dense={loss_dense.item():.6f} "
        f"fused={loss_fused.item():.6f} rel={rel.item():.2e}"
    )


def test_fused_ce_returns_none_logits_when_targets_given():
    """The fused kernel intentionally never materializes logits — the GPT
    forward must signal that by returning `None` in the logits slot when
    fused_ce is on AND targets are provided. Callers needing logits must
    set fused_ce=False; this is the contract midgpt also follows.
    """
    pytest.importorskip("liger_kernel")
    if not torch.cuda.is_available():
        pytest.skip("liger-kernel needs CUDA + Triton")
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                      d_model=32, d_ffn=64, max_seq_len=16, fused_ce=True)
    m = GPT(cfg).cuda()
    x = torch.randint(0, 64, (2, 8), device="cuda")
    logits, loss = m(x, x)
    assert logits is None
    assert loss is not None and torch.isfinite(loss)


def test_fused_ce_inference_path_still_returns_logits():
    """When `targets is None` (inference / generation), the model must
    return the last-token logits regardless of fused_ce setting — the
    fused kernel only handles the training loss path.
    """
    pytest.importorskip("liger_kernel")
    if not torch.cuda.is_available():
        pytest.skip("liger-kernel needs CUDA + Triton")
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=2,
                      d_model=32, d_ffn=64, max_seq_len=16, fused_ce=True)
    m = GPT(cfg).cuda()
    x = torch.randint(0, 64, (2, 8), device="cuda")
    logits, loss = m(x, targets=None)
    assert loss is None
    assert logits is not None
    assert logits.shape == (2, 1, 64)
