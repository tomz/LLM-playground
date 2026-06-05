import sys, pathlib, math
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig
from utils import cosine_lr


def test_param_count_124m():
    cfg = GPTConfig(vocab_size=50304, block_size=1024, n_layer=12, n_head=12, d_model=768, d_ffn=3072)
    m = GPT(cfg)
    n = m.num_params(non_embedding=True)
    # GPT-2 124M has ~124M non-embedding params
    assert 1.1e8 < n < 1.4e8, n


def test_grad_checkpoint_runs():
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32, d_ffn=64)
    m = GPT(cfg, grad_checkpoint=True).train()
    x = torch.randint(0, 64, (2, 32))
    _, loss = m(x, x)
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_attention_backend_default_and_unknown_rejected():
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, d_model=32, d_ffn=64)
    assert cfg.attention_backend == "sdpa"
    bad = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2,
                    d_model=32, d_ffn=64, attention_backend="nope")
    m = GPT(bad)
    x = torch.randint(0, 64, (1, 8))
    try:
        m(x, x)
    except ValueError as e:
        assert "attention_backend" in str(e)
        return
    raise AssertionError("unknown attention backend should raise")


def test_flex_attention_backend_if_available():
    import pytest
    pytest.importorskip("torch.nn.attention.flex_attention")
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2,
                    d_model=32, d_ffn=64, attention_backend="flex")
    m = GPT(cfg).eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits, loss = m(x, x)
    assert logits.shape == (1, 8, 64)
    assert torch.isfinite(loss)


def test_qk_norm_opt_in():
    # Default off (GPT-2 parity), on when requested.
    base = GPT(GPTConfig(vocab_size=64, block_size=32, n_layer=1, n_head=2, d_model=32, d_ffn=64))
    assert base.blocks[0].attn.q_norm is None
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, d_model=32, d_ffn=64, qk_norm=True)
    m = GPT(cfg).train()
    assert m.blocks[0].attn.q_norm is not None
    assert m.blocks[0].attn.k_norm is not None
    x = torch.randint(0, 64, (2, 32))
    _, loss = m(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in m.parameters())


def test_zero_init_proj_off_by_default():
    """GPT-2 parity: without the flag the residual-write projections are the
    1/sqrt(2N)-rescaled normal init, not zero."""
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32, d_ffn=64)
    assert cfg.zero_init_proj is False
    m = GPT(cfg)
    assert not torch.all(m.blocks[0].attn.proj.weight == 0)
    assert not torch.all(m.blocks[0].mlp.proj.weight == 0)


def test_zero_init_proj_makes_blocks_identity_at_init():
    """With zero_init_proj the attn ``proj`` and MLP ``proj`` start at exactly
    zero, so every block is the identity map ``x + 0`` at init. Running the
    trunk therefore leaves the input embedding unchanged (dropout off / eval).
    This is the muP-like stability property the knob buys."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=3, n_head=2,
                    d_model=32, d_ffn=64, zero_init_proj=True)
    m = GPT(cfg).eval()
    # Both residual-write matrices are exactly zero (after the rescale loop).
    for blk in m.blocks:
        assert torch.all(blk.attn.proj.weight == 0)
        assert torch.all(blk.mlp.proj.weight == 0)
    # The trunk is the identity at init: post-block hidden == input embedding.
    idx = torch.randint(0, 64, (2, 16))
    pos = torch.arange(16)
    x0 = m.tok_emb(idx) + m.pos_emb(pos)
    x = x0.clone()
    for blk in m.blocks:
        x = blk(x, rope=None)
    assert torch.allclose(x, x0, atol=1e-6)


def test_zero_init_proj_trains_after_init():
    """Identity-at-init must not be a dead start: gradients flow into the zeroed
    projections on the first step (the upstream norms/attention are nonzero, so
    d_loss/d_proj.weight != 0) and a step moves them off zero."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                    d_model=32, d_ffn=64, zero_init_proj=True)
    m = GPT(cfg).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.randint(0, 64, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.blocks[0].attn.proj.weight.grad is not None
    assert torch.any(m.blocks[0].attn.proj.weight.grad != 0)
    opt.step()
    assert not torch.all(m.blocks[0].attn.proj.weight == 0)


def test_zero_init_proj_composes_with_swiglu():
    """The knob keys on the ``proj`` attribute, which both MLP and SwiGLU share,
    so it must also zero the SwiGLU down-projection under a llamafied config."""
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, d_model=32,
                    d_ffn=64, zero_init_proj=True, mlp_kind="swiglu",
                    norm_kind="rmsnorm", pos_kind="rope")
    m = GPT(cfg)
    assert torch.all(m.blocks[0].attn.proj.weight == 0)
    assert torch.all(m.blocks[0].mlp.proj.weight == 0)


def test_cosine_lr():
    assert cosine_lr(0, 100, 1000, 1.0, 0.1) > 0
    assert math.isclose(cosine_lr(100, 100, 1000, 1.0, 0.1), 1.0, abs_tol=1e-9)
    assert math.isclose(cosine_lr(2000, 100, 1000, 1.0, 0.1), 0.1, abs_tol=1e-9)
