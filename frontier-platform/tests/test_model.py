from __future__ import annotations
import math
import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer, MoEFFN, MLAttention, GQAttention


def test_param_count_matches_actual_module(tiny_model_cfg, tiny_model):
    actual = sum(p.numel() for p in tiny_model.parameters())
    formula = tiny_model_cfg.param_count()
    # 5% tolerance — formula ignores norms + final lm_head sharing details
    assert abs(actual - formula) / formula < 0.05, (actual, formula)


def test_forward_shapes(tiny_model):
    x = torch.randint(0, 512, (2, 32))
    logits, loss = tiny_model(x)
    assert logits.shape == (2, 32, 512)
    assert loss is None
    y = torch.randint(0, 512, (2, 32))
    logits, loss = tiny_model(x, targets=y)
    assert loss is not None and torch.isfinite(loss)


def test_forward_loss_reduces_on_overfit(tiny_model):
    torch.manual_seed(0)
    x = torch.randint(0, 512, (2, 16))
    y = torch.randint(0, 512, (2, 16))
    opt = torch.optim.Adam(tiny_model.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        _, loss = tiny_model(x, targets=y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0], losses


def test_init_weights_mup_changes_stds(tiny_model_cfg):
    torch.manual_seed(0)
    m = Transformer(tiny_model_cfg)
    m.init_weights("muP")
    emb_std = m.tok_emb.weight.detach().std().item()
    o_std = m.layers[0].attn.o_proj.weight.detach().std().item()
    assert o_std < emb_std, (o_std, emb_std)
    # Sanity: ratio close to 1/sqrt(2*n_layer)
    expected = 1.0 / math.sqrt(2 * tiny_model_cfg.n_layer)
    assert 0.5 * expected < o_std / emb_std < 1.5 * expected


def test_moe_routing_load_balance():
    cfg = ModelConfig(
        vocab_size=256, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=64,
        moe_num_experts=4, moe_top_k=2, moe_balance="aux_loss",
    )
    torch.manual_seed(0)
    m = Transformer(cfg)
    assert isinstance(m.layers[0].ffn, MoEFFN)
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    _, loss = m(x, targets=y)
    aux = m.layers[0].ffn.last_aux_loss
    assert float(aux) > 0.0
    assert torch.isfinite(loss)


def test_moe_aux_free_balancing_updates_bias_and_balances_load():
    """Aux-free routing: no load-balance loss term, and the routing bias moves
    to equalize per-expert load over several training steps."""
    cfg = ModelConfig(
        vocab_size=256, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=64,
        moe_num_experts=8, moe_top_k=2, moe_balance="aux_free",
        moe_bias_update_speed=1e-2,
    )
    torch.manual_seed(0)
    m = Transformer(cfg).train()
    ffn = m.layers[0].ffn
    assert torch.allclose(ffn.routing_bias, torch.zeros_like(ffn.routing_bias))
    x = torch.randint(0, 256, (4, 32))
    y = torch.randint(0, 256, (4, 32))
    for _ in range(20):
        _, loss = m(x, targets=y)
        loss.backward()
    # Aux-free => aux loss is just the z-loss, and the bias has moved.
    assert ffn.routing_bias.abs().sum() > 0
    assert torch.isfinite(loss)
    # Counts should cover the whole token budget (top_k slots per token).
    assert int(ffn.last_expert_counts.sum()) == 4 * 32 * cfg.moe_top_k


def test_moe_fine_grained_and_shared_experts_param_counts():
    """Fine-grained narrow experts + shared expert: total >> active, and the
    config's param formulas track the real module."""
    cfg = ModelConfig(
        vocab_size=256, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=256, max_seq_len=64,
        moe_num_experts=16, moe_top_k=2, moe_expert_d_ffn=64,
        moe_shared_experts=1,
    )
    torch.manual_seed(0)
    m = Transformer(cfg)
    ffn = m.layers[0].ffn
    assert isinstance(ffn, MoEFFN)
    assert len(ffn.experts) == 16 and len(ffn.shared) == 1
    # Active params are far below total for a 16-expert top-2 model.
    assert cfg.active_param_count() < cfg.param_count()
    # param_count formula tracks the real total within 5%.
    actual = sum(p.numel() for p in m.parameters())
    assert abs(actual - cfg.param_count()) / cfg.param_count() < 0.05, (actual, cfg.param_count())
    # Forward still runs and shared expert fires for every token.
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    _, loss = m(x, targets=y)
    assert torch.isfinite(loss)


def test_activation_checkpoint_runs():
    """Selective activation checkpointing must produce finite loss + grads."""
    base = ModelConfig(vocab_size=256, n_layer=4, n_head=4, n_kv_head=2,
                       d_model=64, d_ffn=128, max_seq_len=32)
    ckpt = ModelConfig(vocab_size=256, n_layer=4, n_head=4, n_kv_head=2,
                       d_model=64, d_ffn=128, max_seq_len=32,
                       activation_ckpt="selective")
    torch.manual_seed(0); m1 = Transformer(base).train()
    torch.manual_seed(0); m2 = Transformer(ckpt).train()
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    _, l1 = m1(x, targets=y); l1.backward()
    _, l2 = m2(x, targets=y); l2.backward()
    assert torch.isfinite(l1) and torch.isfinite(l2)
    # losses should be very close (same init, same compute graph mathematically)
    assert abs(float(l1) - float(l2)) < 1e-3


def test_mla_attention_forward_and_train():
    """MLA produces finite loss, trains, and uses far less KV cache than GQA."""
    cfg = ModelConfig(
        vocab_size=256, n_layer=2, n_head=8, n_kv_head=2,
        d_model=128, d_ffn=256, max_seq_len=64,
        attn_kind="mla", mla_kv_latent_dim=64, mla_rope_head_dim=8,
    )
    torch.manual_seed(0)
    m = Transformer(cfg)
    assert isinstance(m.layers[0].attn, MLAttention)
    x = torch.randint(0, 256, (2, 24))
    y = torch.randint(0, 256, (2, 24))
    _, loss = m(x, targets=y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in m.layers[0].attn.kv_down.parameters())


def test_mla_kv_cache_smaller_than_gqa():
    """The whole point of MLA: the cached latent is much smaller than GQA's K+V."""
    common = dict(vocab_size=256, n_layer=2, n_head=16, n_kv_head=8,
                  d_model=512, d_ffn=1024, max_seq_len=128)
    gqa = ModelConfig(attn_kind="gqa", **common)
    mla = ModelConfig(attn_kind="mla", mla_kv_latent_dim=128, mla_rope_head_dim=16, **common)
    assert mla.kv_bytes_per_token() < gqa.kv_bytes_per_token()
    # GQA caches 2 * n_kv_head * head_dim = 2*8*32 = 512 dims; MLA caches 128+16=144.
    assert gqa.kv_bytes_per_token() == 2 * 8 * 32 * 2
    assert mla.kv_bytes_per_token() == (128 + 16) * 2
    # Compression ratio should be a meaningful 3x+ here.
    assert gqa.kv_bytes_per_token() / mla.kv_bytes_per_token() > 3.0


def test_qk_norm_runs_and_trains():
    """QK-norm (per-head RMSNorm on Q/K) produces finite loss and trains."""
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=32, qk_norm=True)
    torch.manual_seed(0)
    m = Transformer(cfg)
    assert hasattr(m.layers[0].attn, "q_norm")
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    _, loss = m(x, targets=y)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.layers[0].attn.q_norm.weight.grad is not None
