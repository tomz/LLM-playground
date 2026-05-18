from __future__ import annotations
import math
import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer, MoEFFN


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
        moe_num_experts=4, moe_top_k=2,
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
