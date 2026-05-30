"""Muon optimizer + Multi-Token Prediction tests (ported from nanogpt-edu)."""
from __future__ import annotations

import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.training.muon import Muon, newton_schulz5, split_muon_params
from platform.training.optim import OptimConfig, build_optimizer


def _tiny_cfg(**kw) -> ModelConfig:
    base = dict(vocab_size=256, n_layer=2, n_head=4, n_kv_head=2,
                d_model=64, d_ffn=128, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


# ---------- Newton-Schulz orthogonalization ----------

def test_newton_schulz_approximately_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(32, 16)
    X = newton_schulz5(G, steps=5).float()
    # Columns should be ~orthonormal: X^T X ≈ I (up to NS approximation).
    gram = X.T @ X
    eye = torch.eye(16)
    assert (gram - eye).abs().mean() < 0.15


def test_muon_rejects_non_2d_params():
    p = torch.nn.Parameter(torch.randn(8))  # 1-D
    try:
        Muon([p])
        assert False, "should have raised"
    except ValueError:
        pass


# ---------- param splitting ----------

def test_split_muon_params_excludes_io_and_1d():
    m = Transformer(_tiny_cfg(mtp_tokens=2))
    muon_params, adamw = split_muon_params(m)
    muon_ids = {id(p) for p in muon_params}
    # tok_emb / lm_head must NOT be in the Muon group.
    assert id(m.tok_emb.weight) not in muon_ids
    assert id(m.lm_head.weight) not in muon_ids
    for head in m.mtp_heads:
        assert id(head.weight) not in muon_ids
    # A hidden attention matrix SHOULD be in the Muon group.
    assert id(m.layers[0].attn.q_proj.weight) in muon_ids
    # All Muon params are 2D.
    assert all(p.ndim == 2 for p in muon_params)


# ---------- Muon end-to-end training ----------

def test_muon_optimizer_reduces_loss():
    torch.manual_seed(0)
    m = Transformer(_tiny_cfg())
    opt, sched = build_optimizer(
        m, OptimConfig(name="muon", peak_lr=3e-3, muon_lr=0.02,
                       warmup_steps=2, total_steps=40)
    )
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    losses = []
    for _ in range(40):
        _, loss = m(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
    assert losses[-1] < 0.6 * losses[0], (losses[0], losses[-1])


def test_muon_optimizer_state_dict_roundtrips():
    torch.manual_seed(0)
    m = Transformer(_tiny_cfg())
    opt, _ = build_optimizer(m, OptimConfig(name="muon", total_steps=10))
    x = torch.randint(0, 256, (2, 8)); y = torch.randint(0, 256, (2, 8))
    _, loss = m(x, targets=y); loss.backward(); opt.step()
    sd = opt.state_dict()
    assert "optimizers" in sd and len(sd["optimizers"]) == 2
    opt.load_state_dict(sd)  # should not raise


# ---------- Multi-Token Prediction ----------

def test_mtp_adds_aux_loss_in_train_only():
    torch.manual_seed(0)
    plain = Transformer(_tiny_cfg(mtp_tokens=0))
    torch.manual_seed(0)
    mtp = Transformer(_tiny_cfg(mtp_tokens=2, mtp_weight=0.3))
    assert len(mtp.mtp_heads) == 2 and len(plain.mtp_heads) == 0

    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))

    # Train mode: MTP loss > plain loss (extra aux term), both finite.
    mtp.train(); plain.train()
    _, l_plain = plain(x, targets=y)
    _, l_mtp = mtp(x, targets=y)
    assert torch.isfinite(l_mtp) and l_mtp > l_plain

    # Eval mode: MTP heads are skipped -> pure next-token CE (no aux).
    mtp.eval()
    with torch.no_grad():
        _, l_eval = mtp(x, targets=y)
    # The eval loss should match a from-scratch CE on the main head only.
    import torch.nn.functional as F
    with torch.no_grad():
        logits, _ = mtp(x)
        ref = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
    assert abs(float(l_eval) - float(ref)) < 1e-4


def test_mtp_heads_discarded_at_inference_shape():
    """MTP must not change the main logits shape (inference path is unchanged)."""
    m = Transformer(_tiny_cfg(mtp_tokens=3)).eval()
    x = torch.randint(0, 256, (2, 12))
    logits, _ = m(x)
    assert logits.shape == (2, 12, 256)
