"""Muon optimizer port: orthogonalization, param split, and a tiny train step."""
import sys, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig
from muon import Muon, newton_schulz5, split_muon_params


def _tiny():
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                         d_model=32, d_ffn=64, tie_embeddings=True))


def test_newton_schulz_orthogonalizes():
    # A random tall matrix; after NS5 the columns should be near-orthonormal.
    torch.manual_seed(0)
    g = torch.randn(64, 16)
    q = newton_schulz5(g).float()
    gram = q.T @ q
    eye = torch.eye(16)
    # Not exact (5 steps, bf16) — check it's close to semi-orthogonal.
    assert (gram - eye).abs().mean() < 0.1, (gram - eye).abs().mean().item()


def test_split_excludes_io_layers():
    m = _tiny()
    muon_params, adamw_params = split_muon_params(m)
    # Every Muon param is a 2D hidden weight matrix.
    assert all(p.ndim == 2 for p in muon_params)
    # tok_emb / pos_emb / lm_head must NOT be in the Muon group.
    muon_ids = {id(p) for p in muon_params}
    assert id(m.tok_emb.weight) not in muon_ids
    assert id(m.pos_emb.weight) not in muon_ids
    # lm_head is tied to tok_emb here, so it's excluded via tok_emb too.
    assert id(m.lm_head.weight) not in muon_ids
    # Partition is exhaustive over trainable params (counting tied weight once).
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    assert muon_ids | {id(p) for p in adamw_params} == trainable


def test_split_excludes_pos_emb_when_untied():
    # With untied embeddings, lm_head is its own 2D matrix and must still be excluded.
    m = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2,
                      d_model=32, d_ffn=64, tie_embeddings=False))
    muon_params, _ = split_muon_params(m)
    muon_ids = {id(p) for p in muon_params}
    assert id(m.pos_emb.weight) not in muon_ids
    assert id(m.lm_head.weight) not in muon_ids
    assert id(m.tok_emb.weight) not in muon_ids


def test_muon_rejects_non_2d():
    p = torch.nn.Parameter(torch.zeros(8))
    try:
        Muon([p])
    except ValueError:
        return
    raise AssertionError("Muon should reject 1-D params")


def test_muon_step_updates_weights():
    m = _tiny().train()
    muon_params, adamw_params = split_muon_params(m)
    muon = Muon(muon_params, lr=0.02)
    aux = torch.optim.AdamW(adamw_params, lr=1e-3)
    before = [p.detach().clone() for p in muon_params]
    x = torch.randint(0, 64, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    muon.step(); aux.step()
    assert torch.isfinite(loss)
    # At least one Muon-managed matrix actually moved.
    assert any(not torch.equal(b, p) for b, p in zip(before, muon_params))
