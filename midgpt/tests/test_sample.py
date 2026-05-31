"""Generation modes: greedy, top-k, top-p (nucleus), and combinations."""
from __future__ import annotations
import sys, pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig


def _tiny():
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2,
                         d_model=32, d_ffn=64, tie_embeddings=True)).eval()


def test_greedy_is_deterministic():
    """Temperature 0 → argmax → identical output across calls regardless of
    RNG state. Catches a bug class where someone accidentally falls into
    multinomial at zero temperature (the old code's ``max(temperature, 1e-6)``
    would still draw from a near-one-hot distribution but wasn't strictly
    argmax)."""
    m = _tiny()
    idx = torch.tensor([[0, 1, 2, 3]])
    torch.manual_seed(0); a = m.generate(idx, max_new_tokens=16, temperature=0)
    torch.manual_seed(999); b = m.generate(idx, max_new_tokens=16, temperature=0)
    assert torch.equal(a, b), (a.tolist(), b.tolist())


def test_top_k_restricts_support():
    """With ``top_k=1`` every sampled token must equal the per-step argmax;
    that's just greedy via a different code path."""
    m = _tiny()
    idx = torch.tensor([[0, 1, 2, 3]])
    torch.manual_seed(0); a = m.generate(idx, max_new_tokens=8, temperature=1.0, top_k=1)
    g = m.generate(idx, max_new_tokens=8, temperature=0)
    assert torch.equal(a, g)


def test_top_p_keeps_at_least_one_token():
    """With ``top_p`` ≪ 1 we'd otherwise mask out every token in the support;
    the implementation must always admit the top-1 token so multinomial has
    a valid distribution to sample from."""
    m = _tiny()
    idx = torch.tensor([[0, 1, 2, 3]])
    # top_p smaller than any single token's mass should still produce a
    # valid sample (the argmax token).
    torch.manual_seed(0)
    out = m.generate(idx, max_new_tokens=4, temperature=1.0, top_p=1e-6)
    assert out.shape == (1, 4 + 4)


def test_top_p_combinable_with_top_k():
    """top_k is applied first, then top_p restricts that further. The output
    must always be a valid integer in [0, vocab_size)."""
    m = _tiny()
    idx = torch.tensor([[0, 1, 2, 3]])
    torch.manual_seed(0)
    out = m.generate(idx, max_new_tokens=4, temperature=1.0, top_k=8, top_p=0.5)
    assert (out >= 0).all() and (out < 64).all()
