import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig


def test_forward_and_loss():
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ffn=128)
    m = GPT(cfg)
    x = torch.randint(0, 65, (2, 32))
    logits, loss = m(x, x)
    assert logits.shape == (2, 32, 65)
    assert loss.dim() == 0 and loss.item() > 0


def test_generate():
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ffn=128)
    m = GPT(cfg).eval()
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=10)
    assert out.shape == (1, 11)


def test_overfits_one_batch():
    """A 2-layer model should drive loss << random on 1 batch."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ffn=128)
    m = GPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 20, (4, 16)); y = torch.randint(0, 20, (4, 16))
    for _ in range(150):
        opt.zero_grad(); _, loss = m(x, y); loss.backward(); opt.step()
    import math
    assert loss.item() < math.log(20) * 0.5


def test_speedrun_knobs_change_shape_and_forward():
    """qk_norm adds Q/K norms; untie gives lm_head its own weight; zero_init
    zeroes the residual-write projections. Forward/loss must still work."""
    cfg = GPTConfig(vocab_size=33, block_size=16, n_layer=2, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, qk_norm=True, zero_init_proj=True,
                    tie_embeddings=False)
    m = GPT(cfg)
    # untied: lm_head and tok_emb are distinct tensors
    assert m.lm_head.weight is not m.tok_emb.weight
    # zero-init: residual-write matrices start at exactly zero
    assert torch.all(m.blocks[0].attn.o_proj.weight == 0)
    assert torch.all(m.blocks[0].ffn.w2.weight == 0)
    # qk_norm modules exist
    assert m.blocks[0].attn.q_norm is not None
    x = torch.randint(0, 33, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, 33)
    assert loss.item() > 0


def test_muon_overfits_one_batch():
    """Muon (2D weights) + AdamW (rest) should also overfit a single batch."""
    import math
    from muon import Muon, split_muon_params
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, tie_embeddings=False)
    m = GPT(cfg)
    mp, ap = split_muon_params(m)
    assert len(mp) > 0 and len(ap) > 0
    opts = [Muon(mp, lr=0.02), torch.optim.AdamW(ap, lr=3e-3)]
    x = torch.randint(0, 20, (4, 16)); y = torch.randint(0, 20, (4, 16))
    for _ in range(150):
        for o in opts: o.zero_grad()
        _, loss = m(x, y); loss.backward()
        for o in opts: o.step()
    assert loss.item() < math.log(20) * 0.5


def test_muon_rejects_non_2d_params():
    from muon import Muon
    p = torch.nn.Parameter(torch.zeros(8))  # 1-D
    try:
        Muon([p])
    except ValueError:
        return
    raise AssertionError("Muon should reject non-2D parameters")


def test_newton_schulz_orthogonalizes():
    """After NS iteration the singular values should be ~1 (semi-orthogonal)."""
    from muon import newton_schulz5
    torch.manual_seed(0)
    G = torch.randn(16, 24)
    O = newton_schulz5(G).float()
    # O O^T should be close to identity on the smaller dimension
    s = torch.linalg.svdvals(O)
    assert (s.max() < 1.4) and (s.min() > 0.6)
