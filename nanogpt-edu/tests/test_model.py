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


def test_attention_backend_default_and_unknown_rejected():
    base = GPTConfig(vocab_size=65, block_size=16, n_layer=1, n_head=4, n_kv_head=2, d_model=64, d_ffn=128)
    assert base.attention_backend == "sdpa"
    bad = GPTConfig(vocab_size=65, block_size=16, n_layer=1, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, attention_backend="nope")
    m = GPT(bad)
    x = torch.randint(0, 65, (1, 8))
    try:
        m(x, x)
    except ValueError as e:
        assert "attention_backend" in str(e)
        return
    raise AssertionError("unknown attention backend should raise")


def test_flex_attention_backend_if_available():
    pytest = __import__("pytest")
    pytest.importorskip("torch.nn.attention.flex_attention")
    cfg = GPTConfig(vocab_size=65, block_size=16, n_layer=1, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, attention_backend="flex")
    m = GPT(cfg).eval()
    x = torch.randint(0, 65, (1, 8))
    with torch.no_grad():
        logits, loss = m(x, x)
    assert logits.shape == (1, 8, 65)
    assert torch.isfinite(loss)


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


def test_mtp_adds_heads_and_train_loss_differs_from_eval():
    """MTP creates auxiliary heads; the aux loss is train-only so eval (model
    in .eval()) reports a *smaller* loss than train mode on the same batch."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, mtp_tokens=2, mtp_weight=0.3)
    m = GPT(cfg)
    assert len(m.mtp_heads) == 2
    x = torch.randint(0, 30, (2, 16)); y = torch.randint(0, 30, (2, 16))
    m.train()
    _, train_loss = m(x, y)
    m.eval()
    _, eval_loss = m(x, y)
    # eval loss is pure next-token CE; train loss adds the positive aux term.
    assert train_loss.item() > eval_loss.item()


def test_mtp_off_by_default():
    cfg = GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128)
    m = GPT(cfg)
    assert len(m.mtp_heads) == 0
    x = torch.randint(0, 30, (2, 16))
    m.train(); _, l_train = m(x, x)
    m.eval();  _, l_eval = m(x, x)
    # with no MTP, train and eval losses match exactly (no dropout here)
    assert abs(l_train.item() - l_eval.item()) < 1e-5


def test_mtp_heads_routed_to_adamw_not_muon():
    from muon import split_muon_params
    cfg = GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, mtp_tokens=2, tie_embeddings=False)
    m = GPT(cfg)
    mp, ap = split_muon_params(m)
    n_mtp_in_muon = sum(
        1 for name, p in m.named_parameters()
        if "mtp_heads" in name and any(p is q for q in mp)
    )
    assert n_mtp_in_muon == 0


def test_hidden_matches_forward_logits():
    """`hidden()` is the shared trunk primitive: lm_head(hidden(x)) must equal
    the logits returned by forward()."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=40, block_size=32, n_layer=2, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, mtp_tokens=2)
    m = GPT(cfg).eval()
    x = torch.randint(0, 40, (1, 12))
    logits, _ = m(x)
    h = m.hidden(x)
    assert torch.allclose(m.lm_head(h), logits, atol=1e-5)


def test_mtp_speculative_matches_greedy_exactly():
    """Greedy verification makes the MTP-speculative decoder bit-identical to
    plain greedy decoding — the speedup must be lossless."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=48, block_size=64, n_layer=2, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, mtp_tokens=3)
    m = GPT(cfg).eval()
    idx = torch.randint(0, 48, (1, 8))
    g = m.generate_greedy(idx.clone(), 40)
    s, stats = m.generate_mtp_speculative(idx.clone(), 40)
    assert torch.equal(g, s)
    assert g.size(1) == idx.size(1) + 40           # exact token budget
    assert sum(stats["accepted"]) == 40
    # Each verification round emits between 1 and K+2 tokens (true token + up to
    # K accepted drafts + 1 bonus token when the whole chain is accepted).
    assert all(1 <= a <= len(m.mtp_heads) + 2 for a in stats["accepted"])


def test_mtp_speculative_falls_back_without_heads():
    """With mtp_tokens=0 the speculative path is just greedy decoding."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=48, block_size=64, n_layer=2, n_head=4, n_kv_head=2,
                    d_model=64, d_ffn=128, mtp_tokens=0)
    m = GPT(cfg).eval()
    idx = torch.randint(0, 48, (1, 8))
    g = m.generate_greedy(idx.clone(), 20)
    s, stats = m.generate_mtp_speculative(idx.clone(), 20)
    assert torch.equal(g, s)
    assert stats["rounds"] == 0
