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
