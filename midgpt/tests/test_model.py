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


def test_cosine_lr():
    assert cosine_lr(0, 100, 1000, 1.0, 0.1) > 0
    assert math.isclose(cosine_lr(100, 100, 1000, 1.0, 0.1), 1.0, abs_tol=1e-9)
    assert math.isclose(cosine_lr(2000, 100, 1000, 1.0, 0.1), 0.1, abs_tol=1e-9)
