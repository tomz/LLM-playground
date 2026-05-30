import sys, pathlib, math, os, tempfile
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from distgpt.model.config import ModelConfig
from distgpt.model.transformer import GPT
from distgpt.training.optim import cosine_lr, build_optimizer
from distgpt.training.stability import SpikeMonitor


def test_param_count_scales():
    sm = ModelConfig(n_layer=24, n_head=16, n_kv_head=8, d_model=2048, d_ffn=5632)
    md = ModelConfig(n_layer=32, n_head=32, n_kv_head=8, d_model=4096, d_ffn=11008, tie_embeddings=False)
    lg = ModelConfig(n_layer=80, n_head=64, n_kv_head=8, d_model=8192, d_ffn=28672, tie_embeddings=False)
    a, b, c = sm.param_count(), md.param_count(), lg.param_count()
    assert a < b < c
    assert 8e8 < a < 2e9        # ~1B
    assert 5e9 < b < 1e10       # ~7B
    assert 5e10 < c < 1e11      # ~70B


def test_forward_backward():
    cfg = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=32)
    m = GPT(cfg).train()
    x = torch.randint(0, 128, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_activation_ckpt_runs():
    cfg = ModelConfig(vocab_size=128, n_layer=4, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=32)
    m = GPT(cfg, activation_ckpt="selective").train()
    x = torch.randint(0, 128, (2, 16))
    _, loss = m(x, x)
    loss.backward()


def test_qk_norm_forward_and_modules():
    cfg = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=32, qk_norm=True)
    m = GPT(cfg).train()
    # QK-norm adds per-head RMSNorm on q and k.
    assert m.layers[0].attn.q_norm is not None
    assert m.layers[0].attn.k_norm is not None
    x = torch.randint(0, 128, (2, 16))
    _, loss = m(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    # Default stays off.
    assert GPT(ModelConfig(vocab_size=128, n_layer=1, n_head=4, n_kv_head=2,
                           d_model=64, d_ffn=128)).layers[0].attn.q_norm is None


def test_zero_init_proj_starts_as_identity():
    cfg = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=32, zero_init_proj=True)
    m = GPT(cfg)
    for blk in m.layers:
        assert torch.count_nonzero(blk.attn.o_proj.weight) == 0
        assert torch.count_nonzero(blk.ffn.w2.weight) == 0
    # Identity init: residual stream unchanged at step 0 -> output == embeddings
    # passed through final norm + head. Just assert it runs and is finite.
    x = torch.randint(0, 128, (2, 16))
    _, loss = m.train()(x, x)
    assert torch.isfinite(loss)


def test_cosine_schedule():
    assert cosine_lr(0, 100, 1000, 1.0, 0.1) > 0
    assert math.isclose(cosine_lr(100, 100, 1000, 1.0, 0.1), 1.0, abs_tol=1e-9)
    assert math.isclose(cosine_lr(2000, 100, 1000, 1.0, 0.1), 0.1, abs_tol=1e-9)


def test_optimizer_groups():
    cfg = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ffn=128)
    m = GPT(cfg)
    o = build_optimizer(m, lr=1e-3, betas=[0.9, 0.95], weight_decay=0.1, fused=False)
    assert len(o.param_groups) == 2
    assert o.param_groups[0]["weight_decay"] == 0.1
    assert o.param_groups[1]["weight_decay"] == 0.0


def test_spike_monitor():
    m = SpikeMonitor(window=50, sigma=5.0)
    for _ in range(50):
        assert m.observe(2.0) is False
    assert m.observe(2.05) is False
    assert m.observe(50.0) is True


def test_streaming_loader_roundtrip():
    import numpy as np
    from distgpt.data.streaming import StreamingLoader
    with tempfile.TemporaryDirectory() as d:
        # write 2 small shards
        for i in range(2):
            np.arange(10000, dtype=np.uint16).tofile(os.path.join(d, f"s_{i}.bin"))
        ld = StreamingLoader(d, seq_len=32, micro_batch=4, rank=0, world_size=1, seed=0, device="cpu")
        x, y = ld.next_batch()
        assert x.shape == (4, 32) and y.shape == (4, 32)
        sd = ld.state_dict()
        ld2 = StreamingLoader(d, seq_len=32, micro_batch=4, rank=0, world_size=1, seed=0, device="cpu")
        ld2.load_state_dict(sd)
        assert ld2.state.cursor == ld.state.cursor
