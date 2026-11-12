"""KV-cached generation must match full-recompute generation exactly."""
from __future__ import annotations
import sys, time
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GPT, GPTConfig

ARCHS = [dict(), dict(pos_kind="rope", norm_kind="rmsnorm", mlp_kind="swiglu", qk_norm=True)]


def _m(block=16, **kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=97, block_size=block, n_layer=2, n_head=2, d_model=32, d_ffn=64, **kw)).eval()


@pytest.mark.parametrize("arch", ARCHS)
def test_greedy_identical_including_window_overflow(arch):
    m = _m(**arch)
    prompt = torch.randint(0, 97, (2, 5))
    with torch.no_grad():
        ref = m.generate(prompt, 30, temperature=0)
    assert torch.equal(m.generate_cached(prompt, 30, temperature=0), ref)


@pytest.mark.parametrize("arch", ARCHS)
def test_seeded_sampling_identical(arch):
    m = _m(**arch)
    prompt = torch.randint(0, 97, (1, 4))
    torch.manual_seed(7)
    with torch.no_grad():
        ref = m.generate(prompt, 10, temperature=0.9, top_k=20, top_p=0.9)
    torch.manual_seed(7)
    assert torch.equal(m.generate_cached(prompt, 10, temperature=0.9, top_k=20, top_p=0.9), ref)


def test_cache_is_faster_for_long_decode():
    m = _m(block=512, pos_kind="rope", norm_kind="rmsnorm")
    prompt = torch.randint(0, 97, (1, 8))
    with torch.no_grad():
        t = time.perf_counter(); m.generate(prompt, 200, temperature=0); full = time.perf_counter() - t
        t = time.perf_counter(); m.generate_cached(prompt, 200, temperature=0); cached = time.perf_counter() - t
    assert cached < full
