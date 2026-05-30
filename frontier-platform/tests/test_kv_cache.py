"""Incremental KV-cache decode correctness (GQA + MLA).

The decode path (prefill + per-token steps over a KVCache) must produce the same
next-token logits as the training-style full re-encode forward. This guards the
serving fast-path against drift from the reference math.
"""
from __future__ import annotations

import torch

from platform.model.config import ModelConfig
from platform.model.kv_cache import KVCache
from platform.model.transformer import Transformer


def _check_cache_matches_full(cfg: ModelConfig, atol: float = 1e-4):
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    steps = [torch.randint(0, cfg.vocab_size, (1, 1)) for _ in range(4)]

    # --- reference: full re-encode each step ---
    seq = prompt.clone()
    ref_last_logits = []
    with torch.no_grad():
        logits, _ = model(seq)
        ref_last_logits.append(logits[:, -1, :])
        for s in steps:
            seq = torch.cat([seq, s], dim=1)
            logits, _ = model(seq)
            ref_last_logits.append(logits[:, -1, :])

    # --- cached: prefill then one token at a time ---
    cache = KVCache(cfg.n_layer)
    cached_last_logits = []
    logits = model.forward_with_cache(prompt, cache)
    cached_last_logits.append(logits[:, -1, :])
    for s in steps:
        logits = model.forward_with_cache(s, cache)
        cached_last_logits.append(logits[:, -1, :])

    for i, (r, c) in enumerate(zip(ref_last_logits, cached_last_logits)):
        diff = (r - c).abs().max().item()
        assert diff < atol, f"step {i}: max logit diff {diff} >= {atol}"


def test_kv_cache_decode_matches_full_gqa():
    cfg = ModelConfig(vocab_size=128, n_layer=3, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=64, attn_kind="gqa")
    _check_cache_matches_full(cfg)


def test_kv_cache_decode_matches_full_mla():
    cfg = ModelConfig(vocab_size=128, n_layer=3, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=64,
                      attn_kind="mla", mla_kv_latent_dim=48, mla_rope_head_dim=8)
    _check_cache_matches_full(cfg)


def test_kv_cache_decode_matches_full_gqa_qknorm():
    cfg = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=4,
                      d_model=64, d_ffn=128, max_seq_len=64, qk_norm=True)
    _check_cache_matches_full(cfg)


def test_mla_cache_stores_compressed_latent_not_full_kv():
    cfg = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=64,
                      attn_kind="mla", mla_kv_latent_dim=48, mla_rope_head_dim=8)
    model = Transformer(cfg).eval()
    cache = KVCache(cfg.n_layer)
    model.forward_with_cache(torch.randint(0, 128, (1, 5)), cache)
    lc = cache.layers[0]
    # MLA caches the latent c_kv + rope key, NOT per-head k/v.
    assert "c_kv" in lc.data and "k_rope" in lc.data
    assert "k" not in lc.data and "v" not in lc.data
    # Latent width is mla_kv_latent_dim, far below n_head*head_dim.
    assert lc.data["c_kv"].shape[-1] == cfg.mla_kv_latent_dim
    assert cache.pos == 5


def test_cache_reset():
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=1,
                      d_model=32, d_ffn=64, max_seq_len=32)
    model = Transformer(cfg).eval()
    cache = KVCache(cfg.n_layer)
    model.forward_with_cache(torch.randint(0, 64, (1, 4)), cache)
    assert cache.pos == 4
    cache.reset()
    assert cache.pos == 0 and cache.layers[0].data == {}


def test_serving_engine_cached_greedy_matches_reference():
    """The TorchEngine KV-cache decode path must greedily generate the same
    tokens as a manual full-re-encode greedy loop (CPU, device override)."""
    import asyncio

    from platform.serving.engine import Engine, EngineConfig, GenRequest

    cfg = ModelConfig(vocab_size=128, n_layer=3, n_head=4, n_kv_head=2,
                      d_model=64, d_ffn=128, max_seq_len=64, attn_kind="mla",
                      mla_kv_latent_dim=48, mla_rope_head_dim=8)
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    prompt_ids = [1, 5, 9, 13]

    # Reference greedy via full re-encode.
    seq = torch.tensor([prompt_ids])
    ref = []
    with torch.no_grad():
        for _ in range(6):
            logits, _ = model(seq)
            nxt = int(logits[:, -1, :].argmax(-1).item())
            ref.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)

    eng = Engine(EngineConfig(backend="torch", device="cpu"), model=model)

    async def _run():
        out = []
        async for ch in eng.generate(GenRequest(prompt_ids=prompt_ids, max_new_tokens=6, temperature=0.0)):
            if not ch.get("done"):
                out.append(ch["token_id"])
        return out

    got = asyncio.run(_run())
    assert got == ref
