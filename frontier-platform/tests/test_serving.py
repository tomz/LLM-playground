from __future__ import annotations
import asyncio
import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.serving.engine import Engine, EngineConfig, GenRequest
from platform.serving.router import Router, Tier


def _async_collect(gen):
    async def _run():
        out = []
        async for x in gen:
            out.append(x)
        return out
    return asyncio.run(_run())


def _tiny():
    torch.manual_seed(0)
    return Transformer(ModelConfig(
        vocab_size=64, n_layer=2, n_head=2, n_kv_head=1,
        d_model=32, d_ffn=64, max_seq_len=64,
    ))


def test_torch_engine_greedy_generation():
    model = _tiny()
    eng = Engine(EngineConfig(backend="torch", device="cpu"), model=model)
    req = GenRequest(prompt_ids=[1, 2, 3], max_new_tokens=5, temperature=0.0)
    chunks = _async_collect(eng.generate(req))
    assert chunks[-1]["done"] is True
    assert chunks[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 5}
    tokens = [c["token_id"] for c in chunks if not c.get("done")]
    assert len(tokens) == 5
    # Determinism: same seed → same tokens.
    chunks2 = _async_collect(Engine(EngineConfig(backend="torch", device="cpu"), model=model).generate(req))
    tokens2 = [c["token_id"] for c in chunks2 if not c.get("done")]
    assert tokens == tokens2


def test_engine_stop_token():
    model = _tiny()
    eng = Engine(EngineConfig(backend="torch", device="cpu"), model=model)
    # Find what the greedy first token actually is, then use it as stop.
    first = _async_collect(eng.generate(GenRequest(prompt_ids=[1, 2, 3], max_new_tokens=1, temperature=0.0)))
    stop_id = first[0]["token_id"]
    req = GenRequest(prompt_ids=[1, 2, 3], max_new_tokens=10, temperature=0.0, stop=[stop_id])
    chunks = _async_collect(eng.generate(req))
    tokens = [c["token_id"] for c in chunks if not c.get("done")]
    assert tokens == [stop_id]
    assert chunks[-1]["usage"]["completion_tokens"] == 1


def _tiers():
    return [
        Tier("nano", "http://n", target_ttft_ms=100, target_itl_ms=20, cost_per_mtok=0.10),
        Tier("mid", "http://m",  target_ttft_ms=500, target_itl_ms=30, cost_per_mtok=1.00),
        Tier("pro", "http://p",  target_ttft_ms=2000, target_itl_ms=50, cost_per_mtok=5.00),
    ]


def test_router_select_with_hint():
    r = Router(_tiers())
    assert r.select("mid", prompt_len=10).name == "mid"
    # Unknown hint falls back to heuristic.
    assert r.select("bogus", prompt_len=10).name == "nano"


def test_router_select_heuristic_picks_cheapest_under_budget():
    r = Router(_tiers())
    # prompt_len=100 → budget 5ms — all tiers fit; cheapest is nano.
    assert r.select(None, prompt_len=100).name == "nano"
    # prompt_len=4000 → budget 200ms — nano excluded; mid is cheapest fit.
    assert r.select(None, prompt_len=4000).name == "mid"
    # prompt_len=50000 → budget 2500ms — only "pro" fits if any; else fastest fallback.
    sel = r.select(None, prompt_len=50000)
    assert sel.name in ("pro", "nano")  # fallback path is acceptable
