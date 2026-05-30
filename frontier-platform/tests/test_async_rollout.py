"""Async actor-learner rollout tests (platform/rl/async_rollout.py)."""
from __future__ import annotations

import asyncio
import math
from pathlib import Path

import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.rl.async_rollout import AsyncRolloutConfig, AsyncRolloutEngine, RolloutBuffer
from platform.rl.grpo import GRPOConfig, run_grpo_async
from platform.rl.verifiers import reward_contains
from platform.tokenizer.bytes import BytesTokenizer


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=512, n_layer=2, n_head=4, n_kv_head=2,
                       d_model=64, d_ffn=128, max_seq_len=64)


def _save_base(path: Path, cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    torch.save({"model": Transformer(cfg).state_dict(), "model_cfg": cfg}, path)


def test_async_engine_generates_group_shapes():
    torch.manual_seed(0)
    model = Transformer(_tiny_cfg())
    tok = BytesTokenizer()
    eng = AsyncRolloutEngine(model, tok, AsyncRolloutConfig(
        group_size=3, max_new_tokens=5, seq_len=32, device="cpu"))
    prompts = [tok.encode("Q: a"), tok.encode("Q: b")]
    roll = eng.generate_group(prompts)
    assert roll.n_rows == 2 * 3
    assert sorted(roll.group_index.tolist()) == [0, 0, 0, 1, 1, 1]
    assert roll.resp_mask.shape == roll.ids.shape
    assert roll.resp_mask.sum() > 0
    assert len(roll.response_text) == 6


def test_async_engine_concurrent_matches_request_count():
    """All G×B rollouts complete (concurrency cap doesn't drop any)."""
    torch.manual_seed(0)
    model = Transformer(_tiny_cfg())
    tok = BytesTokenizer()
    eng = AsyncRolloutEngine(model, tok, AsyncRolloutConfig(
        group_size=4, max_new_tokens=4, seq_len=32, device="cpu", max_concurrency=2))
    prompts = [tok.encode("x"), tok.encode("y"), tok.encode("z")]
    roll = asyncio.run(eng.generate_group_async(prompts))
    assert roll.n_rows == 3 * 4


def test_sync_weights_bumps_version():
    model = Transformer(_tiny_cfg())
    tok = BytesTokenizer()
    eng = AsyncRolloutEngine(model, tok, AsyncRolloutConfig(device="cpu"))
    assert eng.weight_version == 0
    eng.sync_weights()
    assert eng.weight_version == 1


def test_rollout_buffer_put_get():
    async def _run():
        buf = RolloutBuffer(maxsize=2)
        await buf.put({"a": 1})
        await buf.put({"a": 2})
        assert buf.qsize() == 2
        first = await buf.get()
        return first
    first = asyncio.run(_run())
    assert first == {"a": 1}


def test_run_grpo_async_e2e(tmp_path):
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base(base, cfg_m)
    gcfg = GRPOConfig(
        policy_ckpt=str(base), out_dir=str(tmp_path / "out"),
        group_size=4, steps=5, lr=5e-3, beta=0.0, max_new_tokens=6, seq_len=32,
    )
    out = run_grpo_async(gcfg, prompts=["Q: one", "Q: two"], verifier=reward_contains("a"))
    assert Path(out).exists()
    hist = torch.load(out, map_location="cpu", weights_only=False)["history"]
    assert len(hist) == 5
    for h in hist:
        assert math.isfinite(h["loss"])
        assert "weight_version" in h
    # weight sync advanced after each step's update; recorded version is the
    # pre-sync value, so step k logs version k-1 (last of 5 steps -> 4).
    assert hist[-1]["weight_version"] == 4
    assert hist[0]["weight_version"] == 0
