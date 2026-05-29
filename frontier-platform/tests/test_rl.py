"""RLVR / GRPO skeleton tests (see platform/rl, docs/15-reasoning-rl-rlvr.md)."""
from __future__ import annotations
import math
from pathlib import Path

import pytest  # noqa: F401
import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
from platform.rl.grpo import GRPOConfig, group_advantages, grpo_step, run_grpo
from platform.rl.rollout import sample_group
from platform.rl.verifiers import (
    CodeUnitTestVerifier,
    MathExactVerifier,
    length_penalty,
    make_verifier,
    reward_contains,
    reward_regex,
)
from platform.tokenizer.bytes import BytesTokenizer


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=512, n_layer=2, n_head=4, n_kv_head=2,
        d_model=64, d_ffn=128, max_seq_len=64,
    )


def _save_base_ckpt(path: Path, cfg: ModelConfig) -> None:
    torch.manual_seed(0)
    m = Transformer(cfg)
    torch.save({"model": m.state_dict(), "model_cfg": cfg}, path)


# ---------- verifiers ----------

def test_reward_contains_and_regex():
    v = reward_contains("yes")
    assert v("p", "the answer is YES") == 1.0
    assert v("p", "no way") == 0.0
    r = reward_regex(r"\d{3}")
    assert r("p", "code 123 ok") == 1.0
    assert r("p", "no digits") == 0.0


def test_math_exact_verifier_parses_last_number():
    v = MathExactVerifier(42)
    assert v("2+40=?", "first 7 then finally 42") == 1.0
    assert v("2+40=?", "the answer is 41") == 0.0
    assert v("2+40=?", "no numbers here") == 0.0


def test_length_penalty_is_nonpositive():
    v = length_penalty(max_tokens=3, coef=0.1)
    assert v("p", "a b c") == 0.0
    assert v("p", "a b c d e") == pytest.approx(-0.2)


def test_make_verifier_registry_and_code_stub():
    assert make_verifier("contains", target="ok")("p", "ok") == 1.0
    code_v = make_verifier("code_tests", tests=["assert f(1)==1"])
    assert isinstance(code_v, CodeUnitTestVerifier)
    with pytest.raises(NotImplementedError):
        code_v("p", "def f(x): return x")


# ---------- group advantages ----------

def test_group_advantages_zero_mean_per_group():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    gidx = torch.tensor([0, 0, 0, 1, 1, 1])
    adv = group_advantages(rewards, gidx)
    # Each group's advantages must sum to ~0 (mean-centered).
    assert abs(float(adv[:3].sum())) < 1e-5
    assert abs(float(adv[3:].sum())) < 1e-5
    # The high-reward member of each group has positive advantage.
    assert adv[0] > 0
    assert adv[3] < 0  # the lone 0.0 in a group of {0,1,1}


def test_group_advantages_constant_group_is_finite():
    # All-equal rewards -> std 0 -> advantages must stay finite (eps guard).
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
    gidx = torch.tensor([0, 0, 0, 0])
    adv = group_advantages(rewards, gidx)
    assert torch.isfinite(adv).all()


# ---------- rollout ----------

def test_sample_group_shapes_and_mask():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg)
    tok = BytesTokenizer()
    prompts = [tok.encode("Q: a"), tok.encode("Q: b")]
    roll = sample_group(
        policy, prompts, group_size=3, max_new_tokens=5,
        seq_len=32, tokenizer=tok, temperature=1.0, seed=0,
    )
    assert roll.n_rows == 2 * 3
    assert roll.ids.shape[0] == 6
    assert roll.resp_mask.shape == roll.ids.shape
    # group_index labels the 2 prompts, 3 rows each.
    assert sorted(roll.group_index.tolist()) == [0, 0, 0, 1, 1, 1]
    # response mask only marks generated (non-prompt) positions.
    assert roll.resp_mask.sum() > 0
    assert len(roll.response_text) == 6


# ---------- grpo step + e2e ----------

def test_grpo_step_runs_and_metrics_finite():
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    from platform.alignment._common import clone_for_reference
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    prompts = [tok.encode("Q: x"), tok.encode("Q: y")]
    roll = sample_group(
        policy, prompts, group_size=4, max_new_tokens=5,
        seq_len=32, tokenizer=tok, seed=1,
    )
    rewards = torch.rand(roll.n_rows)
    gcfg = GRPOConfig(group_size=4, max_new_tokens=5, seq_len=32, beta=0.04)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    metrics = grpo_step(policy, ref, roll, rewards, gcfg, optimizer=opt)
    for k in ("loss", "pg_loss", "kl", "reward_mean"):
        assert math.isfinite(metrics[k])


def test_run_grpo_e2e_increases_verifier_reward(tmp_path):
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)

    # Reward any response containing the byte 'a' — trivially learnable signal
    # that exercises the full sample -> verify -> advantage -> update loop.
    verifier = reward_contains("a")
    gcfg = GRPOConfig(
        policy_ckpt=str(base),
        out_dir=str(tmp_path / "out_grpo"),
        group_size=4, steps=12, lr=5e-3, beta=0.0,
        max_new_tokens=6, seq_len=32, temperature=1.0,
    )
    out = run_grpo(gcfg, prompts=["Q: one", "Q: two"], verifier=verifier)
    assert Path(out).exists()
    state = torch.load(out, map_location="cpu", weights_only=False)
    hist = state["history"]
    assert len(hist) == 12
    for h in hist:
        assert math.isfinite(h["loss"])
    # With beta=0 and a learnable reward, mean reward should trend up.
    early = sum(h["reward_mean"] for h in hist[:3]) / 3
    late = sum(h["reward_mean"] for h in hist[-3:]) / 3
    assert late >= early
