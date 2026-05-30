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


def test_make_verifier_registry_and_code_verifier():
    assert make_verifier("contains", target="ok")("p", "ok") == 1.0
    code_v = make_verifier("code_tests", tests=["assert f(1) == 1"])
    assert isinstance(code_v, CodeUnitTestVerifier)
    # Correct solution passes the hidden test; wrong one fails. Runs in sandbox.
    good = code_v("p", "def f(x):\n    return x")
    bad = code_v("p", "def f(x):\n    return x + 1")
    assert good == 1.0 and bad == 0.0


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


# ---------- reward shaping ----------

def test_format_reward_rewards_think_answer_structure():
    from platform.rl.reward import format_reward
    full = format_reward("<think>reasoning</think>\\boxed{42}")
    partial = format_reward("<think>reasoning</think> 42")
    none = format_reward("just an answer 42")
    assert full > partial > none
    assert none == 0.0


def test_soft_length_penalty_ramps():
    from platform.rl.reward import soft_length_penalty
    short = "w " * 10
    long = "w " * 5000
    assert soft_length_penalty(short, target_tokens=512) == 0.0
    assert soft_length_penalty(long, target_tokens=512, max_tokens=2048, coef=0.1) == pytest.approx(-0.1)


def test_repetition_penalty_catches_loops():
    from platform.rl.reward import repetition_penalty
    varied = "the quick brown fox jumps over the lazy dog today"
    looped = "go go go go go go go go go go"
    assert repetition_penalty(varied) == pytest.approx(0.0, abs=0.2) or repetition_penalty(varied) > -0.2
    assert repetition_penalty(looped) < repetition_penalty(varied)


def test_answer_spam_guard_penalizes_shotgun():
    from platform.rl.reward import answer_spam_guard
    ok = "\\boxed{42}"
    spam = "\\boxed{1}\\boxed{2}\\boxed{3}\\boxed{4}\\boxed{5}"
    assert answer_spam_guard(ok, max_candidates=3) == 0.0
    assert answer_spam_guard(spam, max_candidates=3, coef=1.0) == -1.0


def test_composite_reward_blends_and_clips():
    from platform.rl.reward import CompositeReward, RewardConfig
    base = MathExactVerifier(42)
    comp = CompositeReward(base, RewardConfig())
    bd = comp.breakdown("2+40=?", "<think>2+40</think>\\boxed{42}")
    assert bd["correctness"] == 1.0
    assert bd["format"] > 0.0
    assert bd["total"] == comp("2+40=?", "<think>2+40</think>\\boxed{42}")
    # Bounded by clip range.
    lo, hi = RewardConfig().clip
    assert lo <= bd["total"] <= hi


def test_grpo_logs_reward_breakdown(tmp_path):
    from platform.rl.reward import CompositeReward
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    comp = CompositeReward(reward_contains("a"))
    gcfg = GRPOConfig(
        policy_ckpt=str(base), out_dir=str(tmp_path / "out"),
        group_size=4, steps=3, lr=1e-3, beta=0.0, max_new_tokens=5, seq_len=32,
    )
    out = run_grpo(gcfg, prompts=["Q: one"], verifier=comp)
    hist = torch.load(out, map_location="cpu", weights_only=False)["history"]
    # Composite reward exposes .breakdown -> components logged each step.
    assert all("reward_correctness" in h and "reward_total" in h for h in hist)


# ---------- cold-start reasoning-SFT ----------

def test_format_trace_builds_reasoning_format():
    from platform.rl.coldstart import format_trace
    ex = format_trace("2+2?", "add two and two", "4")
    assert ex["prompt"] == "2+2?"
    assert "<think>" in ex["response"] and "\\boxed{4}" in ex["response"]


def test_run_coldstart_reduces_loss_and_saves(tmp_path):
    from platform.rl.coldstart import ColdStartConfig, format_trace, run_coldstart
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    examples = [
        format_trace("2+2?", "two plus two", "4"),
        format_trace("3+5?", "three plus five", "8"),
    ] * 4
    res = run_coldstart(
        ColdStartConfig(policy_ckpt=str(base), out_dir=str(tmp_path / "cs"),
                        epochs=6, lr=5e-3, batch_size=2, seq_len=64),
        examples,
    )
    assert Path(res.out_path).exists()
    assert len(res.loss_history) > 0
    early = sum(res.loss_history[:2]) / 2
    late = sum(res.loss_history[-2:]) / 2
    assert late < early
    # Cold-start output loads as a GRPO policy checkpoint.
    state = torch.load(res.out_path, map_location="cpu", weights_only=False)
    assert "model" in state and "model_cfg" in state


def test_coldstart_then_grpo_pipeline(tmp_path):
    """End-to-end: reasoning-SFT cold-start -> GRPO loads its checkpoint."""
    from platform.rl.coldstart import ColdStartConfig, format_trace, run_coldstart
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    cs = run_coldstart(
        ColdStartConfig(policy_ckpt=str(base), out_dir=str(tmp_path / "cs"),
                        epochs=2, lr=1e-3, batch_size=2, seq_len=64),
        [format_trace("2+2?", "two plus two", "4")] * 4,
    )
    gcfg = GRPOConfig(
        policy_ckpt=cs.out_path, out_dir=str(tmp_path / "grpo"),
        group_size=4, steps=3, lr=1e-3, beta=0.0, max_new_tokens=5, seq_len=64,
    )
    out = run_grpo(gcfg, prompts=["2+2?"], verifier=reward_contains("4"))
    assert Path(out).exists()
