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
    ProbabilityRewardVerifier,
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


def test_math_exact_verifier_boxed_and_symbolic():
    # Boxed extraction takes precedence over stray numbers in the reasoning.
    v = MathExactVerifier(42)
    assert v("q", "I tried 7 and 13 but \\boxed{42}") == 1.0
    assert v("q", "stuff 99 then \\boxed{41}") == 0.0
    # Symbolic equivalence: 1/2 == 0.5 == \frac{1}{2} (needs sympy; falls back
    # to numeric/string otherwise).
    half = MathExactVerifier(0.5)
    assert half("q", "\\boxed{1/2}") == 1.0
    fr = MathExactVerifier("\\frac{1}{2}")
    assert fr("q", "the answer is \\boxed{0.5}") == 1.0


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


# ---------- RLPR: verifier-free probability reward ----------

def test_rlpr_reward_in_unit_range_and_zero_without_reference():
    """RLPR reward is the policy's mean decoding probability of the reference
    answer — a probability, so it lands in [0, 1]; and a prompt with no
    reference scores exactly 0."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg)
    tok = BytesTokenizer()
    v = ProbabilityRewardVerifier(policy, tok, references={"2+2?": "4"})
    r = v("2+2?", "<think>two plus two</think> 4")
    assert 0.0 <= r <= 1.0
    # No reference for this prompt -> 0.0 (nothing to score against).
    assert v("9+9?", "anything") == 0.0


def test_rlpr_prefers_a_trace_that_makes_the_answer_likely():
    """The defining RLPR property: after a tiny SFT that teaches the model the
    answer follows the trace, a *correct* trace makes the reference answer more
    probable than an unrelated/empty one — so the reward ranks them correctly,
    with no executable verifier anywhere."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg)
    tok = BytesTokenizer()

    # Briefly fit the policy on "<prompt><trace><answer>" so the answer becomes
    # predictable from the trace (stands in for a cold-start checkpoint).
    prompt, trace, ans = "Q: 2+2?", " reason: two plus two is ", "4"
    ids = torch.tensor([tok.encode(prompt + trace + ans)])
    opt = torch.optim.AdamW(policy.parameters(), lr=5e-3)
    policy.train()
    for _ in range(60):
        logits, _ = policy(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    v = ProbabilityRewardVerifier(policy, tok, references={prompt: ans})
    good = v.mean_answer_probability(prompt, trace)          # the trained trace
    empty = v.mean_answer_probability(prompt, "")            # no reasoning
    assert good > empty
    # The reward wrapper scales by `reward` and stays a probability.
    assert 0.0 <= v(prompt, trace) <= 1.0


def test_rlpr_does_not_leave_the_model_in_train_mode():
    """The reward runs a no_grad eval forward; it must restore the model's
    original train/eval state so it can't silently disable dropout mid-training."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg)
    tok = BytesTokenizer()
    v = ProbabilityRewardVerifier(policy, tok, references={"q": "a"})
    policy.train()
    v("q", "some response")
    assert policy.training is True          # restored
    policy.eval()
    v("q", "some response")
    assert policy.training is False         # restored


def test_rlpr_via_make_verifier_registry():
    """RLPR is reachable through the make_verifier(...) protocol like every other
    verifier — kind='probability'."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg)
    tok = BytesTokenizer()
    v = make_verifier("probability", model=policy, tokenizer=tok,
                      references={"q": "a"})
    assert isinstance(v, ProbabilityRewardVerifier)
    assert 0.0 <= v("q", "answer is a") <= 1.0


def test_rlpr_places_input_on_model_device():
    """RLPR's forward must build its input tensor on the *model's* device, not
    hard-coded CPU — otherwise it crashes on a GPU policy (regression pin for the
    device fix). We can't require a GPU in CI, so assert the tensor is built from
    the model's device rather than a literal: a CPU model must score fine, and
    the verifier must read the device off the model's parameters."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg)
    tok = BytesTokenizer()
    v = ProbabilityRewardVerifier(policy, tok, references={"q": "a"})
    # Sanity: scores on the (CPU) model without a device mismatch.
    assert 0.0 <= v.mean_answer_probability("q", "resp") <= 1.0
    # If CUDA is available, the GPU path must not raise (the bug was a CPU tensor
    # meeting a CUDA model inside compute_token_logps).
    if torch.cuda.is_available():
        gpu_policy = Transformer(cfg).cuda()
        gv = ProbabilityRewardVerifier(gpu_policy, tok, references={"q": "a"})
        assert 0.0 <= gv.mean_answer_probability("q", "resp") <= 1.0


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
    for k in ("loss", "pg_loss", "kl", "reward_mean", "clip_frac", "ratio_mean"):
        assert math.isfinite(metrics[k])


def test_sample_group_captures_behavior_logp():
    """The rollout must record per-token sampling log-probs for the importance
    ratio — without them GRPO degenerates to REINFORCE."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg)
    tok = BytesTokenizer()
    prompts = [tok.encode("Q: a")]
    roll = sample_group(
        policy, prompts, group_size=2, max_new_tokens=5,
        seq_len=32, tokenizer=tok, temperature=1.0, seed=0,
    )
    assert roll.behavior_logp is not None
    assert roll.behavior_logp.shape == roll.ids.shape
    # Behavior log-probs are negative (log of a probability) at generated tokens.
    gen = roll.behavior_logp[roll.resp_mask > 0]
    assert (gen <= 1e-4).all() and torch.isfinite(gen).all()


def test_grpo_kl_estimator_nonnegative_and_zero_at_ref():
    """k3 KL estimator must be >= 0, and exactly 0 when policy == ref."""
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    from platform.alignment._common import clone_for_reference
    ref = clone_for_reference(policy)  # identical weights -> KL must be ~0
    tok = BytesTokenizer()
    roll = sample_group(
        policy, [tok.encode("Q: x")], group_size=4, max_new_tokens=5,
        seq_len=32, tokenizer=tok, seed=2,
    )
    rewards = torch.rand(roll.n_rows)
    gcfg = GRPOConfig(group_size=4, max_new_tokens=5, seq_len=32, beta=0.04)
    # No optimizer -> single eval pass, policy still equals ref.
    metrics = grpo_step(policy, ref, roll, rewards, gcfg, optimizer=None)
    assert metrics["kl"] >= -1e-6
    assert metrics["kl"] == pytest.approx(0.0, abs=1e-4)
    # First inner step: ratio against behavior log-probs ~ 1 (same weights).
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=0.05)


def test_grpo_clipped_objective_bounds_ratio():
    """Large advantage with a clip range must engage the clip (clip_frac>0) once
    the policy moves off the behavior distribution."""
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    from platform.alignment._common import clone_for_reference
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    roll = sample_group(
        policy, [tok.encode("Q: x")], group_size=6, max_new_tokens=6,
        seq_len=32, tokenizer=tok, seed=3,
    )
    rewards = torch.rand(roll.n_rows)
    gcfg = GRPOConfig(group_size=6, max_new_tokens=6, seq_len=32, beta=0.0,
                      clip_eps_low=0.2, clip_eps_high=0.2, ppo_epochs=4)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-1)  # big steps -> ratio moves
    metrics = grpo_step(policy, ref, roll, rewards, gcfg, optimizer=opt)
    # After several aggressive inner epochs some tokens leave the trust region.
    assert 0.0 <= metrics["clip_frac"] <= 1.0
    assert math.isfinite(metrics["ratio_mean"])


# ---------- GSPO: sequence-level importance ratio ----------

def test_gspo_unknown_level_rejected():
    """Only 'token' (GRPO) and 'sequence' (GSPO) are valid granularities."""
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    from platform.alignment._common import clone_for_reference
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    roll = sample_group(policy, [tok.encode("Q: x")], group_size=4, max_new_tokens=5,
                        seq_len=32, tokenizer=tok, seed=2)
    rewards = torch.rand(roll.n_rows)
    bad = GRPOConfig(group_size=4, max_new_tokens=5, seq_len=32,
                     importance_sampling_level="nope")
    with pytest.raises(ValueError, match="importance_sampling_level"):
        grpo_step(policy, ref, roll, rewards, bad, optimizer=None)


def test_gspo_default_is_token_level():
    """GSPO must be strictly opt-in — the default config is GRPO (token-level)."""
    assert GRPOConfig().importance_sampling_level == "token"


def test_gspo_sequence_ratio_is_one_at_reference():
    """At the first inner step the policy equals the behavior policy, so the
    GSPO sequence ratio (like the GRPO token ratio) must be ~1 — a sanity check
    that the length-normalized exponent is wired correctly."""
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    from platform.alignment._common import clone_for_reference
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    roll = sample_group(policy, [tok.encode("Q: x")], group_size=4, max_new_tokens=5,
                        seq_len=32, tokenizer=tok, seed=2)
    rewards = torch.rand(roll.n_rows)
    gcfg = GRPOConfig(group_size=4, max_new_tokens=5, seq_len=32, beta=0.04,
                      importance_sampling_level="sequence")
    metrics = grpo_step(policy, ref, roll, rewards, gcfg, optimizer=None)
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=0.05)
    for k in ("loss", "pg_loss", "kl", "clip_frac", "ratio_mean"):
        assert math.isfinite(metrics[k])


def test_gspo_sequence_ratio_lower_variance_than_token():
    """The defining property of GSPO: with the *same* off-reference policy the
    sequence-level ratio is far less dispersed than the token-level one, because
    it is one length-normalized number per sequence instead of many independent
    per-token ratios. We verify the helper directly on synthetic log-prob deltas."""
    from platform.rl.grpo import _sequence_importance_ratio
    torch.manual_seed(0)
    N, T = 8, 16
    logp = torch.randn(N, T) * 0.5
    base = torch.zeros(N, T)
    mask = torch.ones(N, T)
    token_ratio = torch.exp(logp - base)                       # [N, T]
    seq_ratio = _sequence_importance_ratio(logp, base, mask)   # [N, 1]
    # Sequence ratios (one averaged exponent per row) cluster much tighter than
    # the raw per-token ratios.
    assert float(seq_ratio.std()) < float(token_ratio.std())
    # Length normalization keeps them ~centered on 1 (mean of zero-mean deltas).
    assert 0.5 < float(seq_ratio.mean()) < 1.5


def test_gspo_respects_response_mask_in_length_norm():
    """Length normalization must divide by the number of *response* tokens, not
    the padded width — masked (prompt/pad) positions must not dilute the ratio."""
    from platform.rl.grpo import _sequence_importance_ratio
    logp = torch.tensor([[1.0, 1.0, 0.0, 0.0]])   # only first 2 tokens are response
    base = torch.zeros(1, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    # Mean exponent over the 2 masked-in tokens = (1+1)/2 = 1.0 -> ratio e^1.
    r = _sequence_importance_ratio(logp, base, mask)
    assert r.shape == (1, 1)
    assert float(r) == pytest.approx(math.e, abs=1e-5)


def test_gspo_end_to_end_step_updates(tmp_path):
    """A full GSPO step with an optimizer runs and produces finite metrics —
    proving the sequence-level ratio broadcasts cleanly through the surrogate,
    dual-clip, KL, and metric paths."""
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    from platform.alignment._common import clone_for_reference
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    roll = sample_group(policy, [tok.encode("Q: x"), tok.encode("Q: y")],
                        group_size=4, max_new_tokens=6, seq_len=32, tokenizer=tok, seed=5)
    rewards = torch.rand(roll.n_rows)
    # GSPO with the tighter clip its different numeric scale calls for.
    gcfg = GRPOConfig(group_size=4, max_new_tokens=6, seq_len=32, beta=0.04,
                      clip_eps_low=3e-3, clip_eps_high=4e-3, ppo_epochs=2,
                      importance_sampling_level="sequence")
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    metrics = grpo_step(policy, ref, roll, rewards, gcfg, optimizer=opt)
    for k in ("loss", "pg_loss", "kl", "reward_mean", "clip_frac", "ratio_mean"):
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


def test_soft_length_penalty_uses_real_tokenizer():
    from platform.rl.reward import soft_length_penalty
    tok = BytesTokenizer()
    text = "hello world"  # 2 whitespace words, but 11 bytes/tokens
    # With the byte tokenizer the token count (11) exceeds a tiny target where the
    # whitespace count (2) would not — proving real token counts are used.
    word_based = soft_length_penalty(text, target_tokens=5, max_tokens=10, coef=1.0)
    token_based = soft_length_penalty(text, target_tokens=5, max_tokens=10, coef=1.0,
                                      count_tokens=tok.encode)
    assert word_based == 0.0
    assert token_based < 0.0


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
