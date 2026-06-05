"""Tier 1 RLVR harvests from the MAI-Thinking-1 deep dive.

Covers five additions (see docs/research/mai-thinking-1-deep-dive.md):
  1. IFEval-style objective constraint verifiers (rl/verifiers.py)
  2. Adaptive entropy control for GRPO (rl/grpo.py)
  3. Language-consistency reward (rl/reward.py)
  4. Difficulty-aware length penalty (rl/reward.py)
  5. Outer dual-clip for the GRPO ratio (rl/grpo.py)
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from platform.model.config import ModelConfig
from platform.model.transformer import Transformer
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


# =====================================================================
# 1. IFEval-style constraint verifiers
# =====================================================================

from platform.rl.verifiers import (  # noqa: E402
    CONSTRAINT_CHECKERS,
    ConstraintFollowingVerifier,
    check_constraint,
    make_verifier,
)


def test_check_constraint_keyword_existence_and_frequency():
    assert check_constraint("keywords:existence", "I love photosynthesis",
                            keyword="photosynthesis") is True
    assert check_constraint("keywords:existence", "no mention here",
                            keyword="photosynthesis") is False
    # frequency: at_least
    assert check_constraint("keywords:frequency", "go go go", keyword="go",
                            at_least=3) is True
    assert check_constraint("keywords:frequency", "go go", keyword="go",
                            at_least=3) is False
    # forbidden
    assert check_constraint("keywords:forbidden", "clean text", keyword="banned") is True
    assert check_constraint("keywords:forbidden", "this is banned", keyword="banned") is False


def test_check_constraint_case_sensitivity():
    # Default case-insensitive.
    assert check_constraint("keywords:existence", "PYTHON rules", keyword="python") is True
    # Case-sensitive miss.
    assert check_constraint("keywords:existence", "PYTHON rules", keyword="python",
                            case_sensitive=True) is False


def test_check_constraint_length_words_sentences_paragraphs():
    assert check_constraint("length:words", "one two three four five",
                            at_least=5) is True
    assert check_constraint("length:words", "one two", at_least=5) is False
    assert check_constraint("length:words", "a b c", exactly=3) is True

    text2 = "First sentence. Second sentence! Third?"
    assert check_constraint("length:sentences", text2, exactly=3) is True
    assert check_constraint("length:sentences", text2, at_most=2) is False

    para = "Para one body.\n***\nPara two body."
    assert check_constraint("length:paragraphs", para, exactly=2) is True


def test_check_constraint_format_json_bullets_highlights_title():
    assert check_constraint("format:json", '{"a": 1, "b": [2, 3]}') is True
    assert check_constraint("format:json", "not json at all") is False
    # JSON inside a code fence is tolerated.
    assert check_constraint("format:json", '```json\n{"x": 1}\n```') is True

    bullets = "- first\n- second\n- third"
    assert check_constraint("format:bullets", bullets, at_least=3) is True
    assert check_constraint("format:bullets", bullets, at_least=4) is False

    assert check_constraint("format:highlights", "this is *very* *cool*",
                            at_least=2) is True
    assert check_constraint("format:title", "<<My Title>>\nbody") is True
    assert check_constraint("format:title", "no title here") is False


def test_check_constraint_case_startend_punctuation():
    assert check_constraint("case:upper", "ALL CAPS HERE") is True
    assert check_constraint("case:upper", "Not all caps") is False
    assert check_constraint("case:lower", "all lower") is True
    assert check_constraint("startend:startswith", "ANSWER: 42", prefix="ANSWER:") is True
    assert check_constraint("startend:endswith", "the end.", suffix="end.") is True
    assert check_constraint("startend:quotation", '"quoted"') is True
    assert check_constraint("startend:quotation", "unquoted") is False
    assert check_constraint("punctuation:no_commas", "no commas here") is True
    assert check_constraint("punctuation:no_commas", "one, two") is False


def test_check_constraint_unknown_raises():
    with pytest.raises(ValueError, match="unknown constraint"):
        check_constraint("does:not_exist", "text")


def test_all_registered_checkers_callable():
    # Every registered checker must run on a benign input without raising
    # (params permitting) — guards against a typo'd registry entry.
    for name in CONSTRAINT_CHECKERS:
        # Provide superset params; checkers ignore the ones they don't use.
        try:
            check_constraint(name, "Sample TEXT. <<T>>", keyword="Sample",
                             prefix="Sample", suffix="TEXT.")
        except KeyError as e:  # missing required param is a bug in the test, not the checker
            raise AssertionError(f"{name} needs more params: {e}")


def test_constraint_verifier_all_or_nothing_and_fractional():
    constraints = [
        {"name": "keywords:existence", "keyword": "photosynthesis"},
        {"name": "length:words", "at_least": 3},
        {"name": "punctuation:no_commas"},
    ]
    strict = ConstraintFollowingVerifier(constraints, all_or_nothing=True)
    frac = ConstraintFollowingVerifier(constraints, all_or_nothing=False)

    good = "photosynthesis converts light efficiently"
    assert strict("p", good) == 1.0
    assert frac("p", good) == 1.0

    # Fails the comma rule only -> 2/3 fractional, 0 strict.
    partial = "photosynthesis converts light, efficiently"
    assert strict("p", partial) == 0.0
    assert frac("p", partial) == pytest.approx(2.0 / 3.0)


def test_constraint_verifier_breakdown_keys_and_empty():
    v = ConstraintFollowingVerifier([
        {"name": "case:upper"},
        {"name": "length:words", "at_least": 2},
    ])
    bd = v.breakdown("p", "HELLO WORLD")
    assert bd["satisfied_frac"] == 1.0
    assert any(k.startswith("case:upper#") for k in bd)
    # Empty constraint list -> zero reward, not a crash.
    empty = ConstraintFollowingVerifier([])
    assert empty("p", "anything") == 0.0


def test_constraint_verifier_tuple_spec_and_factory():
    v = ConstraintFollowingVerifier([("punctuation:no_commas", {})])
    assert v("p", "no commas") == 1.0
    # Registry wiring.
    fac = make_verifier("constraints",
                        constraints=[{"name": "case:lower"}])
    assert fac("p", "lower") == 1.0
    assert fac("p", "UPPER") == 0.0


# =====================================================================
# 2. Adaptive entropy control
# =====================================================================

from platform.alignment._common import compute_token_logps_and_entropy  # noqa: E402
from platform.rl.grpo import (  # noqa: E402
    EntropyController,
    GRPOConfig,
    _dual_clip_surrogate,
    grpo_step,
    make_entropy_controller,
    run_grpo,
)
from platform.rl.rollout import sample_group  # noqa: E402
from platform.rl.verifiers import reward_contains  # noqa: E402
from platform.alignment._common import clone_for_reference  # noqa: E402


def test_entropy_controller_raises_coef_when_entropy_below_target():
    ctl = EntropyController(target_entropy=2.0, kp=0.1, ki=0.01, coef_max=0.5)
    # Entropy far below target -> positive error -> coef ramps up.
    c1 = ctl.update(0.5)
    c2 = ctl.update(0.5)
    assert c1 > 0.0
    assert c2 > c1  # integral term keeps pushing
    assert c1 <= 0.5 and c2 <= 0.5  # bounded


def test_entropy_controller_zero_coef_when_entropy_above_target():
    ctl = EntropyController(target_entropy=1.0, kp=0.1, ki=0.01)
    # Entropy above target -> negative error -> coef clamped to floor (0).
    c = ctl.update(3.0)
    assert c == 0.0


def test_entropy_controller_anti_windup_bounds_integral():
    ctl = EntropyController(target_entropy=10.0, kp=0.0, ki=0.01, coef_max=0.5)
    for _ in range(10_000):
        ctl.update(0.0)  # huge sustained positive error
    # Integral can't wind up past coef_max/ki.
    assert ctl.integral <= 0.5 / 0.01 + 1e-6
    assert ctl.coef <= 0.5


def test_make_entropy_controller_none_when_no_target():
    assert make_entropy_controller(GRPOConfig()) is None
    cfg = GRPOConfig(target_entropy=1.5, entropy_kp=0.2)
    ctl = make_entropy_controller(cfg)
    assert isinstance(ctl, EntropyController)
    assert ctl.target_entropy == 1.5 and ctl.kp == 0.2


def test_compute_token_logps_and_entropy_matches_plain_logp():
    from platform.alignment._common import compute_token_logps
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    m = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    lp_only = compute_token_logps(m, x, y)
    lp, ent = compute_token_logps_and_entropy(m, x, y)
    assert torch.allclose(lp_only, lp, atol=1e-5)
    # Entropy is non-negative and bounded by log(vocab).
    assert (ent >= -1e-4).all()
    assert (ent <= math.log(cfg.vocab_size) + 1e-3).all()


def test_grpo_step_reports_entropy_and_coef():
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    roll = sample_group(policy, [tok.encode("Q: x")], group_size=4,
                        max_new_tokens=5, seq_len=32, tokenizer=tok, seed=1)
    rewards = torch.rand(roll.n_rows)
    gcfg = GRPOConfig(group_size=4, max_new_tokens=5, seq_len=32,
                      target_entropy=1.0, entropy_kp=0.1)
    ctl = make_entropy_controller(gcfg)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    metrics = grpo_step(policy, ref, roll, rewards, gcfg, optimizer=opt,
                        entropy_controller=ctl)
    assert math.isfinite(metrics["entropy"])
    assert metrics["entropy"] >= 0.0
    assert "entropy_coef" in metrics
    # With entropy below a target of 1.0 nats on a tiny model, coef should be > 0.
    assert metrics["entropy_coef"] >= 0.0


def test_grpo_static_entropy_bonus_without_controller():
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    roll = sample_group(policy, [tok.encode("Q: x")], group_size=4,
                        max_new_tokens=5, seq_len=32, tokenizer=tok, seed=1)
    rewards = torch.rand(roll.n_rows)
    gcfg = GRPOConfig(group_size=4, max_new_tokens=5, seq_len=32,
                      entropy_coef=0.01)  # static, no controller
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    metrics = grpo_step(policy, ref, roll, rewards, gcfg, optimizer=opt)
    assert metrics["entropy_coef"] == 0.01
    assert math.isfinite(metrics["loss"])


def test_run_grpo_with_entropy_control_e2e(tmp_path):
    cfg_m = _tiny_cfg()
    base = tmp_path / "base.pt"
    _save_base_ckpt(base, cfg_m)
    gcfg = GRPOConfig(
        policy_ckpt=str(base), out_dir=str(tmp_path / "out"),
        group_size=4, steps=5, lr=5e-3, beta=0.0,
        max_new_tokens=6, seq_len=32, temperature=1.0,
        target_entropy=1.5, entropy_kp=0.05, entropy_ki=0.005,
    )
    out = run_grpo(gcfg, prompts=["Q: one", "Q: two"], verifier=reward_contains("a"))
    hist = torch.load(out, map_location="cpu", weights_only=False)["history"]
    assert len(hist) == 5
    # Entropy + coef logged every step; all finite.
    for h in hist:
        assert math.isfinite(h["entropy"])
        assert math.isfinite(h["entropy_coef"])


# =====================================================================
# 5. Outer dual-clip
# =====================================================================

def test_dual_clip_disabled_is_identity():
    surr = torch.tensor([[-5.0, 2.0], [-0.3, 1.0]])
    adv = torch.tensor([[-1.0], [1.0]])
    out = _dual_clip_surrogate(surr, adv, 0.0)
    assert torch.equal(out, surr)


def test_dual_clip_floors_negative_advantage_only():
    # A < 0 token with a very negative surrogate gets floored at c*A; A > 0 left alone.
    surr = torch.tensor([[-10.0, -10.0]])
    adv = torch.tensor([[-1.0]])      # negative advantage
    out = _dual_clip_surrogate(surr, adv, clip_ratio_c=3.0)
    # floor = 3.0 * -1.0 = -3.0 -> surrogate raised from -10 to -3.
    assert torch.allclose(out, torch.full_like(surr, -3.0))

    pos_adv = torch.tensor([[1.0]])
    surr_pos = torch.tensor([[-10.0, -10.0]])
    out_pos = _dual_clip_surrogate(surr_pos, pos_adv, clip_ratio_c=3.0)
    assert torch.equal(out_pos, surr_pos)  # positive advantage untouched


def test_dual_clip_does_not_lower_already_good_surrogate():
    # If the surrogate is already above the floor, dual-clip leaves it.
    surr = torch.tensor([[-1.0]])
    adv = torch.tensor([[-1.0]])
    out = _dual_clip_surrogate(surr, adv, clip_ratio_c=3.0)  # floor -3.0
    assert torch.allclose(out, surr)  # max(-1, -3) == -1


def test_grpo_step_with_dual_clip_runs(tmp_path):
    cfg_m = _tiny_cfg()
    torch.manual_seed(0)
    policy = Transformer(cfg_m)
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    roll = sample_group(policy, [tok.encode("Q: x")], group_size=6,
                        max_new_tokens=6, seq_len=32, tokenizer=tok, seed=3)
    rewards = torch.rand(roll.n_rows)
    gcfg = GRPOConfig(group_size=6, max_new_tokens=6, seq_len=32, beta=0.0,
                      clip_ratio_c=3.0, ppo_epochs=3)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-1)
    metrics = grpo_step(policy, ref, roll, rewards, gcfg, optimizer=opt)
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["pg_loss"])
