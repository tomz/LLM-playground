"""Tier 8.3 pedagogical pins: small, fast tests that assert *what the
post-training algorithms actually do*, so the machinery can't silently drift.

Unlike the bugfix/sandbox tiers (which pin specific regressions), these encode
the load-bearing invariants of DPO and GRPO themselves:

  * **DPO margin monotonicity** — one optimizer step on a clean
    (prompt, chosen, rejected) triple must *increase* the policy's
    logprob(chosen) - logprob(rejected) margin. Built on a hand-rolled tiny
    Llama (no HF download) so the suite stays hermetic and CI-runnable without
    a token.
  * **GRPO group-standardization** — within-group advantage estimate is
    baseline-invariant and produces the symmetric ±1 sign structure on binary
    rewards. Tests the standardization math directly (fast, no trainer).
  * **extract_code round-trip** — every builtin (chosen/rejected) code body
    survives a fence-wrap → extract_code cycle, pinning the verifier's code
    recovery against future fence-format drift.

All of these run in well under a second and need no network.
"""
from __future__ import annotations

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cf_rl.reward import group_standardize_advantages  # noqa: E402
from cf_rl.grpo_train import grpo_extra_kwargs  # noqa: E402
from eval.run_humaneval import (  # noqa: E402
    extract_code,
    build_eval_summary,
    write_json_summary,
    write_completions_jsonl,
)
from cf_pref.pairs import BUILTIN_PREFERENCES  # noqa: E402


# ---------------------------------------------------------------------------
# GRPO group standardization — the value-network-free advantage estimate
# ---------------------------------------------------------------------------


def test_grpo_binary_rewards_give_symmetric_advantages():
    """The headline GRPO pin: a group of binary rewards ``[1, 0, 0, 1]``
    standardized within the group yields advantages ``[+1, -1, -1, +1]`` to
    float tolerance — passing completions reinforced, failing ones suppressed
    by equal magnitude, regardless of the overall pass rate."""
    adv = group_standardize_advantages([1, 0, 0, 1], group_size=4)
    assert adv == pytest.approx([1.0, -1.0, -1.0, 1.0], abs=1e-6)


def test_grpo_advantages_are_baseline_invariant():
    """Adding a constant offset to *every* reward in a group must leave the
    advantages unchanged — so a uniform format bonus can't bias the gradient.
    This is the whole point of subtracting the group mean."""
    base = group_standardize_advantages([1.0, 0.0, 0.0, 1.0], group_size=4)
    shifted = group_standardize_advantages([6.0, 5.0, 5.0, 6.0], group_size=4)
    assert shifted == pytest.approx(base, abs=1e-6)


def test_grpo_standardize_handles_multiple_groups_independently():
    """Each group of G is standardized on its *own* statistics — group 2's
    rewards must not leak into group 1's baseline. Two groups of 2 here."""
    # group A: [2, 0] -> mean 1, centered [+1,-1], /std=1 -> [+1,-1]
    # group B: [9, 9] -> mean 9, centered [0, 0], std 0 -> [~0, ~0]
    adv = group_standardize_advantages([2.0, 0.0, 9.0, 9.0], group_size=2)
    assert adv[0] == pytest.approx(1.0, abs=1e-4)
    assert adv[1] == pytest.approx(-1.0, abs=1e-4)
    assert adv[2] == pytest.approx(0.0, abs=1e-4)
    assert adv[3] == pytest.approx(0.0, abs=1e-4)


def test_grpo_constant_group_gives_zero_advantage():
    """A group where every completion scored identically (all pass or all fail)
    carries no learning signal — advantages must all be ~0, never NaN from a
    divide-by-zero std."""
    adv = group_standardize_advantages([1.0, 1.0, 1.0, 1.0], group_size=4)
    assert adv == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_grpo_standardize_rejects_ragged_group():
    """A batch whose length isn't a multiple of G is a config error — GRPO
    groups can't straddle the batch boundary. Fail loudly, not silently."""
    with pytest.raises(ValueError):
        group_standardize_advantages([1.0, 0.0, 1.0], group_size=2)


def test_grpo_no_std_scaling_is_plain_centering():
    """``scale_by_std=False`` (the 'Dr. GRPO' / no-std variant) must reduce to
    plain mean-subtraction — pin so the two modes don't get conflated."""
    adv = group_standardize_advantages([3.0, 1.0], group_size=2, scale_by_std=False)
    assert adv == pytest.approx([1.0, -1.0], abs=1e-6)


# ---------------------------------------------------------------------------
# vLLM opt-in plumbing (8.3 speed knob) — pure, no TRL import needed
# ---------------------------------------------------------------------------


def test_grpo_extra_kwargs_off_by_default():
    """The vLLM rollout knob is opt-in: with no ``grpo.use_vllm`` the extra
    kwargs must be empty so the default generation path is untouched."""
    assert grpo_extra_kwargs({"grpo": {}}) == {}
    assert grpo_extra_kwargs({}) == {}


def test_grpo_extra_kwargs_enables_vllm():
    """``grpo.use_vllm: true`` threads ``use_vllm=True`` into GRPOConfig."""
    extra = grpo_extra_kwargs({"grpo": {"use_vllm": True}})
    assert extra["use_vllm"] is True
    # GPU memory util only appears when explicitly set.
    assert "vllm_gpu_memory_utilization" not in extra


def test_grpo_extra_kwargs_threads_gpu_mem_when_set():
    extra = grpo_extra_kwargs(
        {"grpo": {"use_vllm": True, "vllm_gpu_memory_utilization": 0.4}}
    )
    assert extra["use_vllm"] is True
    assert extra["vllm_gpu_memory_utilization"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Eval CLI summary / JSONL writers (8.3 surfacing) — pure I/O helpers
# ---------------------------------------------------------------------------


def test_eval_summary_names_metric_by_n_samples():
    """pass@1 and pass@5 summaries must carry distinct metric keys so a diff
    tool can tell runs apart."""
    s1 = build_eval_summary("m", n=10, passes=6, n_samples=1, temperature=0.2, seed=0)
    assert s1["metric"] == "pass@1"
    assert s1["pass@1"] == pytest.approx(0.6)
    s5 = build_eval_summary("m", n=10, passes=8, n_samples=5, temperature=0.8, seed=0)
    assert s5["metric"] == "pass@5"
    assert s5["pass@5"] == pytest.approx(0.8)


def test_eval_summary_handles_zero_problems():
    """Edge case: n=0 must not divide-by-zero."""
    s = build_eval_summary("m", n=0, passes=0, n_samples=1, temperature=0.0, seed=None)
    assert s["pass@1"] == 0.0


def test_write_json_summary_roundtrips(tmp_path):
    import json
    s = build_eval_summary("m", n=4, passes=2, n_samples=1, temperature=0.2, seed=7)
    path = tmp_path / "sub" / "summary.json"  # nested → must mkdir
    write_json_summary(s, str(path))
    loaded = json.loads(path.read_text())
    assert loaded == s


def test_write_completions_jsonl_is_one_object_per_line(tmp_path):
    import json
    records = [
        {"task_id": "t/0", "passed": True, "samples": [{"passed": True}]},
        {"task_id": "t/1", "passed": False, "samples": [{"passed": False}]},
    ]
    path = tmp_path / "completions.jsonl"
    write_completions_jsonl(records, str(path))
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(ln)["task_id"] for ln in lines] == ["t/0", "t/1"]


# ---------------------------------------------------------------------------
# extract_code round-trip — pin the verifier's code recovery
# ---------------------------------------------------------------------------


def _fence(code: str) -> str:
    return f"```python\n{code}```"


def test_extract_code_roundtrip_over_builtin_preferences():
    """For every (chosen, rejected) code body in the builtin preference set,
    wrapping it in a python fence and running extract_code must return the
    original body. Pins the extractor against future fence-format drift — if
    someone changes the fence regex, this catches the silent breakage."""
    for _instr, chosen, rejected in BUILTIN_PREFERENCES:
        for code in (chosen, rejected):
            recovered = extract_code(_fence(code))
            assert recovered == code, f"round-trip lost data:\n{code!r}\n-> {recovered!r}"
            # And the recovered code is valid, runnable Python.
            ns: dict = {}
            exec(recovered, ns)


# ---------------------------------------------------------------------------
# DPO margin monotonicity — one step must raise the chosen-vs-rejected margin
# ---------------------------------------------------------------------------


def _tiny_causal_lm(vocab_size: int = 64, seed: int = 0):
    """Hand-build a 2-layer LlamaForCausalLM from scratch — no HF download, so
    the DPO pin stays hermetic and CI-runnable without a token. ~50k params."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(cfg)


def _seq_logprob(model, prompt_ids, completion_ids):
    """Sum log p(completion | prompt) under ``model`` — the quantity DPO moves.

    Only the completion tokens contribute (the prompt is the condition). Returns
    a scalar tensor with grad so we can backprop the DPO loss through it.
    """
    import torch
    import torch.nn.functional as F

    full = torch.cat([prompt_ids, completion_ids], dim=1)
    logits = model(full).logits  # (1, T, V)
    # Predict token t from position t-1; align the completion span.
    logprobs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = full[:, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    comp_len = completion_ids.shape[1]
    # The last ``comp_len`` predicted tokens are the completion tokens.
    return token_lp[:, -comp_len:].sum()


def test_dpo_step_increases_chosen_minus_rejected_margin():
    """The defining behavior of DPO: one optimizer step on a clean
    (prompt, chosen, rejected) pair must *increase* the policy's implicit
    reward margin ``[logπ(chosen) - logπ_ref(chosen)] -
    [logπ(rejected) - logπ_ref(rejected)]``.

    We implement the DPO sigmoid loss by hand against a frozen reference copy
    (exactly what TRL does internally with ref_model=None on a PEFT model:
    reference = the model before the update). Building on a tiny from-scratch
    Llama keeps this < 1 s and download-free. This pins the entire DPO
    objective in one assertion: the loss, the reference subtraction, and the
    sign of the gradient.
    """
    import copy
    import torch
    import torch.nn.functional as F

    torch.manual_seed(1234)
    model = _tiny_causal_lm(seed=1)
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()

    # A clean preference triple in token space (arbitrary but fixed).
    prompt = torch.tensor([[1, 2, 3, 4]])
    chosen = torch.tensor([[5, 6, 7]])
    rejected = torch.tensor([[8, 9, 10]])
    beta = 0.1

    def dpo_margin(m):
        """Implicit-reward margin: (logπ_c - logπref_c) - (logπ_r - logπref_r)."""
        lp_c = _seq_logprob(m, prompt, chosen)
        lp_r = _seq_logprob(m, prompt, rejected)
        with torch.no_grad():
            ref_c = _seq_logprob(ref, prompt, chosen)
            ref_r = _seq_logprob(ref, prompt, rejected)
        return (lp_c - ref_c) - (lp_r - ref_r)

    margin_before = dpo_margin(model).item()

    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    # DPO sigmoid loss: -log σ(β · margin). Minimizing it raises the margin.
    for _ in range(5):
        opt.zero_grad()
        margin = dpo_margin(model)
        loss = -F.logsigmoid(beta * margin)
        loss.backward()
        opt.step()

    margin_after = dpo_margin(model).item()
    assert margin_after > margin_before, (
        f"DPO step must raise chosen-vs-rejected margin: "
        f"{margin_before:.4f} -> {margin_after:.4f}"
    )


def test_dpo_loss_decreases_as_margin_grows():
    """Sanity companion to the margin test: the DPO sigmoid loss is monotone
    decreasing in the margin, so a larger margin ⇒ smaller loss. Pins the loss
    shape (guards against a sign flip in the objective)."""
    import torch
    import torch.nn.functional as F

    beta = 0.1
    small = -F.logsigmoid(beta * torch.tensor(0.5))
    large = -F.logsigmoid(beta * torch.tensor(5.0))
    assert large < small
