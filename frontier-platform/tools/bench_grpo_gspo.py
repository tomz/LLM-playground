"""Hardware measurement: GRPO (token-level) vs GSPO (sequence-level) + RLPR.

Three things this harness measures on a real GPU, on a small **MoE** policy
(GSPO's actual target — Qwen reported the sequence-level ratio matters most for
MoE-RL, where per-token ratios are noisiest because routing makes them jump):

  1. **Importance-ratio variance — GRPO vs GSPO.** On the *same* rollouts and
     the *same* off-reference policy, the per-token GRPO ratio vs the GSPO
     length-normalized sequence ratio. The paper's central claim is that the
     sequence ratio is far lower-variance; we measure the std and the resulting
     clip fraction directly.

  2. **Training stability — GRPO vs GSPO.** Two otherwise-identical GRPO runs on
     a learnable verifier task (reward = response contains a target token),
     differing only in `importance_sampling_level`. We log reward, clip_frac,
     and ratio variance per step and report which trajectory is steadier.

  3. **RLPR drives reward up.** A verifier-free run where the reward is the
     policy's own mean decoding probability of a reference answer
     (`ProbabilityRewardVerifier`) — no executable checker — and we confirm the
     verified-token reward climbs.

Runs on CPU (small/slow) or one GPU (`--device cuda`). Writes a JSON summary and
prints a table. This is a *measurement*, not a new code path: it drives the
shipped `grpo_step` / `ProbabilityRewardVerifier` unchanged.

    python tools/bench_grpo_gspo.py --device cuda --steps 40 --moe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The local package is literally named `platform`, which shadows stdlib
# `platform`. torch imports stdlib platform at load time, so evict any stdlib
# `platform` modules first, import torch (binds the real stdlib platform), then
# let our local `platform` package resolve for the platform.* imports below —
# the same dance tests/conftest.py does.
for _m in [k for k in list(sys.modules) if k == "platform" or k.startswith("platform.")]:
    del sys.modules[_m]
_orig_path = list(sys.path)
sys.path[:] = [p for p in sys.path if p not in ("", ".", str(ROOT))]
import torch  # noqa: E402  (binds torch to stdlib platform)
sys.path[:] = [str(ROOT)] + _orig_path
for _m in [k for k in list(sys.modules) if k == "platform" or k.startswith("platform.")]:
    del sys.modules[_m]

from platform.alignment._common import clone_for_reference, compute_token_logps  # noqa: E402
from platform.model.config import ModelConfig  # noqa: E402
from platform.model.transformer import Transformer  # noqa: E402
from platform.rl.grpo import (  # noqa: E402
    GRPOConfig,
    _sequence_importance_ratio,
    grpo_step,
)
from platform.rl.rollout import sample_group  # noqa: E402
from platform.rl.verifiers import ProbabilityRewardVerifier, reward_contains  # noqa: E402
from platform.tokenizer.bytes import BytesTokenizer  # noqa: E402


def _make_policy(moe: bool, device: str, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    kw = dict(vocab_size=512, n_layer=3, n_head=4, n_kv_head=2,
              d_model=128, d_ffn=256, max_seq_len=128)
    if moe:
        # Fine-grained MoE — the regime where token-level ratios are noisiest.
        kw.update(moe_num_experts=8, moe_top_k=2, d_ffn=256)
    cfg = ModelConfig(**kw)
    return Transformer(cfg).to(device)


def _to_device(roll, device: str):
    """Move a GroupRollout's tensors to ``device`` (the sampler builds on CPU)."""
    for attr in ("ids", "resp_mask", "group_index", "behavior_logp"):
        v = getattr(roll, attr, None)
        if isinstance(v, torch.Tensor):
            setattr(roll, attr, v.to(device))
    return roll


# --------------------------------------------------------------------------- #
# 1. Ratio-variance comparison on identical rollouts.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_ratio_variance(policy, ref, roll, *, k_sigma: float = 2.0) -> dict:
    """On the SAME off-reference policy + rollouts, compare the dispersion of the
    GRPO token-level ratio vs the GSPO sequence-level ratio, and the clip
    fraction each incurs at a clip range scaled to its *own* spread (±k_sigma·σ
    around 1) — so the clip_frac reflects the ratio's shape, not an arbitrary
    fixed threshold that one scale saturates."""
    x, y, m = roll.ids[:, :-1], roll.ids[:, 1:], roll.resp_mask[:, 1:]
    logp = compute_token_logps(policy, x, y)
    base = roll.behavior_logp[:, 1:] if roll.behavior_logp is not None else logp
    # GRPO: per-token ratio.
    tok_ratio = torch.exp(logp - base)
    msel = m > 0
    tok_vals = tok_ratio[msel]
    # GSPO: one length-normalized ratio per sequence.
    seq_ratio = _sequence_importance_ratio(logp, base, m).squeeze(1)  # [N]
    tok_std = float(tok_vals.std())
    seq_std = float(seq_ratio.std())
    # Clip each at ±k_sigma of its own spread → a fair, scale-matched clip_frac.
    tok_eps = k_sigma * tok_std
    seq_eps = k_sigma * seq_std
    tok_clip = (((tok_ratio < 1 - tok_eps) | (tok_ratio > 1 + tok_eps)).float() * m).sum() / m.sum().clamp_min(1)
    seq_clip = ((seq_ratio < 1 - seq_eps) | (seq_ratio > 1 + seq_eps)).float().mean()
    return {
        "token_ratio_std": tok_std,
        "token_ratio_mean": float(tok_vals.mean()),
        "seq_ratio_std": seq_std,
        "seq_ratio_mean": float(seq_ratio.mean()),
        "token_clip_frac": float(tok_clip),
        "seq_clip_frac": float(seq_clip),
        "variance_reduction": tok_std / max(seq_std, 1e-9),
    }


# --------------------------------------------------------------------------- #
# 2. Training-stability comparison: GRPO vs GSPO, identical except level.
# --------------------------------------------------------------------------- #
def run_arm(level: str, *, moe: bool, device: str, steps: int, group_size: int,
            target: str = "a", seed: int = 0) -> dict:
    policy = _make_policy(moe, device, seed=seed)
    ref = clone_for_reference(policy)
    tok = BytesTokenizer()
    prompts = ["Q: alpha", "Q: beta"]
    prompt_ids = [tok.encode(p) for p in prompts]
    verifier = reward_contains(target)

    # GSPO ratios are length-normalized so they sit on a tighter scale than the
    # per-token ratios; its clip should be correspondingly tighter. (At
    # production MoE scale Qwen uses ~3e-3 vs GRPO's 0.2 — ~2 orders tighter;
    # on this toy the sequence-ratio std is ~0.3 around 1, so we scale the clip
    # proportionally rather than hard-coding the production value.)
    if level == "sequence":
        cfg = GRPOConfig(group_size=group_size, max_new_tokens=12, seq_len=64,
                         beta=0.02, lr=5e-4, clip_eps_low=0.05, clip_eps_high=0.06,
                         ppo_epochs=2, importance_sampling_level="sequence")
    else:
        cfg = GRPOConfig(group_size=group_size, max_new_tokens=12, seq_len=64,
                         beta=0.02, lr=5e-4, clip_eps_low=0.2, clip_eps_high=0.2,
                         ppo_epochs=2, importance_sampling_level="token")
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    history = []
    for step in range(steps):
        roll = sample_group(policy, prompt_ids, group_size=group_size,
                            max_new_tokens=cfg.max_new_tokens, seq_len=cfg.seq_len,
                            tokenizer=tok, temperature=1.0, seed=step)
        roll = _to_device(roll, device)
        pairs = [(prompts[int(gi)], txt) for gi, txt in zip(roll.group_index, roll.response_text)]
        rewards = torch.tensor([verifier(p, r) for p, r in pairs],
                               dtype=torch.float32, device=device)
        m = grpo_step(policy, ref, roll, rewards, cfg, optimizer=opt)
        history.append({"step": step, "reward": m["reward_mean"],
                        "clip_frac": m["clip_frac"], "ratio_mean": m["ratio_mean"],
                        "loss": m["loss"], "kl": m["kl"]})
    rewards = [h["reward"] for h in history]
    clips = [h["clip_frac"] for h in history]
    return {
        "level": level,
        "reward_first3": sum(rewards[:3]) / 3,
        "reward_last3": sum(rewards[-3:]) / 3,
        "reward_delta": sum(rewards[-3:]) / 3 - sum(rewards[:3]) / 3,
        "mean_clip_frac": sum(clips) / len(clips),
        "history": history,
    }


# --------------------------------------------------------------------------- #
# 3. RLPR: verifier-free probability reward drives reward up.
# --------------------------------------------------------------------------- #
def run_rlpr(*, device: str, steps: int, group_size: int, seed: int = 0) -> dict:
    policy = _make_policy(False, device, seed=seed)
    tok = BytesTokenizer()
    # Short, single-token-ish reference answers. RLPR (like all RLVR) needs the
    # answer to be *reachable* — a random-init policy that never emits the target
    # has zero advantage variance to learn from. So we warm-start with a few SFT
    # steps on "<prompt><answer>" (exactly how RLPR is used in practice: SFT →
    # RL), then let the verifier-free RLPR reward sharpen it.
    references = {"Q: best letter?": "z", "Q: a digit?": "7"}
    prompts = list(references)
    prompt_ids = [tok.encode(p) for p in prompts]
    contains = {p: reward_contains(a) for p, a in references.items()}

    # --- brief SFT warmup so the answer is reachable (verifier-free RL needs a
    #     non-degenerate starting policy, not a cold random init) ---
    warm_opt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    policy.train()
    for _ in range(30):
        for p, a in references.items():
            seq = torch.tensor([tok.encode(p + a)], dtype=torch.long, device=device)
            logits, _ = policy(seq[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), seq[:, 1:].reshape(-1))
            warm_opt.zero_grad(); loss.backward(); warm_opt.step()

    ref = clone_for_reference(policy)
    verifier = ProbabilityRewardVerifier(policy, tok, references)
    # beta>0 (KL-to-reference) anchors the RL update to the warm-started policy.
    # With beta=0 the verifier-free reward is easy to hack — the policy drifts
    # off the SFT solution and collapses. A small KL keeps it near the reachable
    # answer while RLPR sharpens it (this is the standard RLVR recipe, not a
    # special case). lr kept gentle for the same reason.
    cfg = GRPOConfig(group_size=group_size, max_new_tokens=8, seq_len=48,
                     beta=0.1, lr=3e-4, importance_sampling_level="token")
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    history = []
    for step in range(steps):
        roll = sample_group(policy, prompt_ids, group_size=group_size,
                            max_new_tokens=cfg.max_new_tokens, seq_len=cfg.seq_len,
                            tokenizer=tok, temperature=1.0, seed=step)
        pairs = [(prompts[int(gi)], txt) for gi, txt in zip(roll.group_index, roll.response_text)]
        # RLPR reward: policy's own mean decoding prob of the reference answer.
        rewards = torch.tensor([verifier(p, r) for p, r in pairs],
                               dtype=torch.float32, device=device)
        # Cross-check (not used for the gradient): emit-rate of the actual answer.
        emit = sum(contains[p](p, r) for p, r in pairs) / len(pairs)
        m = grpo_step(policy, ref, roll, rewards, cfg, optimizer=opt)
        history.append({"step": step, "reward": m["reward_mean"], "emit_rate": emit,
                        "loss": m["loss"]})
    rewards = [h["reward"] for h in history]
    emits = [h["emit_rate"] for h in history]
    return {
        "reward_first3": sum(rewards[:3]) / 3,
        "reward_last3": sum(rewards[-3:]) / 3,
        "reward_delta": sum(rewards[-3:]) / 3 - sum(rewards[:3]) / 3,
        "emit_first3": sum(emits[:3]) / 3,
        "emit_last3": sum(emits[-3:]) / 3,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--moe", action="store_true", help="use an MoE policy (GSPO's target regime)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="seeds to average the ratio-variance measure over")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    print(f"[bench] device={device}  steps={args.steps}  group_size={args.group_size}  moe={args.moe}")
    t0 = time.perf_counter()
    tok = BytesTokenizer()

    # --- 1. ratio variance on one shared rollout, policy nudged off-reference ---
    # Average over a few seeds so the headline number isn't a single-draw fluke.
    var_runs = []
    for s in range(args.seeds):
        p = _make_policy(args.moe, device, seed=args.seed + s)
        r = clone_for_reference(p)
        pids = [tok.encode("Q: alpha"), tok.encode("Q: beta")]
        rl = sample_group(p, pids, group_size=args.group_size, max_new_tokens=12,
                          seq_len=64, tokenizer=tok, temperature=1.0, seed=123 + s)
        rl = _to_device(rl, device)
        o = torch.optim.AdamW(p.parameters(), lr=5e-4)
        xx, yy, mm = rl.ids[:, :-1], rl.ids[:, 1:], rl.resp_mask[:, 1:]
        for _ in range(2):
            lp = compute_token_logps(p, xx, yy)
            o.zero_grad(); (-(lp * mm).sum()).backward(); o.step()
        var_runs.append(measure_ratio_variance(p, r, rl))
    var = {k: sum(d[k] for d in var_runs) / len(var_runs) for k in var_runs[0]}

    # --- 2. GRPO vs GSPO training stability ---
    grpo = run_arm("token", moe=args.moe, device=device, steps=args.steps,
                   group_size=args.group_size, seed=args.seed)
    gspo = run_arm("sequence", moe=args.moe, device=device, steps=args.steps,
                   group_size=args.group_size, seed=args.seed)

    # --- 3. RLPR ---
    rlpr = run_rlpr(device=device, steps=args.steps, group_size=args.group_size, seed=args.seed)

    dt = time.perf_counter() - t0

    print("\n  (1) importance-ratio dispersion on identical rollouts (off-reference policy)")
    print(f"      GRPO token-level ratio:  std={var['token_ratio_std']:.3f}  "
          f"mean={var['token_ratio_mean']:.3f}")
    print(f"      GSPO sequence-level:     std={var['seq_ratio_std']:.3f}  "
          f"mean={var['seq_ratio_mean']:.3f}")
    print(f"      → sequence ratio is {var['variance_reduction']:.1f}× lower-variance "
          f"than the token ratio (the GSPO stability claim)")

    print("\n  (2) GRPO vs GSPO training stability (reward should rise, clip stay sane)")
    print(f"      {'arm':<10}{'reward Δ':>12}{'mean clip_frac':>18}")
    print(f"      {'GRPO':<10}{grpo['reward_delta']:>+12.3f}{grpo['mean_clip_frac']:>18.3f}")
    print(f"      {'GSPO':<10}{gspo['reward_delta']:>+12.3f}{gspo['mean_clip_frac']:>18.3f}")

    print("\n  (3) RLPR verifier-free reward (policy's own prob of the reference answer)")
    print(f"      mean answer-prob {rlpr['reward_first3']:.4f} → {rlpr['reward_last3']:.4f}  "
          f"(Δ {rlpr['reward_delta']:+.4f})")
    print(f"      cross-check emit-rate {rlpr['emit_first3']:.2f} → {rlpr['emit_last3']:.2f}  "
          f"(did the policy actually start emitting the answer?)")
    print(f"\n  wall: {dt:.1f}s")

    summary = {"device": device, "steps": args.steps, "moe": args.moe,
               "ratio_variance": var, "grpo": {k: v for k, v in grpo.items() if k != "history"},
               "gspo": {k: v for k, v in gspo.items() if k != "history"},
               "rlpr": {k: v for k, v in rlpr.items() if k != "history"},
               "wall_s": dt}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({**summary, "grpo_history": grpo["history"],
                       "gspo_history": gspo["history"], "rlpr_history": rlpr["history"]}, f, indent=2)
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
