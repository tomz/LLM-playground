"""GRPO — Group-Relative Policy Optimization (DeepSeekMath / DeepSeek-R1).

The key idea vs. PPO (see `platform.alignment.ppo`): drop the value network.
For each prompt, sample a group of G responses, score each with a *verifier*
(see `platform.rl.verifiers`), and use the group's mean/std to form a baseline:

    advantage_i = (r_i - mean(group)) / (std(group) + eps)

The per-token objective is the PPO-style **clipped** surrogate with a
group-relative advantage broadcast to every generated token, minus a
per-token KL penalty to the reference policy:

    ratio_t   = exp(logp_theta(t) - logp_behavior(t))
    surr_t    = min(ratio_t * A, clip(ratio_t, 1-eps_lo, 1+eps_hi) * A)
    kl_t      = exp(logp_ref - logp_theta) - (logp_ref - logp_theta) - 1   # k3, >=0
    loss      = -mean_t(surr_t) + beta * mean_t(kl_t)

This is the production GRPO objective: per-token importance ratios against the
**behavior** policy that generated the rollout (so it is correct under async
actor–learner skew, where the sampling policy lags the learner), Schulman's k3
unbiased KL estimator (always >= 0, low variance), and decoupled lower/upper
clip ranges (DAPO-style ``clip_higher``). When a rollout carries no behavior
log-probs (legacy/greedy), the ratio degenerates to ``exp(logp - logp.detach())
== 1`` on the first inner step, recovering the REINFORCE surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..alignment._common import (
    clone_for_reference,
    compute_token_logps,
    compute_token_logps_and_entropy,
)
from ..model.config import ModelConfig
from ..model.transformer import Transformer
from ..tokenizer.bytes import BytesTokenizer
from .rollout import GroupRollout, sample_group
from .verifiers import Verifier


@dataclass
class GRPOConfig:
    policy_ckpt: str = ""
    ref_ckpt: str = ""                 # defaults to a frozen copy of the policy
    out_dir: str = "out/grpo"
    group_size: int = 4               # G rollouts per prompt
    steps: int = 10
    lr: float = 1e-5
    beta: float = 0.04                # KL-to-reference coefficient
    clip_eps_low: float = 0.2         # PPO lower clip (1 - eps_low)
    clip_eps_high: float = 0.2        # PPO upper clip (1 + eps_high); DAPO raises this
    clip_ratio_c: float = 0.0         # outer "dual-clip" bound (0 disables); see grpo_step
    ppo_epochs: int = 1               # inner optimization passes per rollout batch
    max_new_tokens: int = 16
    seq_len: int = 256
    temperature: float = 1.0
    grad_clip: float = 1.0
    # --- entropy control (averts entropy collapse during RLVR) ---
    entropy_coef: float = 0.0         # static entropy-bonus coefficient (used when
                                      # target_entropy is None); 0.0 = no bonus
    target_entropy: float | None = None  # H*; when set, a PI controller adapts the
                                      # entropy coef each step to hold entropy ~ H*
    entropy_kp: float = 0.05          # PI proportional gain
    entropy_ki: float = 0.005         # PI integral gain
    entropy_coef_max: float = 0.5     # upper bound on the (adapted) entropy coef
    model_cfg: ModelConfig | None = None


@dataclass
class EntropyController:
    """PI controller that adapts the GRPO entropy-bonus coefficient to hold the
    policy's mean token entropy near a target ``H*``.

    Entropy collapse — the policy sharpening into a near-deterministic mode and
    losing the exploration RLVR needs — is a classic GRPO failure. A fixed
    entropy bonus is hard to tune: too small and entropy still collapses, too
    large and it destabilises the policy. Instead we *measure* mean token entropy
    each step and drive a coefficient with a proportional-integral controller so
    entropy tracks ``H*``:

        error      = H* - measured_entropy        # >0 when entropy too LOW
        integral  += error                        # (anti-windup bounded)
        coef       = clip(kp*error + ki*integral, 0, coef_max)

    The coefficient multiplies an entropy *bonus* subtracted from the loss, so a
    higher coef pushes entropy up. Anti-windup bounds the integral so a long
    saturation can't cause overshoot.
    """

    target_entropy: float
    coef: float = 0.0
    kp: float = 0.05
    ki: float = 0.005
    coef_min: float = 0.0
    coef_max: float = 0.5
    integral: float = 0.0

    def update(self, measured_entropy: float) -> float:
        error = self.target_entropy - measured_entropy
        self.integral += error
        if self.ki > 0:  # anti-windup: bound the integral's contribution
            cap = self.coef_max / self.ki
            self.integral = max(-cap, min(cap, self.integral))
        raw = self.kp * error + self.ki * self.integral
        self.coef = float(min(self.coef_max, max(self.coef_min, raw)))
        return self.coef


def make_entropy_controller(cfg: GRPOConfig) -> EntropyController | None:
    """Build an :class:`EntropyController` from ``cfg`` iff adaptive entropy
    control is requested (``cfg.target_entropy is not None``)."""
    if cfg.target_entropy is None:
        return None
    return EntropyController(
        target_entropy=cfg.target_entropy,
        coef=cfg.entropy_coef,
        kp=cfg.entropy_kp,
        ki=cfg.entropy_ki,
        coef_max=cfg.entropy_coef_max,
    )


def group_advantages(rewards: torch.Tensor, group_index: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative, standardized advantages.

    ``rewards`` and ``group_index`` are ``[N]``. Within each group (prompt), the
    advantage is ``(r - group_mean) / (group_std + eps)``. Returns ``[N]``.
    """
    adv = torch.zeros_like(rewards)
    for gi in group_index.unique():
        sel = group_index == gi
        g = rewards[sel]
        adv[sel] = (g - g.mean()) / (g.std(unbiased=False) + eps)
    return adv


def _shift(ids: torch.Tensor, mask: torch.Tensor):
    return ids[:, :-1], ids[:, 1:], mask[:, 1:]


def _dual_clip_surrogate(surrogate: torch.Tensor, adv_tok: torch.Tensor,
                         clip_ratio_c: float) -> torch.Tensor:
    """DAPO / dual-clip-PPO outer clip.

    On **negative-advantage** tokens the inner ``min(ratio*A, clip(ratio)*A)``
    can still go arbitrarily negative when the importance ratio explodes
    (``ratio*A -> -inf`` as ``ratio -> inf`` for ``A < 0``), which makes
    ``-surrogate`` (the loss) blow up. Lower-bounding the surrogate by
    ``clip_ratio_c * A`` (a fixed negative floor, since ``A < 0``) caps it.
    Positive-advantage tokens are untouched. ``clip_ratio_c <= 0`` disables it.
    """
    if not clip_ratio_c or clip_ratio_c <= 0:
        return surrogate
    floor = clip_ratio_c * adv_tok                 # negative where A < 0
    neg = (adv_tok < 0).expand_as(surrogate)
    return torch.where(neg, torch.maximum(surrogate, floor), surrogate)


def grpo_step(
    policy,
    ref,
    roll: GroupRollout,
    rewards: torch.Tensor,
    cfg: GRPOConfig,
    *,
    optimizer=None,
    entropy_controller: "EntropyController | None" = None,
) -> dict:
    """One GRPO update over a group rollout (real per-token clipped objective).

    Computes group-relative advantages, then for ``cfg.ppo_epochs`` inner passes
    applies the PPO-clipped surrogate with a per-token importance ratio against
    the rollout's behavior log-probs and a k3 KL penalty to ``ref``. Returns a
    metrics dict. When ``optimizer`` is None, runs a single forward to populate
    metrics without updating (advantages still computed).

    Two stability mechanisms layer on the base objective:

    * **Outer "dual" clip** (``cfg.clip_ratio_c > 0``, DAPO/dual-clip-PPO): on
      tokens with *negative* advantage the standard lower clip can still let an
      exploding ratio produce a large positive surrogate; bounding the surrogate
      below by ``clip_ratio_c * A`` caps that. Disabled at 0.

    * **Adaptive entropy bonus** (``entropy_controller`` set): an entropy term
      ``-coef * mean_entropy`` is added to the loss, with ``coef`` driven by a PI
      controller toward ``cfg.target_entropy`` to avert entropy collapse. With no
      controller, a static ``cfg.entropy_coef`` bonus is used (0 = off)."""
    adv_seq = group_advantages(rewards, roll.group_index)        # [N]
    x, y, m = _shift(roll.ids, roll.resp_mask)                   # [N, T-1]

    # Behavior (sampling) log-probs aligned to the target positions y.
    if roll.behavior_logp is not None:
        old_logp = roll.behavior_logp[:, 1:].detach()            # [N, T-1]
    else:
        old_logp = None

    # Per-token advantage = broadcast group advantage over generated tokens.
    adv_tok = adv_seq.unsqueeze(1).detach()                      # [N, 1]
    tok_count = m.sum().clamp_min(1.0)

    use_entropy = entropy_controller is not None or cfg.entropy_coef != 0.0
    n_epochs = max(1, cfg.ppo_epochs) if optimizer is not None else 1
    last_metrics: dict = {}
    for _ in range(n_epochs):
        policy.train()
        if use_entropy:
            logp, ent_tok = compute_token_logps_and_entropy(policy, x, y)  # [N, T-1]
        else:
            logp = compute_token_logps(policy, x, y)            # [N, T-1], grad
            ent_tok = None
        with torch.no_grad():
            ref_logp = compute_token_logps(ref, x, y)           # [N, T-1]

        base = old_logp if old_logp is not None else logp.detach()
        ratio = torch.exp(logp - base)                          # [N, T-1]
        unclipped = ratio * adv_tok
        clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high) * adv_tok
        surrogate = torch.minimum(unclipped, clipped)           # [N, T-1]
        # Outer dual-clip: lower-bound the surrogate by clip_ratio_c * A on
        # negative-advantage tokens (where a blown-up ratio would otherwise make
        # min(...) large-magnitude negative and explode the loss).
        surrogate = _dual_clip_surrogate(surrogate, adv_tok, cfg.clip_ratio_c)

        # Schulman k3 unbiased KL estimator: exp(d) - d - 1, d = ref_logp - logp.
        d = ref_logp - logp
        kl_tok = torch.exp(d) - d - 1.0                         # >= 0

        pg_loss = -(surrogate * m).sum() / tok_count
        kl_loss = cfg.beta * (kl_tok * m).sum() / tok_count
        loss = pg_loss + kl_loss

        # Entropy bonus (subtracted from loss -> pushes entropy up). The coef is
        # adapted by the PI controller toward target_entropy when present.
        mean_entropy = 0.0
        ent_coef = 0.0
        if use_entropy:
            mean_entropy = float(((ent_tok * m).sum() / tok_count).detach())
            if entropy_controller is not None:
                ent_coef = entropy_controller.update(mean_entropy)
            else:
                ent_coef = cfg.entropy_coef
            if ent_coef != 0.0:
                ent_bonus = (ent_tok * m).sum() / tok_count
                loss = loss - ent_coef * ent_bonus

        with torch.no_grad():
            clip_frac = (((ratio < 1.0 - cfg.clip_eps_low) | (ratio > 1.0 + cfg.clip_eps_high)).float() * m).sum() / tok_count
        last_metrics = {
            "loss": float(loss.detach()),
            "pg_loss": float(pg_loss.detach()),
            "kl": float((kl_tok * m).sum().detach() / tok_count),
            "clip_frac": float(clip_frac.detach()),
            "ratio_mean": float(((ratio * m).sum() / tok_count).detach()),
            "reward_mean": float(rewards.mean().detach()),
            "reward_std": float(rewards.std(unbiased=False).detach()),
            "adv_abs_mean": float(adv_seq.abs().mean().detach()),
            "entropy": mean_entropy,
            "entropy_coef": ent_coef,
        }
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optimizer.step()
    return last_metrics


def _load_policy(cfg: GRPOConfig) -> tuple[Transformer, BytesTokenizer]:
    tok = BytesTokenizer()
    if cfg.policy_ckpt and Path(cfg.policy_ckpt).exists():
        state = torch.load(cfg.policy_ckpt, map_location="cpu", weights_only=False)
        mcfg = state.get("model_cfg") or cfg.model_cfg
        if mcfg is None:
            raise ValueError("policy_ckpt has no model_cfg; pass cfg.model_cfg")
        m = Transformer(mcfg)
        if "model" in state:
            m.load_state_dict(state["model"])
        return m, tok
    if cfg.model_cfg is None:
        raise ValueError("either policy_ckpt must exist or cfg.model_cfg must be set")
    return Transformer(cfg.model_cfg), tok


def run_grpo(
    cfg: GRPOConfig,
    prompts: list[str],
    verifier: Verifier,
) -> str:
    """Run the GRPO loop against a single ``verifier`` over ``prompts``.

    For brevity this uses one verifier for all prompts; a real run carries a
    per-prompt verifier spec (the answer key / hidden tests live with the prompt).

    ``verifier`` may be a bare correctness verifier or a
    :class:`platform.rl.reward.CompositeReward` (format/length/anti-hacking
    shaping); if it exposes ``.breakdown(prompt, response)`` the per-component
    means are logged into each step's metrics for monitoring.
    """
    policy, tok = _load_policy(cfg)
    if cfg.ref_ckpt and Path(cfg.ref_ckpt).exists():
        ref_state = torch.load(cfg.ref_ckpt, map_location="cpu", weights_only=False)
        ref = Transformer(ref_state["model_cfg"])
        ref.load_state_dict(ref_state["model"])
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
    else:
        ref = clone_for_reference(policy)

    prompt_ids = [tok.encode(p) for p in prompts]
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    ent_ctl = make_entropy_controller(cfg)
    history: list[dict] = []
    has_breakdown = hasattr(verifier, "breakdown")

    for step in range(cfg.steps):
        roll = sample_group(
            policy, prompt_ids,
            group_size=cfg.group_size,
            max_new_tokens=cfg.max_new_tokens,
            seq_len=cfg.seq_len,
            tokenizer=tok,
            temperature=cfg.temperature,
            seed=step,
        )
        pairs = [(prompts[int(gi)], txt) for gi, txt in zip(roll.group_index, roll.response_text)]
        rewards = torch.tensor([verifier(p, r) for p, r in pairs], dtype=torch.float32)
        metrics = grpo_step(policy, ref, roll, rewards, cfg, optimizer=opt,
                            entropy_controller=ent_ctl)
        if has_breakdown:
            # Average each shaped-reward component across the rollout group.
            comps: dict[str, float] = {}
            bd = [verifier.breakdown(p, r) for p, r in pairs]
            for key in bd[0]:
                comps[f"reward_{key}"] = sum(b[key] for b in bd) / len(bd)
            metrics.update(comps)
        history.append(metrics)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grpo.pt"
    torch.save(
        {"model": policy.state_dict(), "model_cfg": policy.cfg, "history": history},
        out_path,
    )
    return str(out_path)


def run_grpo_async(
    cfg: GRPOConfig,
    prompts: list[str],
    verifier: Verifier,
) -> str:
    """GRPO using the **async actor–learner** rollout engine (real serving Engine
    with KV-cache decode) instead of the synchronous sampler.

    Identical learner math to :func:`run_grpo`; only generation is swapped for
    :class:`platform.rl.async_rollout.AsyncRolloutEngine`, which generates the
    group concurrently and (in production) decouples actors from the learner with
    weight sync. The learner calls ``actor.sync_weights()`` after each update.
    """
    from .async_rollout import AsyncRolloutConfig, AsyncRolloutEngine

    policy, tok = _load_policy(cfg)
    ref = clone_for_reference(policy)
    actor = AsyncRolloutEngine(
        policy, tok,
        AsyncRolloutConfig(
            group_size=cfg.group_size, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, seq_len=cfg.seq_len,
        ),
    )
    prompt_ids = [tok.encode(p) for p in prompts]
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    ent_ctl = make_entropy_controller(cfg)
    history: list[dict] = []
    has_breakdown = hasattr(verifier, "breakdown")

    for _ in range(cfg.steps):
        roll = actor.generate_group(prompt_ids)
        pairs = [(prompts[int(gi)], txt) for gi, txt in zip(roll.group_index, roll.response_text)]
        rewards = torch.tensor([verifier(p, r) for p, r in pairs], dtype=torch.float32)
        metrics = grpo_step(policy, ref, roll, rewards, cfg, optimizer=opt,
                            entropy_controller=ent_ctl)
        metrics["weight_version"] = actor.weight_version
        if has_breakdown:
            bd = [verifier.breakdown(p, r) for p, r in pairs]
            for key in bd[0]:
                metrics[f"reward_{key}"] = sum(b[key] for b in bd) / len(bd)
        history.append(metrics)
        actor.sync_weights()   # learner -> actor weight sync after the update

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grpo_async.pt"
    torch.save(
        {"model": policy.state_dict(), "model_cfg": policy.cfg, "history": history},
        out_path,
    )
    return str(out_path)
