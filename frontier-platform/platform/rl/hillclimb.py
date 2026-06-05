"""Hill-climb orchestrator (harvest of MAI-Thinking-1 §5, the conceptual core).

The paper's signature recipe is not one monolithic training run but an iterative
*climb* (docs/research/mai-thinking-1-deep-dive.md §5):

    1. Start from a base model.
    2. Train several **specialists** via RL (the paper: SWE/Agentic, STEM,
       Helpfulness & Safety) — each is the base policy RL'd against its own
       prompt pool + verifier.
    3. **Distill** the specialists back into one consolidated model via SFT on
       their (rejection-sampled) outputs.
    4. A final RL **climb** on the consolidated model.

This module orchestrates exactly that loop over the platform's existing pieces:

* specialists ← :func:`platform.rl.grpo.run_grpo` (one GRPO run per specialist),
* distillation harvest ← :func:`platform.rl.rollout.sample_group` + **rejection
  sampling** against each specialist's verifier (keep only winning rollouts, the
  same filter :mod:`platform.data.synthetic` uses),
* consolidation ← :func:`platform.rl.coldstart.run_coldstart` (SFT on the
  harvested traces),
* final climb ← one more :func:`~platform.rl.grpo.run_grpo` on the union pool.

Everything runs end-to-end on CPU with the byte tokenizer and the tiny test
model — it is a faithful *shape* of the recipe, not a frontier-scale run. The
swap-the-backend boundaries are the same as the rest of `platform.rl`: a real
generation actor and a real verifier library.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..tokenizer.bytes import BytesTokenizer
from .coldstart import ColdStartConfig, run_coldstart
from .grpo import GRPOConfig, run_grpo
from .rollout import sample_group
from .verifiers import Verifier


@dataclass
class Specialist:
    """One RL specialist: a name, its prompt pool, and its verifier.

    ``grpo_overrides`` optionally tweaks the GRPO config for this domain (e.g. a
    higher ``clip_eps_high`` for an exploration-heavy STEM specialist). The
    orchestrator fills in ``policy_ckpt``/``out_dir`` from the base + run dir.
    """

    name: str
    prompts: list[str]
    verifier: Verifier
    grpo_overrides: dict = field(default_factory=dict)


@dataclass
class HillClimbConfig:
    base_ckpt: str
    out_dir: str = "out/hillclimb"
    # Per-specialist GRPO defaults (overridable per specialist).
    specialist_steps: int = 5
    group_size: int = 4
    lr: float = 5e-3
    beta: float = 0.0
    max_new_tokens: int = 8
    seq_len: int = 32
    temperature: float = 1.0
    # Distillation (rejection-sampling) harvest.
    distill_samples_per_prompt: int = 4
    distill_reward_threshold: float = 1.0   # keep rollouts with reward >= this
    min_distill_examples: int = 1           # fallback: keep best-by-reward if few pass
    distill_epochs: int = 3
    distill_lr: float = 5e-3
    # Final climb.
    final_steps: int = 5
    final_lr: float = 5e-3


@dataclass
class StageResult:
    name: str
    ckpt: str
    metrics: dict = field(default_factory=dict)


@dataclass
class HillClimbResult:
    base_ckpt: str
    specialists: list[StageResult]
    distill: StageResult
    final: StageResult
    out_dir: str

    @property
    def best_ckpt(self) -> str:
        """The end-of-climb checkpoint (what you'd ship)."""
        return self.final.ckpt

    def summary(self) -> dict:
        return {
            "base": self.base_ckpt,
            "specialists": {s.name: s.metrics for s in self.specialists},
            "distill": self.distill.metrics,
            "final": self.final.metrics,
            "final_ckpt": self.final.ckpt,
        }


def _grpo_cfg(cfg: HillClimbConfig, policy_ckpt: str, out_dir: str,
              steps: int, lr: float, overrides: dict | None = None) -> GRPOConfig:
    base = dict(
        policy_ckpt=policy_ckpt, out_dir=out_dir, steps=steps, lr=lr,
        beta=cfg.beta, group_size=cfg.group_size, max_new_tokens=cfg.max_new_tokens,
        seq_len=cfg.seq_len, temperature=cfg.temperature,
    )
    base.update(overrides or {})
    return GRPOConfig(**base)


def _mean_reward_history(ckpt_path: str) -> dict:
    """Pull the GRPO ``history`` off a saved checkpoint into early/late means."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hist = state.get("history") or []
    if not hist:
        return {"steps": 0}
    rewards = [h.get("reward_mean", 0.0) for h in hist]
    k = max(1, len(rewards) // 3)
    return {
        "steps": len(rewards),
        "reward_early": sum(rewards[:k]) / k,
        "reward_late": sum(rewards[-k:]) / k,
        "reward_final": rewards[-1],
    }


def train_specialist(cfg: HillClimbConfig, spec: Specialist) -> StageResult:
    """RL one specialist from the base policy against its own verifier."""
    out_dir = str(Path(cfg.out_dir) / "specialists" / spec.name)
    gcfg = _grpo_cfg(cfg, cfg.base_ckpt, out_dir, cfg.specialist_steps, cfg.lr,
                     spec.grpo_overrides)
    ckpt = run_grpo(gcfg, spec.prompts, spec.verifier)
    return StageResult(spec.name, ckpt, _mean_reward_history(ckpt))


def harvest_distillation_data(
    cfg: HillClimbConfig, specialists: list[Specialist],
    specialist_ckpts: list[str], tokenizer=None,
) -> list[dict]:
    """Rejection-sample each trained specialist on its prompts; keep winners.

    For every specialist we sample ``distill_samples_per_prompt`` completions and
    keep the (prompt, response) pairs whose verifier reward clears
    ``distill_reward_threshold``. If fewer than ``min_distill_examples`` clear it
    (common with a weak/tiny policy), we fall back to the highest-reward samples
    so the consolidation SFT always has data — the paper's pipeline assumes the
    specialists succeed often enough that this fallback rarely triggers.
    """
    from ..model.transformer import Transformer

    tok = tokenizer or BytesTokenizer()
    examples: list[dict] = []
    for spec, ckpt_path in zip(specialists, specialist_ckpts):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        policy = Transformer(state["model_cfg"])
        policy.load_state_dict(state["model"])
        policy.eval()

        prompt_ids = [tok.encode(p) for p in spec.prompts]
        roll = sample_group(
            policy, prompt_ids,
            group_size=cfg.distill_samples_per_prompt,
            max_new_tokens=cfg.max_new_tokens, seq_len=cfg.seq_len,
            tokenizer=tok, temperature=cfg.temperature, seed=0,
        )
        scored: list[tuple[float, dict]] = []
        for gi, text in zip(roll.group_index, roll.response_text):
            prompt = spec.prompts[int(gi)]
            reward = spec.verifier(prompt, text)
            scored.append((reward, {"prompt": prompt, "response": text,
                                     "specialist": spec.name, "reward": reward}))
        winners = [ex for r, ex in scored if r >= cfg.distill_reward_threshold]
        if len(winners) < cfg.min_distill_examples:
            # Fallback: top-reward samples so SFT isn't empty.
            scored.sort(key=lambda t: t[0], reverse=True)
            winners = [ex for _, ex in scored[: cfg.min_distill_examples]]
        examples.extend(winners)
    return examples


def run_hill_climb(cfg: HillClimbConfig, specialists: list[Specialist]) -> HillClimbResult:
    """Run the full specialists → distill → climb loop.

    Returns a :class:`HillClimbResult` with the per-stage checkpoints + metrics.
    ``result.best_ckpt`` is the final consolidated+climbed model.
    """
    if not specialists:
        raise ValueError("hill-climb needs at least one specialist")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = BytesTokenizer()

    # 1. Specialists.
    spec_results = [train_specialist(cfg, s) for s in specialists]
    spec_ckpts = [r.ckpt for r in spec_results]

    # 2. Distill: rejection-sample winners, write a lineage JSONL, SFT-consolidate.
    distill_examples = harvest_distillation_data(cfg, specialists, spec_ckpts, tok)
    distill_dir = out_dir / "distill"
    distill_dir.mkdir(parents=True, exist_ok=True)
    lineage = distill_dir / "distill_data.jsonl"
    with lineage.open("w", encoding="utf-8") as f:
        for ex in distill_examples:
            f.write(json.dumps(ex) + "\n")

    cs = run_coldstart(
        ColdStartConfig(
            policy_ckpt=cfg.base_ckpt, out_dir=str(distill_dir),
            epochs=cfg.distill_epochs, lr=cfg.distill_lr,
            batch_size=max(1, min(4, len(distill_examples))), seq_len=cfg.seq_len,
        ),
        # cold-start consumes {prompt, response}; strip the bookkeeping keys.
        [{"prompt": ex["prompt"], "response": ex["response"]} for ex in distill_examples],
    )
    distill_metrics = {
        "n_examples": len(distill_examples),
        "per_specialist": {s.name: sum(1 for e in distill_examples
                                        if e["specialist"] == s.name)
                           for s in specialists},
        "loss_first": cs.loss_history[0] if cs.loss_history else None,
        "loss_last": cs.loss_history[-1] if cs.loss_history else None,
    }
    distill_result = StageResult("distill", cs.out_path, distill_metrics)

    # 3. Final climb on the union of all specialist prompts.
    union_prompts: list[str] = []
    for s in specialists:
        union_prompts.extend(s.prompts)
    # Use the first specialist's verifier set is wrong; instead climb against a
    # composite that rewards each prompt under its own specialist verifier.
    prompt_to_verifier = {}
    for s in specialists:
        for p in s.prompts:
            prompt_to_verifier.setdefault(p, s.verifier)

    def union_verifier(prompt: str, response: str) -> float:
        v = prompt_to_verifier.get(prompt)
        return v(prompt, response) if v is not None else 0.0

    final_cfg = _grpo_cfg(cfg, cs.out_path, str(out_dir / "final"),
                          cfg.final_steps, cfg.final_lr)
    final_ckpt = run_grpo(final_cfg, union_prompts, union_verifier)
    final_result = StageResult("final", final_ckpt, _mean_reward_history(final_ckpt))

    result = HillClimbResult(
        base_ckpt=cfg.base_ckpt,
        specialists=spec_results,
        distill=distill_result,
        final=final_result,
        out_dir=str(out_dir),
    )
    # Persist a run summary for inspection.
    (out_dir / "summary.json").write_text(json.dumps(result.summary(), indent=2),
                                          encoding="utf-8")
    return result


__all__ = [
    "Specialist",
    "HillClimbConfig",
    "HillClimbResult",
    "StageResult",
    "train_specialist",
    "harvest_distillation_data",
    "run_hill_climb",
]
