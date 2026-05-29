# 15 — Reasoning & RL on Verifiable Rewards (RLVR)

> **Status: design stub.** This is the single largest capability gap identified
> in `14-gap-analysis-vs-frontier.md` (gap #1). 2025 was *"the year of reasoning,
> RLVR, and GRPO"*; this blueprint's `07-alignment-sft-rlhf.md` stops at the
> 2024 SFT→RM→PPO/DPO paradigm. This doc specifies the missing post-training
> regime. No code yet — `platform/rl/` is unimplemented.

---

## Why this exists

The modern frontier post-trains a base model into a **reasoning model** by
running large-scale RL where the reward is a *deterministic verifier*, not a
learned reward model:

- **Math:** exact-answer / symbolic equivalence check.
- **Code:** unit tests + compiler/runtime in a sandbox.
- **Formal:** proof checkers (Lean/Isabelle), constraint solvers.
- **Structured:** schema/JSON validators, regex, deterministic graders.

Reasoning behavior (long chain-of-thought, self-verification, backtracking)
*emerges* from this RL rather than being imitated from human traces. Post-training
compute now rivals or exceeds pretraining compute for these models — so RLVR is a
first-class **compute sink**, not a fine-tuning afterthought.

## Pipeline

```
base model
   │
   ├── (optional) reasoning SFT cold-start — a few k long-CoT traces to
   │     stabilize format before RL (R1 used a small cold-start set)
   │
   ├── RLVR main loop  ──────────────────────────────────────────────┐
   │     • sample G rollouts per prompt (group)                        │
   │     • score each with a verifier → scalar reward                  │
   │     • GRPO: advantage = (r - mean(group)) / std(group)            │
   │     • policy-gradient update, KL-to-reference penalty             │
   │     • NO value network (that's the GRPO win over PPO)             │
   │                                                                   │
   └── (optional) general preference alignment (DPO/RLHF) for tone/safety
```

## GRPO in one box

```
for prompt in batch:
    rollouts = actor.generate(prompt, n=G)            # generation-heavy
    rewards  = [verify(prompt, r) for r in rollouts]  # sandboxed workers
    adv      = (rewards - rewards.mean()) / (rewards.std() + eps)
    loss     = -(adv * logprob(rollouts)).mean() + beta * KL(pi || pi_ref)
```

Key properties vs. PPO (`07`): no critic/value head → ~2× less memory; reward is
external and deterministic → no reward-model drift; group-relative baseline →
stable advantages without GAE.

## The missing subsystem: `platform/rl/`

This is an **async actor–learner system**, not a loss function. Components:

| Module (proposed) | Responsibility |
|-------------------|----------------|
| `rl/rollout.py` | async inference engine (vLLM/SGLang) producing G samples/prompt at high throughput; weight-sync from learner |
| `rl/verifiers/` | pluggable reward workers: `math.py`, `code.py` (sandboxed exec — gVisor/Firecracker), `formal.py`, `schema.py` |
| `rl/grpo.py` | GRPO/PPO-with-verifier learner; KL-to-ref; group advantage |
| `rl/buffer.py` | prompt/rollout/reward queue between actors and learner |
| `rl/reward.py` | reward shaping, length penalties, format rewards, reward hacking guards |

Reuse: the serving stack (`10-serving-inference.md`) *is* the rollout engine;
the sandbox from `09-safety-redteam.md` *is* the code verifier. RLVR mostly
wires existing subsystems into an async loop.

## Data (feeds `01-data-pipeline.md`)

- **Verifiable prompt sets:** math with known answers, coding tasks with hidden
  tests, formal/symbolic problems. Quality + difficulty curriculum matters more
  than volume.
- **Reasoning-trace SFT cold-start:** a small, high-quality long-CoT set.
- **Decontamination is critical:** RLVR on contaminated math/code is benchmark
  fraud. Extend the n-gram bloom filters to the RL prompt pool.

## Stability & failure modes

- **Reward hacking:** models exploit verifier gaps (printing expected output,
  degenerate tests). Mitigate with held-out tests, adversarial graders.
- **Length explosion:** unbounded CoT growth; add length penalties / budgets.
- **KL collapse / mode collapse:** monitor KL-to-reference, entropy, pass@k.
- **Verifier throughput:** the sandbox fleet, not the GPUs, is often the
  bottleneck. Budget reward-worker CPU like you budget dataloader CPU.

## Inference-time reasoning (serving + eval implications)

- Serving (`10`) needs a **thinking budget** knob: trade test-time tokens for
  accuracy; hide/show reasoning traces; bound max reasoning tokens per request.
- Eval (`08`) needs **score-vs-test-time-compute curves**, not single points,
  and pass@k / majority-vote / best-of-n reporting.

## Simulator hook (`13-simulation.md`)

Add `platform/sim/reasoning_rl_sim.py`: model rollout compute (G × samples ×
seq_len), verifier CPU cost, and a capability bump that depends on *post-training*
compute — today the sim's eval scores are pretraining-only and wouldn't move if
you added RL.

## References

- DeepSeek-AI, *DeepSeek-R1* — arXiv 2501.12948.
- Shao et al., *DeepSeekMath* (GRPO) — arXiv 2402.03300.
- Survey of frontiers in LLM reasoning — arXiv 2504.09037.
- See also `docs/2026-05-sota-llm-agi.md` §8 (reasoning post-training).
