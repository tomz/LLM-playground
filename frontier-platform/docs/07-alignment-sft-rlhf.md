# 07 — Alignment: SFT, Reward Model, RLHF, DPO

## Pipeline

```
base model
   │
   ├── SFT on 50k–500k high-quality (prompt, response) pairs
   │     • mix: instruction following, code, math, reasoning, refusals, tool use
   │     • epochs: 2–4, low LR (5e-6 to 2e-5), cosine decay
   │
   ├── Reward Model (RM)
   │     • init from SFT model, replace LM head with scalar head
   │     • train on (prompt, chosen, rejected) preference pairs
   │     • Bradley-Terry loss; calibrate with KL probe
   │
   └── Policy optimization
         ├── PPO  — clip ε=0.2, KL penalty β scheduled by KL controller,
         │          value head from RM init, GAE λ=0.95, γ=1.0
         ├── DPO  — no RM needed, β=0.1, ref model = SFT model
         ├── IPO / KTO / ORPO — variants for stability or unpaired data
         └── RLAIF / Constitutional — AI-generated preferences for scale
```

## Data sources

- Human contractors (vendor-managed, ~$3–$30 per pair depending on domain).
- Model-graded with rubric (cheaper, biased toward grader's preferences).
- Existing public sets (UltraFeedback, HH-RLHF, Nectar) for cold start only.
- Domain-targeted: code (compiler/test feedback), math (verifier), reasoning (process supervision PRM).

## Stability

RLHF is *the* stage where models silently degrade. Mandatory guards:

- Held-out KL to SFT model — alarm if >25 nats.
- Capability evals (MMLU, HumanEval) every 100 PPO steps — alarm on >2pt regression.
- Length-bias monitor — RMs love long answers; penalize.
- Refusal-rate monitor on benign prompts.
