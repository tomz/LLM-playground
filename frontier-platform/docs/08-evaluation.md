# 08 — Evaluation

## Tiers

1. **Train-time eval** (every 1000 steps, fast): held-out perplexity on 8 domain slices, plus 5–10 zero-shot tasks (HellaSwag, ARC-easy, PIQA, BoolQ, OpenBookQA). Runs in <5 min on 8 GPUs.
2. **Checkpoint eval** (every 10000 steps): full lm-eval-harness suite — MMLU, GPQA, BBH, MATH, GSM8K, HumanEval, MBPP, IFEval, TruthfulQA, MT-Bench. ~2h on 64 GPUs.
3. **Release eval** (manual gate): everything above + arena ELO via blind pairwise vs reference models, internal red-team report, capability uplift studies, contamination report.

## Infrastructure

- Eval cluster is **separate** from training cluster (avoid noisy-neighbor; eval is bursty).
- Every eval run pins: model SHA, eval-harness git SHA, prompt template SHA, decoding params.
- All results land in a Postgres + dashboard; PRs auto-comment with regression deltas.

## Contamination

For every eval set, store n-gram (n=13) bloom filter. Pretraining filter must subtract these before sharding. Audit: top-50 nearest training docs to each eval question, flag matches >0.8 Jaccard.
