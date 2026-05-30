# 04 — Pretraining

## Optimizer

- **AdamW**, β=(0.9, 0.95), ε=1e-8, weight decay 0.1 (excluding norms/biases/embeddings).
- LR schedule: linear warmup (2000 steps) → cosine decay to 10% of peak.
- Peak LR scales with width per μP, or use heuristic 3e-4 (1B) → 1.5e-4 (70B) → 8e-5 (400B).
- Gradient clipping: global norm 1.0.
- Batch size: ramp from 1M tokens → 4–16M tokens. Sequence length 4k or 8k initially.

## Numerics

- BF16 weights, BF16 activations, FP32 master weights & optimizer state, FP8 matmul on Hopper/Blackwell via Transformer Engine.
- Loss in FP32. Reduction in FP32. No FP16 anywhere (overflow risk too high for long runs).

## Stability tooling

- **QK-norm** (`qk_norm=True`): per-head RMSNorm on queries and keys before
  attention — caps attention-logit growth, a cheap stabilizer for large-scale
  runs (monitor "attention-logit max" below).
- Per-step monitor: loss, grad-norm, param-norm, update/param ratio, attention-logit max.
- Auto-rewind on loss spike (>4σ over 200-step rolling): roll back N steps, lower LR, skip 1k steps of data.
- Spike forensics: dump bad batch + activations to S3 for offline replay.

## Checkpoint cadence

Every 1000 steps (≈ every 30 min at 16M batch). Keep last 10 + every 10000th forever. See `06-checkpointing.md`.

## Curriculum

1. Phase A (90% of tokens): stable web-heavy mix at 4k context.
2. Phase B (8% of tokens): long-context extension to 32k–128k via RoPE base scaling.
3. Phase C (2% of tokens): annealing — high-quality data only (textbooks, math, curated code), LR decayed to 0. Gives big eval bumps.
