# 00 — System Overview

## What "frontier" means here

A frontier model in 2024-2025 has roughly these properties:

- **Parameters**: 70B–2T (dense) or 200B–10T (MoE)
- **Training tokens**: 5T–20T
- **Pretraining compute**: 1e24 – 1e26 FLOPs
- **Cluster**: 8k–100k H100/H200/B200 GPUs for 60–180 days
- **Capex**: $50M – $1B for a single run; $200M–$5B/yr for a program
- **Headcount**: 80–400 people across research, infra, data, safety, product

This platform's design targets the lower end (≈10B–100B dense) but the abstractions extend upward.

## End-to-end flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. DATA                                                                  │
│  Common Crawl, GitHub, books, papers, code, multilingual, synthetic      │
│         │                                                                │
│         ▼  (extract → language ID → quality filter → dedup → PII scrub)  │
│   sharded token files (.bin / WebDataset / Mosaic Streaming)             │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. TOKENIZER (BPE / Unigram / tiktoken-style)                            │
│  Train once on representative sample → vocab.json + merges.txt           │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. PRETRAINING                                                           │
│  Decoder-only transformer (RoPE, SwiGLU, RMSNorm, GQA, optional MoE)     │
│  3D parallelism: data + tensor + pipeline (+ expert + sequence)          │
│  Optimizer: AdamW or Lion + ZeRO-3 or FSDP                               │
│  Mixed precision: BF16 weights + FP32 master + FP8 matmul (Hopper+)      │
│  Curriculum: stable mix → long-context extension → annealing             │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. MIDTRAINING / CONTEXT EXTENSION                                       │
│  RoPE base rescaling, YaRN, long-doc data, position interpolation        │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 5. ALIGNMENT                                                             │
│  SFT (instruction tuning) → Reward Model → RLHF (PPO) or DPO/IPO/KTO     │
│  Constitutional AI / RLAIF for scale                                     │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 6. EVAL  (continuous, gates every checkpoint)                            │
│  MMLU, GPQA, HumanEval, MATH, BBH, IFEval, ARC, perplexity, ELO arena    │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 7. SAFETY / RED-TEAM                                                     │
│  CBRN, cyber, persuasion, autonomy evals; jailbreak suites; classifiers  │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 8. SERVING                                                               │
│  Quantization (FP8/INT8/AWQ), continuous batching, paged KV cache,       │
│  speculative decoding, tensor-parallel inference, autoscaling            │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
  Telemetry → user feedback → new preference data → loop back to (5)
```

## Org chart implied by this system

| Pod              | Headcount | Owns |
|------------------|-----------|------|
| Pretraining      | 10–25     | model code, training loop, scaling, debugging loss spikes |
| Data             | 15–40     | crawl, filter, dedup, mix design, synthetic data |
| Infra / Systems  | 20–50     | cluster, scheduler, storage, networking, observability |
| Tokenizer / Eval | 5–10      | tokenizer, eval harness, leaderboards |
| Alignment        | 10–25     | SFT data, RM, RLHF, DPO, persona |
| Safety           | 10–30     | policy, red-team, classifiers, dangerous-capability evals |
| Inference        | 10–25     | serving stack, quantization, latency |
| Product / API    | 10–40     | API, fine-tuning service, billing, docs |

## Non-goals of this repo

- We do **not** ship trained weights.
- We do **not** ship a working CUDA kernel library — assume FlashAttention / Transformer Engine / Megatron-Core.
- We do **not** solve alignment. We provide hooks where alignment teams plug in.
