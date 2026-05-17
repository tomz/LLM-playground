# Frontier Model Training Platform

A design-doc + skeleton-code blueprint for an end-to-end system capable of training, aligning, evaluating, and serving GPT-class frontier models (1B–500B+ parameters).

This repository is **architecture-only**. It defines interfaces, module boundaries, infrastructure assumptions, and reference implementations stubbed at the function-signature level. It does *not* perform real training — that requires a multi-thousand-GPU cluster, $10M–$500M of compute, and a dedicated team.

## Why this exists

Training a frontier LLM is not one problem; it is roughly a dozen production systems glued together. Most public "train your own GPT" repos cover only step 3 (pretraining) at toy scale. This blueprint covers all of:

```
  raw web  →  data pipeline  →  pretraining  →  midtraining  →  SFT  →  preference data  →  RLHF/DPO  →  eval  →  red-team  →  serving  →  telemetry → (back to data)
```

## Layout

```
frontier-platform/
├── docs/                 # Design docs — read these first
│   ├── 00-overview.md
│   ├── 01-data-pipeline.md
│   ├── 02-tokenizer.md
│   ├── 03-model-architecture.md
│   ├── 04-pretraining.md
│   ├── 05-distributed-training.md
│   ├── 06-checkpointing.md
│   ├── 07-alignment-sft-rlhf.md
│   ├── 08-evaluation.md
│   ├── 09-safety-redteam.md
│   ├── 10-serving-inference.md
│   ├── 11-infrastructure.md
│   ├── 12-cost-and-scaling-laws.md
│   └── 13-simulation.md     # ← discrete-event simulator (NEW)
├── platform/             # Skeleton Python packages
│   ├── data/             # ingestion, dedup, filter, mix, shard
│   ├── tokenizer/        # BPE training + serving
│   ├── model/            # transformer, attention variants, MoE
│   ├── training/         # pretraining loop, optimizer, parallelism
│   ├── alignment/        # SFT, reward model, PPO, DPO
│   ├── eval/             # benchmark harness
│   ├── safety/           # classifiers + red-team harness
│   ├── serving/          # inference server, KV-cache, batching
│   └── infra/            # cluster, scheduler, storage, observability
├── configs/              # YAML configs per model size (1B / 7B / 70B / 400B)
├── scripts/              # entry-point CLIs
└── tests/                # smoke tests for the skeletons
```

## Reading order

1. `docs/00-overview.md` — system diagram and team/cost realities
2. `docs/12-cost-and-scaling-laws.md` — what you are signing up for
3. `docs/01-data-pipeline.md` through `docs/11-infrastructure.md` in order
4. `platform/*/README.md` for each subsystem

## Status

| Subsystem        | Design doc | Skeleton code | Tests |
|------------------|:---------:|:-------------:|:-----:|
| Data pipeline    | ✅        | ✅            | smoke |
| Tokenizer        | ✅        | ✅            | smoke |
| Model            | ✅        | ✅            | smoke |
| Pretraining      | ✅        | ✅            | smoke |
| Alignment        | ✅        | ✅            | smoke |
| Evaluation       | ✅        | ✅            | smoke |
| Safety           | ✅        | ✅            | —     |
| Serving          | ✅        | ✅            | smoke |
| Infra            | ✅        | ✅            | —     |

## What works today

- ✅ Tier 1 — **data pipeline**: acquire local files, extract, filter, dedup, decontaminate, shard, mix, stream-load (resumable).
- ✅ Tier 1 — **tokenizer**: bytes-level (stdlib) + optional HF BPE.
- ✅ Tier 1 — **safety**: classifiers, gates, red-team toy suite. **Infra**: cluster, local scheduler, observability.
- ✅ Tier 1 — **eval**: in-process perplexity + arena ELO.
- ✅ Tier 2 — real **Transformer** (RoPE / RMSNorm / SwiGLU / GQA / MoE) with muP-style init.
- ✅ Tier 2 — **training**: AdamW (WD-by-dim), cosine+warmup LR, single-process ParallelEngine, DCP-style checkpointing, SpikeMonitor + RewindController, `Trainer.fit`.
- ✅ Tier 2 — **serving**: in-process `TorchEngine` with streaming generate, router.
- ✅ Tier 3 — **alignment**: SFT (assistant-token loss mask), BT reward model, DPO (sigmoid / IPO / KTO variants), PPO with GAE + KL-to-reference penalty + clipped objective + value head.
- ✅ End-to-end **smoke pipeline** (`bash scripts/smoke_pipeline.sh`, < 10s CPU): corpus → shards → pretrain → SFT → RM → DPO → PPO → eval → generate.

Still `NotImplementedError` (intentionally — out of scope for a single-machine blueprint):

- CommonCrawl / GitHub / Arxiv / Wikipedia source connectors (need real network + scrubbing).
- FSDP2 / TP / PP backends (use `distgpt` for those).
- vLLM / TRT-LLM / SGLang serving backends (lazy-import stubs).

## Running tests

```bash
cd frontier-platform
.venv/bin/python -m pytest -q             # 69 tests, ~4 s on CPU
bash scripts/smoke_pipeline.sh            # full pipeline, ~6 s on CPU
```
