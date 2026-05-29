# SOTA Watch — LLM & AGI · 2026-05

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-05  ·  **Status:** published

> Inaugural edition. Theme: **train better models faster and cheaper, at every
> scale.** This consolidates two prior internal research passes — a
> modded-nanogpt/Muon survey aimed at the from-scratch trainers, and an
> FSDP-scoped review originally written for `distgpt` — into one ranked,
> deduplicated digest.
>
> **Framing:** content is *not* constrained by any one workstation's GPUs. For
> each project we assume both a **minimal** box (enough to run a meaningful job)
> and an **ideal** box (enough to unlock the full technique set), and recommend
> what is *correct at that scale*. Hardware is a sizing note, never a blocker.
> "Harvest status" tracks what we've implemented in-repo so far.
>
> Audience: anyone training, fine-tuning, or serving LLMs from 10M to 500B+
> parameters who wants the current best practice per scale tier.

---

## TL;DR — this month's harvest

- **Muon optimizer** shipped in `nanogpt-edu` (Newton-Schulz-orthogonalized
  updates for 2D hidden weights; AdamW for embeddings/head/norms). ~1.35×
  sample-efficiency; the single biggest algorithmic lever for from-scratch
  training, and a recommended port to `midgpt`/`distgpt`.
- **Multi-Token Prediction** (DeepSeek-V3 style) shipped in `nanogpt-edu` —
  denser gradient, train-only, and the head doubles as a speculative-decoding
  draft model at serving time.
- **Liger Kernel + DoRA + rsLoRA + NEFTune** shipped as config knobs in
  `coder-finetune` — fused Triton kernels plus three near-free quality
  upgrades on the TRL/PEFT stack.
- **Data scaling** is the highest-ROI knob at every scale: `nanogpt-edu` now
  has FineWeb-Edu / DCLM BPE prep. Over-training past Chinchilla (20–50×) is the
  modern norm, not a luxury.
- **Datacenter-scale techniques are first-class here**, sized to where they pay
  off: **FP8/NVFP4** (Hopper/Blackwell, >1B), **FlashAttention-3** (Hopper+),
  **Multi-head Latent Attention** (long-context + serving), **MoE** (≥3B
  active-equivalent), **3D/5D parallelism + overlapped comms** (multi-node).

---

## Hardware envelopes per project

Aspirational targets, not a description of any single machine. "Minimal" runs a
meaningful job; "ideal" unlocks the full technique set for that scale.

| Project | Scale | Minimal | Ideal | What ideal unlocks |
|---------|-------|---------|-------|--------------------|
| **nanogpt-edu** | 10M–100M | laptop CPU / 8 GB GPU | 1× H100 80 GB | bf16→FP8 head, long-ctx FlexAttention, fast full sweeps in minutes |
| **midgpt** | 124M–1.5B | 1× 16 GB GPU | 8× H100 single node | FP8 matmul, FlashAttention-3, real Chinchilla-optimal token budgets |
| **distgpt** | 1B–70B | 1 node × 8× A100 | 8–64 nodes × 8× H100/B200 + NVLink/InfiniBand | 3D/5D parallelism, FP8, comms/compute overlap, MoE expert parallelism |
| **coder-finetune** | 0.5B–7B | 1× 8 GB GPU (QLoRA) | 1× 80 GB or 2–8× for full-FT | full-FT of 7B, long-context packing, fast multi-epoch SFT + DPO/ORPO |
| **frontier-platform** | 1B–500B+ | design doc (no GPU) | 1k–16k× H100/B200 | the entire program: pretrain → SFT → RLHF → eval → serve at frontier scale |

---

## Tier 1 — high-ROI, broadly applicable

These help from a laptop to a 16k-GPU cluster.

### 1. Muon optimizer (hidden 2D weights) — **shipped (nanogpt-edu)**

Replaces the Adam update for 2D weight matrices with the nearest semi-orthogonal
matrix via a 5-step Newton-Schulz iteration (stable in bf16). Orthogonalizing
amplifies the "rare directions" a near-low-rank momentum buffer drowns out.

- **Win:** ~1.35× sample-efficiency over tuned AdamW; the modded-nanogpt 124M
  speedrun (45 min → ~3 min on 8×H100) is largely Muon. Scales to 1.5B
  (GPT-2-XL quality in 10 vs 13.3 8×H100-hours) and is being pushed toward the
  7B+ regime.
- **Scope rule:** Muon for hidden matmul weights; AdamW for embeddings, lm_head,
  MTP heads, all 1-D params.
- **Min HW:** any (per-parameter, FSDP-safe). **Ideal:** the distributed Muon
  variant amortizes the NS overhead across ranks on multi-node.
- **Source:** [1], [2]  ·  **Harvest:** shipped → `nanogpt-edu`. **Roadmap:**
  port to `midgpt` + `distgpt`.

### 2. Multi-Token Prediction (MTP) — **shipped (nanogpt-edu)**

Auxiliary heads predict tokens *n+2/n+3* alongside *n+1*; loss = CE_next +
λ·mean(CE_future), λ=0.3 (DeepSeek-V3).

- **Win:** denser gradient → better sample efficiency; ~0.5–1 pp on downstream
  benchmarks at fixed compute at scale. **The MTP head is reusable as a
  speculative-decoding draft model at inference** — a serving speedup, not just
  a training trick.
- **Cost:** ~5% VRAM, ~3% throughput. Ours simplifies DeepSeek's per-depth
  transformer modules to linear heads for legibility; the full module form is
  the ideal-scale upgrade.
- **Design notes (ours):** aux loss is train-only (eval/val stays comparable);
  `generate()` uses only the main head; MTP heads routed to AdamW.
- **Source:** [3]  ·  **Harvest:** shipped → `nanogpt-edu`. **Roadmap:**
  full-module MTP + speculative decode in `midgpt`/serving.

### 3. Liger Kernel (fused Triton kernels) — **shipped (coder-finetune)**

Fused RMSNorm/RoPE/SwiGLU + `FusedLinearCrossEntropy`, one-line patch.

- **Win:** ~20% throughput, up to ~60% memory, **exact**. Fused linear-CE never
  materializes the `[batch·seq, vocab]` logits — decisive over Qwen2.5-Coder's
  ~150K vocab; scales to any large-vocab model and composes with FSDP.
- **Min HW:** any Triton GPU. **Ideal:** pairs with FSDP on multi-GPU for the
  full 20%/60% headline (benchmarked LLaMA-3-8B on 8× A100).
- **Source:** [4]  ·  **Harvest:** shipped → `coder-finetune`. **Roadmap:**
  enable in `midgpt`/`distgpt` training loops.

### 4. DoRA + rsLoRA + NEFTune — **shipped (coder-finetune)**

- **DoRA** (weight-decomposed LoRA): better quality at low rank, ~10–20% slower.
- **rsLoRA** (α/√r scaling): lets higher ranks help; free.
- **NEFTune** (embedding noise in SFT): better instruction-following; free.
- **Source:** [5], [6], [7]  ·  **Harvest:** shipped → `coder-finetune`.

### 5. QK-Norm, zero-init projections, untied embeddings — **shipped (nanogpt-edu)**

The cheap, legible slice of the modded-nanogpt architecture set: per-head
RMSNorm on Q/K (stabilizes higher LR), zero-init residual-write matrices
(identity-init blocks, muP-like), untied embed/head (helps loss once tokens
support it). **Source:** [1]  ·  **Harvest:** shipped → `nanogpt-edu`.

### 6. Data scaling: FineWeb-Edu / DCLM + over-train past Chinchilla — **shipped (prep)**

The most-replicated single intervention in pretraining is *better tokens, more
of them*.

- **Quality:** DCLM-Baseline-1.0 beats FineWeb-Edu ~10% on MMLU at matched
  compute; either gives ~30–50% sample-efficiency over raw web text.
- **Quantity:** the modern small-model norm is **20–50× Chinchilla**
  (Llama-3-8B ≈ 1875×). Over-training small models is correct.
- **Source:** [8], [9]  ·  **Harvest:** shipped → `nanogpt-edu/prepare_fineweb.py`
  (`--dataset fineweb-edu|dclm`). **Roadmap:** Chinchilla-optimal+ token budgets
  for `midgpt`/`distgpt` on ideal hardware.

### 7. `torch.compile` + correct mixed precision — **present**

`torch.compile(model)` (10–30% wall-clock via fusion; already in `nanogpt-edu`,
`midgpt`); bf16 by default, FP16 GradScaler path for older Tensor Core gens
(present in `nanogpt-edu`). **Source:** [1]

---

## Tier 2 — scale- or hardware-gated wins

These are *recommended* at the right scale — sized by where they pay off, not
blocked by any one machine.

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| **FP8 / NVFP4 training** (torchao / TransformerEngine) | 8-/4-bit matmul w/ per-tensor scaling | ~1.5–1.6× throughput, ~2× memory; validated at 12B/10T tokens | Hopper/Blackwell Tensor Cores; pays off >1B | [10] | **ideal** | midgpt, distgpt, frontier |
| **FlashAttention-3** | Hopper-optimized attention kernel | ~1.5× over SDPA on H100; warp-specialized + FP8 | Hopper+; long sequences | [11] | **ideal** | midgpt, distgpt |
| **Multi-head Latent Attention (MLA)** | Low-rank KV compression | 5–10× smaller KV cache at near-equal quality | long-context training + any serving | [3] | **planned** | distgpt, serving |
| **Mixture-of-Experts** (fine-grained + shared expert) | Sparse FFN, more params at ~constant FLOPs | large quality/$ win | ≥3B active-equiv; needs expert parallelism | [3] | **ideal** | distgpt, frontier |
| **3D/5D parallelism + comms overlap** | FSDP2 + TP + PP + EP + SP, overlapped collectives | near-linear scaling to 1000s of GPUs | multi-node + fast interconnect | [12] | **partial** | distgpt, frontier |
| **Unsloth fast-path** | Custom autograd kernels for LoRA/QLoRA | ~2× faster, ~70% less memory | single-GPU PEFT | [13] | **shipped** | coder-finetune |
| **Spectrum / targeted full-FT** | Full-FT high-SNR layers, freeze rest | beats LoRA at similar memory | mid-size SFT | — | **planned** | coder-finetune |

---

## Tier 3 — research bets (track, don't build yet)

- **Hybrid Mamba/SSM-Transformer** (e.g. NVIDIA's 12B NVFP4 work): strong
  long-context throughput; recipe still thin outside originating papers.
- **Diffusion / next-byte / tokenizer-free LMs:** promising, pre-product.
- **Long-context position schemes** (YaRN / NTK-by-parts; NoPE in alternating
  layers): adopt when training at block_size ≫ 1024.
- **muP / hyperparameter transfer** at frontier scale: zero-shot HP transfer
  from small proxies — high value for `frontier-platform`'s scaling program.

---

## Roadmap by project

Everything is on a roadmap, sized by hardware tier — nothing is "blocked."

| Project | Next harvest | Tier | Notes |
|---------|--------------|------|-------|
| nanogpt-edu | full-module MTP; FlexAttention long-context | minimal→ideal | keep the core legible; advanced bits opt-in |
| midgpt | Muon; Liger; FP8 matmul; FA-3 | minimal→ideal | FP8/FA-3 light up on 8× H100 |
| distgpt | Muon (distributed); FP8; MoE + expert parallelism; MLA | ideal | the 3D-parallel showcase; validate at multi-node |
| coder-finetune | Spectrum; DPO/ORPO via Liger fused losses; full-FT 7B | minimal→ideal | full-FT 7B wants 1× 80 GB or 2–8× |
| frontier-platform | FP8/NVFP4 economics; MoE $/quality; muP transfer in the sim | ideal | update the cost/throughput models with FP8 + MoE |

---

## What shipped this month

Config-gated, default-off, full test coverage:

- **`nanogpt-edu`:** Muon (`muon.py`), QK-Norm / zero-init / untied embeddings,
  Multi-Token Prediction, FineWeb-Edu + DCLM BPE prep, new configs
  (`tiny_muon.py`, `tiny_mtp.py`, `small_fineweb.py`), multi-optimizer loop with
  shared cosine schedule + backward-compatible checkpoints. Tests 9 → 12.
- **`coder-finetune`:** Liger Kernel, NEFTune, DoRA, rsLoRA, Unsloth fast-path;
  `lora_5060ti.yaml` ships with the safe set enabled. Tests: 4.

Both repos: ruff-clean, end-to-end train/resume/sample smoke-verified.

---

## Sources

1. KellerJordan/modded-nanogpt — NanoGPT speedrun + technique list + record
   history. <https://github.com/KellerJordan/modded-nanogpt>
2. Keller Jordan, *Muon: An optimizer for hidden layers in neural networks*
   (Dec 2024). <https://kellerjordan.github.io/posts/muon/>
3. DeepSeek-AI, *DeepSeek-V3 Technical Report* — MTP (λ=0.3, 4 tokens), MLA,
   fine-grained MoE at 671B.
4. LinkedIn, *Liger-Kernel: Efficient Triton Kernels for LLM Training* —
   ~20%/~60%; arXiv 2410.10989. <https://github.com/linkedin/Liger-Kernel>
5. Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024).
6. Kalajdzievski, *rsLoRA: A Rank-Stabilized Scaling for LoRA* (2023).
7. Jain et al., *NEFTune: Noisy Embeddings Improve Instruction Finetuning* (2023).
8. Penedo et al., *FineWeb / FineWeb-Edu* (HuggingFaceFW).
   <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
9. Li et al., *DataComp-LM (DCLM-Baseline-1.0)*.
10. NVIDIA, *NVFP4 low-precision model training* — 1.59× on B200, near-bf16
    accuracy; arXiv 2509.25149.
11. Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony
    and Low-precision* (2024).
12. PyTorch, *FSDP2 / TorchTitan* — 3D parallelism + overlapped collectives for
    large-scale training.
13. Unsloth — single-GPU LoRA/QLoRA kernels. <https://unsloth.ai>

> **Methodology note.** Merges `distgpt/docs/sota_review.md` with a
> primary-source pass on the modded-nanogpt / Muon / Liger material. Where live
> search was rate-limited, primary sources (repos, blog, papers) were fetched
> directly. Hardware envelopes are aspirational sizing targets, not a constraint
> from any one machine. Flag anything stale for the next edition.
