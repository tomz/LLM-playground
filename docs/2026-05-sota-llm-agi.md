# SOTA Watch — LLM & AGI · 2026-05

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-05  ·  **Status:** published

> Inaugural edition. Theme: **train better models faster and cheaper on
> commodity GPUs.** This consolidates two prior internal research passes — a
> modded-nanogpt/Muon survey aimed at the from-scratch trainers, and an
> FSDP-scoped review originally written for `distgpt` — into one ranked,
> deduplicated digest. Where a technique was actionable on our hardware
> (RTX 3050 8 GB, RTX 5060 Ti 16 GB) we implemented it this month; the rest is
> tiered and put on a watchlist with explicit unblocking conditions.
>
> Audience: anyone training or fine-tuning sub-10B models on one or a few
> consumer/prosumer GPUs. The frontier-scale items are tracked but deferred —
> we don't have the silicon to validate them honestly.

---

## TL;DR — this month's harvest

- **Muon optimizer** shipped in `nanogpt-edu` (Newton-Schulz-orthogonalized
  updates for 2D hidden weights; AdamW for embeddings/head/norms). ~1.35×
  sample-efficiency on the FineWeb speedrun task; the single biggest
  algorithmic lever for from-scratch training.
- **Multi-Token Prediction** (DeepSeek-V3 style, simplified) shipped in
  `nanogpt-edu` — denser gradient signal, train-only, zero inference cost.
- **Liger Kernel + DoRA + rsLoRA + NEFTune** shipped as config knobs in
  `coder-finetune` — fused Triton kernels (~20% faster, up to ~60% less memory)
  plus three near-free quality/throughput upgrades that plug straight into the
  TRL/PEFT stack.
- **Data scaling** is the highest-ROI knob nobody flips: `nanogpt-edu` now has
  a FineWeb-Edu / DCLM BPE prep, fixing the "everything overfits 1 MB of
  Shakespeare" story documented in its README.
- **Deferred, on hardware grounds:** FP8/NVFP4 training, FlashAttention-3,
  Multi-head Latent Attention. All pay off on Hopper/Blackwell datacenter
  cards we don't run; flagged with unblocking conditions.

---

## Hardware reality check

Most of this matters because the techniques below split cleanly into "helps on
any GPU" vs "helps only on H100/B200." Our fleet:

| Card | Arch | VRAM | bf16 | FP8 | Role in this repo |
|------|------|-----:|:----:|:---:|-------------------|
| RTX 3050 | Ampere sm_86 | 8 GB | ✓ | ✗ | smoke runs, small LoRA, `nanogpt-edu` tiny |
| RTX 5060 Ti | Blackwell sm_120 | 16 GB | ✓ | HW yes / SW immature | main single-GPU workhorse (`midgpt` 350M, `distgpt` 416M, `coder-finetune` 3B) |
| 8× H100 / B200 | Hopper / Blackwell DC | 80 GB | ✓ | ✓ | **not owned** — reference recipes only |

Consequence: **FP8 matmul and FlashAttention-3 are excluded from every
recommendation below**, because they need Tensor Core generations our cards
don't have. We call this out per-item rather than silently dropping it.

---

## Tier 1 — high-ROI, low-risk (recommend doing)

### 1. Muon optimizer (hidden 2D weights) — **shipped**

Replaces the SGD/Adam update for 2D weight matrices with its nearest
semi-orthogonal matrix, computed by a 5-step Newton-Schulz iteration that runs
stably in bf16. Orthogonalizing the update amplifies the "rare directions" that
a near-low-rank momentum buffer otherwise drowns out.

- **Win:** ~1.35× sample-efficiency over a tuned AdamW on the FineWeb GPT-2
  task; the modded-nanogpt 124M speedrun (45 min → ~3 min on 8×H100) is largely
  Muon. Scales to 1.5B (GPT-2-XL quality in 10 vs 13.3 8×H100-hours). On the
  350M–1B runs in this repo, expect 1.3–1.8× wall-clock to a fixed loss.
- **Scope rule:** Muon for hidden matmul weights (attn q/k/v/o, MLP); AdamW for
  embeddings, lm_head, MTP heads, and all 1-D params (norms, biases).
- **Cost/risk:** ~80 lines (`nanogpt-edu/muon.py`); different LR scale than
  AdamW (use lr≈0.02, momentum 0.95). The 7B+ regime is less battle-tested.
- **Source:** [1], [2]  ·  **Harvest:** shipped → `nanogpt-edu`
  (`optimizer='muon'`, `configs/tiny_muon.py`). Candidate port to `midgpt` /
  `distgpt` (per-parameter, no FSDP conflict) — see watchlist.

### 2. Multi-Token Prediction (MTP) — **shipped**

Auxiliary heads predict tokens *n+2* (and *n+3*) alongside the standard *n+1*
head; loss = CE_next + λ·mean(CE_future), DeepSeek-V3 used λ=0.3.

- **Win:** denser gradient signal → better sample efficiency in early training
  (exactly where our tiny runs live); ~0.5–1 pp on downstream benchmarks at
  fixed compute at scale. The MTP head also enables speculative decoding later.
- **Cost/risk:** ~5% VRAM for the extra head, ~3% throughput. We simplified
  DeepSeek's per-depth transformer modules to plain linear heads for legibility.
- **Design notes (ours):** the aux loss is **train-only** (gated on
  `self.training`) so eval/val loss stays directly comparable to a non-MTP run;
  `generate()` uses only the main head → **zero inference cost**; MTP heads are
  routed to AdamW (output layers, not Muon).
- **Source:** [3]  ·  **Harvest:** shipped → `nanogpt-edu`
  (`mtp_tokens`/`mtp_weight`, `configs/tiny_mtp.py`).

### 3. Liger Kernel (fused Triton kernels) — **shipped**

One-line patch giving fused RMSNorm/RoPE/SwiGLU + `FusedLinearCrossEntropy`.

- **Win:** ~20% throughput, up to ~60% memory reduction, **exact** (not an
  approximation). The fused linear-CE is the standout for our use: it never
  materializes the full `[batch·seq, vocab]` logits tensor — a huge activation
  over Qwen2.5-Coder's ~150K vocab — directly buying longer context / bigger
  batch on an 8 GB card.
- **Cost/risk:** needs `pip install liger-kernel` and a Triton-capable GPU
  (works on our Ampere/Blackwell cards; no-op on CPU).
- **Source:** [4]  ·  **Harvest:** shipped → `coder-finetune`
  (`train.use_liger_kernel`, enabled in `configs/lora_5060ti.yaml`).

### 4. DoRA + rsLoRA + NEFTune (PEFT/TRL quality knobs) — **shipped**

Three near-free upgrades to the LoRA recipe:

- **DoRA** (weight-decomposed LoRA): better quality at low rank (our r=16), ~10–20% slower step.
- **rsLoRA** (rank-stabilized scaling, α/√r): lets higher ranks actually help; free.
- **NEFTune** (embedding noise during SFT): better instruction-following; free, train-only.
- **Source:** [5], [6], [7]  ·  **Harvest:** shipped → `coder-finetune`
  (`lora.use_dora`, `lora.use_rslora`, `train.neftune_noise_alpha`).

### 5. QK-Norm, zero-init projections, untied embeddings — **shipped**

The cheap, legible slice of the modded-nanogpt architecture set:

- **QK-Norm:** per-head RMSNorm on Q/K before RoPE — stabilizes attention
  logits, supports higher LR.
- **Zero-init residual-write matrices** (attn `o_proj`, ffn down-proj): each
  block starts as identity (muP-like), stable high-LR warmup.
- **Untied embeddings:** gives `lm_head` its own weight; helps loss once tokens
  support the extra params (the speedrun unties at 2/3 of training).
- **Source:** [1]  ·  **Harvest:** shipped → `nanogpt-edu` (`qk_norm`,
  `zero_init_proj`, `tie_embeddings`).

### 6. Data scaling: FineWeb-Edu / DCLM + over-train past Chinchilla — **shipped (prep) / planned (runs)**

The most-replicated single intervention in pretraining is *better tokens, more
of them*, not a fancier optimizer.

- **Quality:** DCLM-Baseline-1.0 beats FineWeb-Edu by ~10% on MMLU at matched
  compute; either gives ~30–50% sample-efficiency over unfiltered web text.
- **Quantity:** our examples sit at **0.005×–0.37× Chinchilla**. The modern
  small-model norm is 20–50× (Llama-3-8B ≈ 1875×). Over-training tiny models is
  correct, not wasteful.
- **Harvest:** shipped → `nanogpt-edu/prepare_fineweb.py` (`--dataset
  fineweb-edu|dclm`, GPT-2 BPE, same shard format; `configs/small_fineweb.py`).
  Planned: actually extend the `midgpt`/`distgpt` 5060 Ti runs to 6–12k steps
  (val ppl 60 → ~30, cheap chart upgrade) and publish.
- **Source:** [8], [9]

### 7. `torch.compile` + FP16 GradScaler for older cards — **already present / planned**

- `torch.compile(model)`: 10–30% wall-clock on consumer GPUs via graph fusion.
  Already wired in `nanogpt-edu` (`compile` flag) and `midgpt`.
- **FP16 GradScaler path:** present in `nanogpt-edu` (`enabled=(dtype==float16
  and is_cuda)`). The gap flagged in the original `distgpt` review — a Pascal
  P100 example would run ~1.8× faster in fp16 — is a **planned** `distgpt`
  port.
- **Source:** [1]

---

## Tier 2 — meaningful but heavier lifts

| Technique | What it does | Win | Cost / risk | Harvest |
|-----------|--------------|-----|-------------|---------|
| **Unsloth fast-path** | Custom autograd kernels for LoRA/QLoRA | ~2× faster, ~70% less memory on our model family | heavier dep; diverges from "just standard HF plumbing" teaching goal | **shipped** as opt-in loader in `coder-finetune` (`model.use_unsloth`); best for the 7B QLoRA recipe |
| **FlashAttention-3 / FlexAttention** | Faster attention kernels | ~1.5× SDPA — **H100 only** | no flash kernel on Pascal; ~5–10% headroom over SDPA on 5060 Ti | **deferred** — revisit with H100 access |
| **FP8 / NVFP4 training** (torchao / TransformerEngine) | 8-/4-bit matmul | ~1.5–1.6× throughput, ~2× memory — **Hopper/Blackwell DC** | per-tensor scaling state, selective bf16 layers, convergence subtleties at small batch | **deferred** — bf16 saturates our cards |
| **Multi-head Latent Attention (MLA)** | Low-rank KV compression | 5–10× smaller KV cache | training win modest; breaks checkpoint compat; ~150 LOC | **deferred** — revisit when we have a serving story |
| **Spectrum / targeted full-FT** | Full-FT only high-SNR layers, freeze rest | sometimes beats LoRA at similar memory | layer-selection heuristic per model | **planned** — candidate for `coder-finetune` `tiny.yaml` tier |

---

## Tier 3 — research bets (track, don't build yet)

- **Mixture-of-Experts** at <1B active params underperforms dense — revisit at ≥3B.
- **Hybrid Mamba-Transformer** (used in NVIDIA's 12B NVFP4 work): interesting,
  but adds significant arch complexity and the recipe is poorly documented
  outside that paper.
- **Diffusion / next-byte LMs:** pre-product.
- **Long-context position schemes** (YaRN / NTK-by-parts, NoPE in alternating
  layers): relevant only once we train at block_size ≫ 1024.

---

## Watchlist / deferred (with gating condition)

Carry this table forward each month — the "unblocks when" column turns
"someday" into a trigger.

| Technique | Why deferred | Unblocks when |
|-----------|--------------|---------------|
| FP8 / NVFP4 training | Needs Hopper/Blackwell DC Tensor Cores; our 5060 Ti has the HW but the SW stack is immature | We get H100/B200 time, or torchao FP8 on sm_120 stabilizes |
| FlashAttention-3 | H100-only kernel; SDPA already near-optimal on our cards | H100 access for the reference recipes |
| MLA | Inference-focused; breaks GQA checkpoint compat | We stand up a serving path worth optimizing |
| Muon in `midgpt`/`distgpt` | Implemented in `nanogpt-edu` only so far | Next porting pass — per-parameter, FSDP-safe, low risk |
| FP16 GradScaler in `distgpt` | Trainer assumes bf16, no scaler | Redo the Pascal P100 example (~1.8× faster in fp16) |
| Extend 5060 Ti runs past 1× Chinchilla | Just wall-time | A free afternoon — val ppl 60 → ~30, much stronger chart |

---

## What shipped this month

Two commits landed the Tier-1 harvest (config-gated, default-off, full test
coverage):

- **`nanogpt-edu`:** Muon optimizer (`muon.py`), QK-Norm / zero-init / untied
  embeddings, Multi-Token Prediction, FineWeb-Edu + DCLM BPE prep
  (`prepare_fineweb.py`), new configs `tiny_muon.py` / `tiny_mtp.py` /
  `small_fineweb.py`, multi-optimizer training loop with shared cosine schedule
  and backward-compatible checkpoints. Tests: 9 → 12 passing.
- **`coder-finetune`:** Liger Kernel, NEFTune, DoRA, rsLoRA, Unsloth fast-path
  loader — all opt-in YAML knobs; `lora_5060ti.yaml` ships with the safe set
  enabled. Tests: 4 passing.

Both repos: ruff-clean, end-to-end train/resume/sample smoke-verified.

---

## Sources

1. KellerJordan/modded-nanogpt — NanoGPT speedrun (124M to llm.c target in
   ~3 min on 8×H100), technique list + world-record history.
   <https://github.com/KellerJordan/modded-nanogpt>
2. Keller Jordan, *Muon: An optimizer for hidden layers in neural networks*
   (Dec 2024). <https://kellerjordan.github.io/posts/muon/> ·
   repo <https://github.com/KellerJordan/Muon>
3. DeepSeek-AI, *DeepSeek-V3 Technical Report* — Multi-Token Prediction (λ=0.3,
   4 future tokens) + MLA at 671B scale.
4. LinkedIn, *Liger-Kernel: Efficient Triton Kernels for LLM Training* —
   ~20% throughput / ~60% memory; arXiv 2410.10989.
   <https://github.com/linkedin/Liger-Kernel>
5. Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024).
6. Kalajdzievski, *rsLoRA: A Rank-Stabilized Scaling for LoRA* (2023).
7. Jain et al., *NEFTune: Noisy Embeddings Improve Instruction Finetuning* (2023).
8. Penedo et al., *FineWeb / FineWeb-Edu* (HuggingFaceFW).
   <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
9. Li et al., *DataComp-LM (DCLM-Baseline-1.0)* — beats FineWeb-Edu ~10% on
   MMLU at matched compute.
10. NVIDIA, *NVFP4 low-precision model training* — 1.59× throughput on B200,
    near-bf16 accuracy; arXiv 2509.25149 (validating 4-bit pretraining on
    Blackwell, used for the FP8/NVFP4 watchlist item).

> **Methodology note.** This edition merges `distgpt/docs/sota_review.md` with a
> primary-source pass on the modded-nanogpt / Muon / Liger material. Where live
> web search was rate-limited, we fetched primary sources (repos, blog, papers)
> directly and leaned on previously verified references. Flag anything stale and
> it'll be corrected in the next edition.
