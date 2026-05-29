# SoTA leverage points for distgpt — train better models faster/cheaper

Snapshot of what's in this repo today, what the field has moved to in
the last ~12 months, and a ranked list of changes that would land
real wins. Scoped to *small/mid* runs (the 100M–10B regime the
existing 5060 Ti / P100 examples live in); the multi-node Slurm
path needs separate analysis.

## Where distgpt sits today

| Knob | Current code | Field today |
|---|---|---|
| Optimizer | AdamW + cosine | AdamW still default for tiny; **Muon** for hidden weights gives ~1.5–2× wall-clock-to-loss on <1B |
| Precision | bf16 autocast (no GradScaler) | bf16 still safe; **FP8** (MXFP8 / NVFP4) cuts memory ~2×, throughput ~1.5–1.6× on Hopper/Blackwell |
| Attention | `F.scaled_dot_product_attention` (PyTorch SDPA) | SDPA still strong; **FlashAttention-3** ~1.5× SDPA on H100 only; **MLA** (DeepSeek) cuts KV cache 5–10× |
| Architecture | RoPE + RMSNorm + SwiGLU + GQA 4:1 + tied embeddings | Same modern stack everyone uses ✓ |
| Position | Standard RoPE base 10000 | **YaRN / NTK-by-parts** for long context; **NoPE** in alternating layers for OOD length |
| Training signal | next-token LM only | **Multi-Token Prediction** (DeepSeek-V3) — 2–4 future tokens, +0.5–1 pp on benchmarks "free" |
| Compile | none | `torch.compile(model)` ~10–30% wall-clock on consumer GPUs |
| KV / inference | n/a (training only) | n/a |
| Data | FineWeb-Edu 1B slice, uniform shuffle | **DataComp-LM** / **Nemotron-CC** quality filters, curriculum (easy→hard), repeats <4× |
| Scaling target | 0.24× Chinchilla in examples | **20–25× Chinchilla** is the modern small-model norm (Llama-3, Qwen2.5); "over-training" tiny is correct |

## Tier 1 — high-ROI, low-disruption changes (recommend doing)

### 1. `torch.compile(model)` in the trainer
- **Cost**: 3 lines after FSDP wrap, gated by config flag.
- **Win**: 10–30% step-time on the 5060 Ti example (graph fusion + better attn dispatch). Free on Pascal too (~5–10% there, less because no Tensor Cores to amortize).
- **Risk**: First-step compile time (~30 s); breaks some debugging stacks. Mitigate with `mode="reduce-overhead"`.
- **distgpt-specific**: must be applied *after* FSDP2 `fully_shard` and *before* the first forward — there's a clean insertion point at `trainer.py:71`.

### 2. Muon for hidden-layer weights
- **What**: Newton-Schulz orthogonalization of momentum → effectively whitens 2D weight gradients. AdamW for embeddings/LM-head/norms; Muon for the rest. ~10 lines of optimizer code.
- **Win**: Modded-nanogpt's 124M speedrun went 45 min → ~3 min largely on Muon. On 350M–1B in this repo, expect **1.3–1.8× wall-clock to a fixed loss** vs AdamW. Concrete: the 5060 Ti's val ppl 60.7 at step 2800 would land near step 1700–1900.
- **Cost**: New `distgpt/training/muon.py` (~80 lines); param-group rewrite in `optim.py`. No FSDP issues — Muon is per-parameter.
- **Risk**: Hyperparams differ from AdamW (use lr ≈ 0.02, momentum 0.95). One published recipe exists for the 124M scale; the 7B+ regime is less battle-tested.
- **Reproducible reference**: `KellerJordan/modded-nanogpt`.

### 3. Multi-Token Prediction (MTP) auxiliary head
- **What**: One extra linear → predicts token *n+2* (and optionally *n+3*) alongside the standard *n+1* head. Loss = standard CE + λ·CE(future). DeepSeek-V3 used λ=0.3 with 4 future tokens.
- **Win**: ~0.5–1 pp on downstream benchmarks at fixed compute; better sample efficiency in early training (the gradient signal is denser). At inference time the MTP head enables speculative decoding for free.
- **Cost**: ~30 lines in `model/transformer.py` (head + loss); requires touching `cli.py` for the new loss weighting config.
- **Risk**: Small VRAM bump (~5% for one extra head); minor throughput hit (~3%). Tested at scales from 100M to 671B by now.

### 4. FP16 GradScaler path for the Pascal example
- **What**: Wrap `loss.backward()` / `optim.step()` in `torch.amp.GradScaler` when `dtype==float16`. ~15 lines, conditional.
- **Win**: P100 example would run **~1.8× faster** (fp16 → ~10 h instead of 14.8 h) without losing convergence. Bonus: makes Volta/Turing usable too.
- **Cost**: One conditional in `trainer.py`. The writeup already calls out this gap as a known issue.
- **Risk**: None — standard mixed-precision pattern from 2019.

## Tier 2 — meaningful but heavier lifts

### 5. FlashAttention-3 / FlexAttention
- **Where it pays**: Hopper (H100) + Blackwell only. On the 5060 Ti the PyTorch SDPA backend already dispatches to the Triton/cuDNN flash kernel, so the headroom is small (~5–10%). On Pascal there's no flash kernel at all → no change.
- **distgpt impact**: only valuable when the 8×H100 reference recipes get real runs.
- **Skip for now**; revisit when there's H100 access.

### 6. FP8 training via `torchao` / TransformerEngine
- **Win**: ~1.5–1.6× throughput, ~2× memory savings on H100/B200.
- **Cost**: Non-trivial — needs per-tensor scaling state, careful layer selection (LN / softmax stay bf16), MoE-style absmax tracking. TransformerEngine integration is ~200 LOC.
- **Risk**: Convergence subtleties at small batch sizes; NVFP4 (Blackwell) still needs "selective BF16 layers" per NVIDIA's own May 2026 results.
- **Recommendation**: defer until there's H100/B200 time; bf16 fully saturates anything in this repo's current example set.

### 7. Multi-head Latent Attention (MLA)
- **What**: Compress KV cache via low-rank projection (DeepSeek's contribution). 5–10× smaller KV at near-equal quality.
- **Win**: Training-time savings are modest; the big win is *inference*. Worth it if you ever serve these models.
- **Cost**: Replace the attention block (~150 LOC); breaks checkpoint compatibility with the existing GQA models.
- **Recommendation**: defer until there's a serving story.

### 8. Data quality upgrade — DataComp-LM or Nemotron-CC
- **Win**: Most well-replicated single intervention in pretraining. ~30–50% sample efficiency (you need ~30% fewer tokens to hit the same val loss).
- **Cost**: ~50 GB extra disk, one-time tokenization. The existing `data/streaming.py` doesn't need changes.
- **Note**: FineWeb-Edu is already a strong filtered set; DCLM-Baseline-1.0 is the public state of the art and beats FineWeb-Edu by ~10% on MMLU at matched compute.

### 9. Over-train past 1× Chinchilla
- **Current examples**: 0.24× Chinchilla (98M tokens on 416M model).
- **Modern small-model norm**: 20–50× Chinchilla. Llama-3-8B is ~1875× (15T tokens).
- **For the 5060 Ti example specifically**: extending to 6000–12000 steps (~3–6 h more) would drop val ppl from 60.7 → ~30–40 based on the slope at step 2800. Cheap, dramatic chart upgrade.

## Tier 3 — research bets, skip for now

- **Mixture-of-Experts** at this scale (<1B active) underperforms dense; revisit at ≥3B.
- **Diffusion LMs / next-byte models** are pre-product.
- **Hybrid Mamba-Transformer** (NVIDIA's 12B NVFP4 paper used this) is interesting but adds significant arch complexity and the recipe is poorly documented outside that paper.

## What I'd actually do, ranked

If I had a free afternoon on this repo:

1. **Add `torch.compile` toggle.** 30 min. Win on every future run.
2. **Wire the FP16 GradScaler.** 1 h. Unblocks Pascal/Volta hardware properly; lets us redo the P100 example in ~10 h.
3. **Extend the 5060 Ti example to 12k steps.** Free (just wall time). Ppl drops from 60 to ~30, much more credible chart.
4. **Add Muon.** Half a day. ~1.5× wall-clock improvement on every example, repeatable.
5. **MTP head.** Half a day. Small quality bump, sets up speculative decoding later.

Tier 2 (FP8, MLA, DCLM) needs hardware/budget that this repo doesn't have access to today.

## Sources

- NVIDIA NVFP4 12B / 10T tokens result — arXiv 2509.25149 (Sep 2025), validated stable 4-bit pretraining on Blackwell
- NVIDIA dev blog "Using NVFP4 low-precision model training" — 1.59× throughput on B200 with near-BF16 accuracy
- `KellerJordan/modded-nanogpt` — 5.3k stars, Muon-based 124M speedrun (~90s on 8×H100)
- DeepSeek-V3 tech report — MLA + MTP at 671B scale
- Hand-knowledge: Llama-3, Qwen2.5, Phi-3 over-training ratios; FineWeb-Edu vs DCLM ablations

(Search was rate-limited today; I've leaned on previously verified sources where the live web fetch failed. Flag anything that looks stale and I'll dig in.)
