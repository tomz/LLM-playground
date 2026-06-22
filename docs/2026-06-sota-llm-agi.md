# SOTA Watch — LLM & AGI · 2026-06

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-06  ·  **Status:** published

> Second edition. Theme: **stop harvesting, start measuring.** May catalogued
> the techniques; June *ran the experiments*. The through-line this month is
> empirical validation — two controlled, reproducible A/Bs on a modest 2× RTX
> 5060 Ti box that put real numbers on claims we had only been citing:
> (1) the **Llama recipe** (RoPE + RMSNorm + SwiGLU + QK-norm) beats GPT-2 by
> **16.8 % perplexity** at iso-param / iso-token, and (2) **FSDP2 over a
> no-NVLink PCIe pair** can be flipped from *negative* (0.69×) to *positive*
> (1.28×) scaling with two specific config/code changes. Alongside the runs:
> a deep harvest of one new frontier paper — Microsoft AI's **MAI-Thinking-1**
> "hill-climbing" reasoning report — landing ten reusable components into
> `frontier-platform`, and the closing-out of May's roadmap (SimPO/KTO, DAPO
> reward shaping, FlexAttention, self-play, distributed-Muon knobs). And a
> forward scan of the frontier — LoRA Without Regret, DeepConf, RLPR / GSPO,
> and an open-weight wave (Kimi K2 Thinking, DeepSeek-V3.2, Qwen3-Next,
> gpt-oss, …) — sets the next round of harvests and recalibrates the roadmap.
>
> **Framing (unchanged):** content is *not* constrained by any one
> workstation's GPUs. For each project we assume both a **minimal** box and an
> **ideal** box and recommend what is correct at that scale. Hardware is a
> sizing note, never a blocker. The May edition
> ([`2026-05-sota-llm-agi.md`](./2026-05-sota-llm-agi.md)) remains the canonical
> deep catalogue of each technique's rationale; this edition records **June
> deltas** — what we measured, what we harvested, and what changed status.

---

## TL;DR — this month's harvest

- **We ran the architecture A/B (midgpt).** A controlled, iso-param /
  iso-token, single-node-DDP comparison at 350 M scale on FineWeb-Edu:
  **llamafied (RoPE + RMSNorm + SwiGLU + QK-norm) reaches val ppl 48.1 vs
  GPT-2's 57.8 — a 16.8 % improvement**, leading at *every one* of 19 evals
  (by up to 40 % early), and reaching GPT-2's *final* quality ~37 % sooner.
  The win is real but bounded: B costs ~23 % throughput and ~1 GB VRAM. A
  citable systems finding fell out of it — **iso-param ≠ iso-activation**
  (SwiGLU's 3 resident FFN activations + QK-norm forced a micro-batch cut).
- **We ran the FSDP2 scaling study (distgpt).** Genuine 2-GPU FSDP2 over a
  **no-NVLink PCIe pair**. Naively, two GPUs were *slower* than one (**0.69×**,
  per-GPU MFU collapsing 16.2 → 5.6 %). Two changes — `reshard_after_forward=
  false` and **gating gradient-sync to the last micro-step** — halved the
  collective volume and flipped it to **1.28×** (64 % efficiency). The full
  4 500-step run landed at **val ppl 41.6** (295 M tokens, 0.71× Chinchilla),
  exactly on the single-GPU run's forecast. The **PCIe tax is an interconnect
  property, not a code one** — the same trainer on NVLink reclaims the points.
- **MAI-Thinking-1 harvested → `frontier-platform`.** Microsoft AI's from-scratch
  1T-total / 35B-active MoE *reasoning* report yielded **ten** pure-Python,
  CPU-testable components across three tiers: IFEval-style constraint verifiers,
  **adaptive entropy control** (PI controller vs entropy collapse),
  language-consistency + difficulty-aware length rewards, dual-clip surrogate,
  a **reasoning-trace archetype rubric**, long-context eval adapters, a
  **Pareto-percentile release gate**, the **hill-climb orchestrator**
  (specialists → distill → climb), and **zero-init attention output**. 72 new
  tests; zero-init attn cross-pollinated into `midgpt`.
- **May's roadmap items shipped.** **SimPO / KTO** (reference-free preference
  objectives, `coder-finetune/cf_pref`), **DAPO** overlong-reward-shaping +
  dynamic-sampling + clip-higher (`cf_rl`), **FlexAttention** backend switch
  (`midgpt` + `nanogpt-edu`), a deterministic **self-play loop**
  (`frontier-platform/rl/selfplay.py`), opt-in **distributed-Muon** weight-decay
  / update-scale (`distgpt`), and an **HF-export validator** with optional vLLM
  smoke (`midgpt`).
- **Multi-GPU went democratic (coder-finetune).** Single-node **DDP**
  (data-parallel) for SFT/DPO/GRPO via a new `cf_dist` topology module, proven
  on real hardware: a 2× 5060 Ti, **real-NCCL** LoRA run (loss 3.38 → 0.74, the
  `reducer.cpp` DDP hook firing across both cards) — a correctness proof to
  complement distgpt's sharding study.
- **Meta-theme:** the repo crossed from *"implements the technique"* to
  *"measures the technique."* Every Tier-1 architecture claim now has an
  in-repo number behind it, reached through ≥2 independent codebases that land
  on the **same FineWeb-Edu scaling curve**.
- **The forward scan — and two of them landed.** This edition also scoped the
  next round of harvests; **two were not just scoped but shipped *and measured***
  this pass (✅, see §§12–13): **LoRA Without Regret** (TML — the r=16-vs-r=256
  A/B **ties at convergence**, with an undertraining-trap lesson) and **DeepConf**
  test-time confidence filtering (confidence tracks correctness; online
  early-abort trades ~10 % tokens for ≈iso-accuracy; the offline vote-lift is a
  large-k/long-trace *sizing fact*). The rest remain `🔜 planned` / `track` /
  `ideal` — *researched, not yet run*: **verifier-free RLVR** (RLPR) + **GSPO**
  (both now *implemented* in `frontier-platform`, see §14) production adoption;
  **native INT4-via-QAT** serving (Kimi K2 Thinking) and **hybrid
  linear-attention + ultra-sparse MoE** (Qwen3-Next, Nemotron Nano 2) promoted
  Tier 3 → Tier 2; and an **open-weight frontier wave** (Kimi K2 Thinking,
  DeepSeek-V3.2, gpt-oss, GLM-4.6, MiniMax-M2, Olmo 3, nanochat) as sizing /
  north-star evidence. See the §§ 12–14 entries and the open-weight
  frontier-wave table below (sources [38]–[49]).

---

## Hardware envelopes per project

Unchanged from May — aspirational targets, not a description of any one machine.
"Minimal" runs a meaningful job; "ideal" unlocks the full technique set. **This
month's empirical runs all used the *minimal* tier** (1–2× RTX 5060 Ti 16 GB),
which is precisely why the systems findings (activation-memory edge, PCIe
collective tax) surfaced — they are exactly the constraints the *ideal* tier
makes disappear.

| Project | Scale | Minimal | Ideal | What ideal unlocks |
|---------|-------|---------|-------|--------------------|
| **nanogpt-edu** | 10M–100M | laptop CPU / 8 GB GPU | 1× H100 80 GB | bf16→FP8 head, long-ctx FlexAttention, fast full sweeps in minutes |
| **midgpt** | 124M–1.5B | 1× 16 GB GPU | 8× H100 single node | FP8 matmul, FlashAttention-3, real Chinchilla-optimal token budgets |
| **distgpt** | 1B–70B | 1 node × 8× A100 | 8–64 nodes × 8× H100/B200 + NVLink/InfiniBand | 3D/5D parallelism, FP8, comms/compute overlap, MoE expert parallelism |
| **coder-finetune** | 0.5B–7B | 1× 8 GB GPU (QLoRA) | 1× 80 GB or 2–8× for full-FT | full-FT of 7B, long-context packing, fast multi-epoch SFT + DPO/ORPO/GRPO |
| **frontier-platform** | 1B–500B+ | design doc (no GPU) | 1k–16k× H100/B200 | the entire program: pretrain → SFT → RLVR/RLHF → eval → vLLM/SGLang-style serve at frontier scale |

---

## Tier 1 — high-ROI, broadly applicable

> Full per-technique rationale lives in the [May edition](./2026-05-sota-llm-agi.md#tier-1--high-roi-broadly-applicable).
> Here we note **June deltas** and add the new entries. Status legend:
> ✅ shipped · 🟡 partial · 🔜 planned.

| # | Technique | June status | Δ this month |
|---|-----------|-------------|--------------|
| 1 | **Muon optimizer** | ✅ nanogpt-edu, midgpt, frontier | **distgpt** gained opt-in `weight_decay` + `update_scale` (Moonshot *Muon is Scalable* fixes); defaults preserve the exact prior update, pinned by test |
| 2 | **Multi-Token Prediction** | ✅ nanogpt-edu, frontier | unchanged (serving benchmark stands) |
| 3 | **Liger fused kernels** | ✅ coder-finetune, midgpt | unchanged |
| 4 | **DoRA + rsLoRA + NEFTune** | ✅ coder-finetune | unchanged |
| 5 | **QK-Norm, zero-init proj, untied emb** | ✅ all four core repos | **empirically validated** — the QK-norm + RoPE + RMSNorm + SwiGLU stack *won* the 350 M A/B by 16.8 % ppl (§ below). **Zero-init attention output** added (MAI harvest), cross-pollinated nanogpt-edu↔midgpt↔distgpt↔frontier |
| 6 | **Data scaling (FineWeb-Edu/DCLM, over-train)** | ✅ prep | two more points on the curve: 350 M @ 131 M tok (ppl 48.1) and 416 M @ 295 M tok (ppl 41.6) |
| 7 | **`torch.compile` + mixed precision** | ✅ present | unchanged |
| 8 | **RLVR + GRPO (+ successors), DPO/ORPO** | ✅ → **expanded** | **DAPO** (clip-higher + dynamic sampling + overlong reward shaping) and **SimPO/KTO** (reference-free) shipped into `coder-finetune` — closes the single biggest May roadmap item. **Adaptive entropy control** + **dual-clip surrogate** + **language/length reward shaping** landed in `frontier-platform` (MAI harvest) |
| 9 | **Serving-aware training (vLLM/SGLang, spec-decode)** | 🟡 in progress | **HF-export validator** (`midgpt/tools/validate_hf_export.py`) with optional vLLM smoke — the first concrete rung of the train→serve bridge |
| 10 | **Agentic post-training + self-play** | 🟡 → **expanded** | deterministic **self-play / evolutionary loop** shipped (`frontier-platform/rl/selfplay.py`, AlphaEvolve/SPIN-shaped); MAI's **hill-climb orchestrator** (specialists → distill → climb) shipped |
| **11** | **🆕 From-scratch reasoning via "hill-climbing" (MAI-Thinking-1)** | ✅ frontier (10 components) | **new** — see below |
| **12** | **🆕 LoRA Without Regret — LoRA *matches* full-FT, configured right** | ✅ shipped + measured (coder-finetune) | **new** — config + pins shipped; 3 A/Bs: tie at convergence, r=16 wins at fixed epoch budget (§ below) |
| **13** | **🆕 DeepConf — test-time confidence filtering of reasoning traces** | ✅ shipped + measured (nanogpt-edu) | **new** — bench shipped; measured on a verifiable toy (§ below) |
| **14** | **🆕 Verifier-free RLVR (RLPR) + GSPO going to production** | ✅ shipped + measured (frontier) / track (coder-finetune) | **new** — both in `frontier-platform/rl`; GPU run confirms GSPO ~4× lower-variance + RLPR sharpens reward (§ below) |

### § The empirical headline: the Llama recipe wins at 350 M — **shipped + measured**

The single most important thing the repo did this month was *run the A/B that
the §5 entry had only asserted*. Two arms, **identical** in everything but
architecture, both trained from random init through `midgpt`'s single-file GPT
on **2× RTX 5060 Ti** under one DDP harness:

| Metric | **A — GPT-2** (learned-pos · LN · GELU) | **B — llamafied** (RoPE · RMSNorm · SwiGLU · QK-norm) | Δ |
|---|---:|---:|---:|
| Params (iso) | 354.60 M | 353.51 M | 0.31 % apart |
| Tokens trained | 131 M (iso) | 131 M (iso) | — |
| **Best val ppl** | **57.8** | **48.1** | **−16.8 %** |
| Best val loss | 4.0562 | 3.8728 | −0.183 nats |
| Throughput | 19.3 k tok/s | 14.8 k tok/s | −23 % |
| Peak VRAM/GPU | ~11.9 GB | ~13.0 GB | +1.1 GB |

B leads at **all 19 evals** — by ~40 % at iter 400, narrowing to a stable ~17 %
as both saturate — and **passes A's best-ever val by iter ~2 400** (≈37 % fewer
tokens). There is no checkpoint at which GPT-2 was the right choice.

**The systems lesson (citable):** iso-param is **not** iso-activation. Arm B
OOM'd twice at `F.cross_entropy` before training cleanly — SwiGLU keeps *three*
FFN activations resident (vs GELU's one), QK-norm adds two RMSNorm activations
per block, and the fp32 logit transient spikes. The fix preserved the iso-token
contract *exactly* (`micro_batch 4→2`, `grad_accum 4→8` → same 32 768 tok/step,
same gradient, same schedule) while halving the activation footprint. **On a
16 GB card you pay for the Llama recipe's quality partly in activation memory →
micro-batch → throughput; on an A100/H100 the headroom makes it free.** Full
writeup: [`midgpt/examples/5060ti_350m_llamafied_AB.md`](../midgpt/examples/5060ti_350m_llamafied_AB.md).

### § 11 — From-scratch reasoning via "hill-climbing" (MAI-Thinking-1) — **shipped (frontier-platform)**

Microsoft AI's first in-house frontier *reasoning* model: a **35 B-active /
1 T-total MoE** trained **from scratch** (no third-party CoT distillation),
reaching **52.8 % SWE-Bench Pro, 97.0 % AIME 2025, 87.7 % LiveCodeBench v6**.
Its framing metaphor — *hill-climbing*: build a base model, then iteratively
climb via RL, training domain **specialists**, **distilling** them back
together, and repeating — is a clean, named alternative to monolithic training,
and the report is unusually candid (real cutoffs, token counts, self-recovering
loss spikes, dead-end ablations).

- **Win:** demonstrates **reproducible-in-principle** frontier reasoning without
  distilling from GPT/Claude; the reasoning-trace taxonomy ("weak models guess,
  strong models work hard / find invariants / are skeptics") is concrete
  evidence of *what* improves when a model gets smarter.
- **Harvested into `frontier-platform`** (ten components, all pure-Python /
  CPU-testable, each behind an existing protocol so a research org can fill
  content without touching platform code):
  - **Tier 1 (RLVR recipe):** IFEval-style `ConstraintFollowingVerifier`
    (16-checker registry); **`EntropyController`** (PI + anti-windup, targets a
    setpoint H* to avert entropy collapse); **language-consistency reward**;
    **difficulty-aware length penalty**; **dual-clip surrogate** (PPO floor on
    negative-advantage tokens).
  - **Tier 2 (eval + safety):** **reasoning-trace archetype rubric**
    (deterministic detectors for backtracking / verification / invariant-seeking
    / self-skepticism / agentic unit-testing); long-context eval adapters
    (Code-NLL, Retrieval-NLL, accuracy-by-depth); **Pareto-percentile release
    gate** (thresholds at a fixed percentile of the *currently-achievable*
    fleet).
  - **Tier 3 (recipe + arch):** **hill-climb orchestrator** (`rl/hillclimb.py`,
    specialists → distill → climb, end-to-end on CPU); **zero-init attention
    output** (`o_proj` zeroed so early attention noise can't perturb MoE
    routing) — also **cross-pollinated into `midgpt`** for parity.
- **Min HW:** any (CPU-testable scaffolds). **Ideal:** the full recipe assumes a
  GB200 NVL72-class cluster, FP8 GEMMs, and a ~30 000-CPU-core SWE-environment
  build — first-class *as a sizing fact*, not a blocker.
- **Source:** [37]  ·  **Harvest:** ✅ `frontier-platform` (72 new tests, suite
  → ~409 passing); zero-init attn → `midgpt`. Deep dive:
  [`docs/research/mai-thinking-1-deep-dive.md`](./research/mai-thinking-1-deep-dive.md).
  **Roadmap:** the long-context staged-extension recipe (16K→64K→256K, 140B-token
  tail) is **now written** as a standalone engineering note
  ([`docs/research/staged-long-context-extension.md`](./research/staged-long-context-extension.md));
  YOLO/SEE/FP8 infra is ideal-scale and out of scope here.

### § 12 — LoRA Without Regret: LoRA *matches* full fine-tuning — **shipped + measured (coder-finetune)**

Thinking Machines Lab (Schulman et al., 2025) ran a systematic SFT-and-RL sweep
to find *when* LoRA equals full fine-tuning (FullFT) — and the answer is
"more often than the folklore says, if you configure it right." This was the most
directly actionable of this edition's harvests because `coder-finetune` is *exactly*
a LoRA-on-TRL stack, and the paper's recommendations partly **disagreed with our
prior configs** — so it was a re-tune, not just a new knob.

- **The four findings, and where each lands on our configs:**
  1. **Apply LoRA to *all* linear layers, not just attention.** Attention-only
     LoRA underperforms even at matched parameter count via higher rank — the
     MLP/MoE matrices carry the capacity. Our configs already target
     `gate/up/down_proj` alongside attention (`lora_5060ti.yaml`), so we're on
     the right side of this — worth pinning a test that nobody narrows it back
     to attention-only.
  2. **Give the adapter enough capacity for the dataset.** Recommended rank
     **≈256 for post-training-scale SFT**; our default is **r=16**. For a 2.5 k-row
     Python SFT that's defensible, but the finding says *"for datasets that
     exceed LoRA capacity, LoRA underperforms FullFT"* — so a `lora_hicap.yaml`
     (r≈128–256) is the correct recipe the moment we train on a real
     instruction mixture.
  3. **RL needs almost no rank.** Policy-gradient extracts ~1 bit/episode, so
     **r=1–32 suffices for GRPO/DPO** — meaning our RL LoRAs can be *smaller and
     cheaper* than the SFT ones. Directly relevant to `cf_rl`/`cf_pref`.
  4. **Use a higher LR than FullFT, ~rank-independent**, and **keep effective
     batch < 32** (LoRA is less batch-tolerant). The `1/r` scaling makes the
     optimal LoRA LR roughly rank-independent; the TML reproduction used
     `lr 1e-5` (LoRA) vs `1e-6` (FullFT) at batch≈8–32.
- **Win:** LoRA reaches FullFT quality at **~⅔ the compute** and a fraction of
  the memory — i.e. the "regret" people accept for using LoRA was a
  *mis-configuration*, not a law. Turns the existing PEFT path from "cheap but
  worse" into "cheap and equal" across the post-training-scale regime that
  `coder-finetune` actually lives in.
- **Min HW:** unchanged — single 8–16 GB GPU (this is a *recipe*, not a kernel).
  **Ideal:** the r≈256 SFT recipe wants a 24 GB+ card or the multi-epoch budget
  of the 5060 Ti tier.
- **Source:** [38]  ·  **Harvest:** ✅ shipped + measured → `coder-finetune`.
  **Shipped:** `configs/lora_hicap.yaml` (r=256, all-linear, rsLoRA, rank-
  independent LR, effective batch 16) + `tests/test_lora_without_regret.py`
  pinning all four findings (all-linear across *every* LoRA config; hicap high-
  rank; LR not lowered for rank; RL adapters stay r≤32) + a README recipe section.
  **Measured** (2× 5060 Ti, Qwen2.5-Coder-0.5B, 8 k-row Magicoder-Python,
  iso-everything-except-rank `ab_lora_r16` vs `ab_lora_r256`, one arm per GPU):
  [`examples/lora_without_regret_ab.md`](../coder-finetune/examples/lora_without_regret_ab.md).
  **At 3 epochs the arms tie** — final token-acc **0.8476 (r=256) vs 0.8469
  (r=16)** — reproducing the thesis: high rank *matches*, doesn't beat, on a
  task r=16 already has capacity for ("no regret," not "free lunch"). **At
  1 epoch r=16 wins (0.83 vs 0.76)** because the 16× larger r=256 adapter (141 M
  vs 8.8 M params) is **undertrained**; and on a bigger, harder **30 k × 9-language**
  mixture at 2 epochs **r=16 still wins** (token-acc 0.840 vs 0.800, and the only
  arm that writes correct held-out Rust/TS) — because the larger dataset also
  gives the big adapter more to learn in the same budget. The sharpened lesson:
  **the binding constraint is training *budget*, not dataset size** — at every
  fixed epoch count tried (1/2/3), r=256's slow warmup leaves it tied-or-behind;
  the predicted r=256 > r=16 *separation* lives at iso-*convergence* on a 24 GB+ /
  many-epoch budget. Two methodology lessons pinned: compare ranks at
  iso-convergence (not iso-epoch); trust token-accuracy over `train_loss` (the
  rsLoRA `alpha/√r` scaling makes loss incomparable across ranks). Mirrors HF
  TRL's [`lora_without_regret`](https://huggingface.co/docs/trl/main/lora_without_regret).

### § 13 — DeepConf: test-time confidence filtering of reasoning traces — **shipped + measured (nanogpt-edu)**

DeepConf (Meta AI / UCSD, Aug 2025) is a **parallel-thinking** decoder that uses
the model's *own* token-confidence (per-token logprob / entropy, aggregated over
sliding windows) to **drop low-confidence reasoning traces** — during generation
(early-abort a doomed chain) or after (confidence-weighted majority vote). It
needs **no training and no extra hyperparameters**, and slots into an existing
serving loop.

- **Win:** up to **99.9 % on AIME 2025** (with GPT-OSS-120B) while **cutting
  generated tokens by up to 84.7 %** vs vanilla self-consistency — i.e. it makes
  test-time scaling *cheaper and better at once*, attacking the diminishing-
  returns problem of plain majority voting. It is the natural test-time complement
  to the train-time entropy work we already shipped (MAI `EntropyController`):
  same signal (confidence/entropy), opposite end of the pipeline.
- **Why it fits us:** it's pure-Python, CPU-testable on a tiny model, and reuses
  machinery we already have — `nanogpt-edu`'s sampler already computes logits, and
  the MTP draft-head benchmark (`tools/bench_mtp_spec.py`) is the obvious sibling.
  At the frontier end it composes with the rollout/serving-economics simulator
  (a confidence-gated *n*-sample vote is a priceable inference strategy).
- **Scope rule:** offline (confidence-weighted vote over *k* finished traces) is
  the trivially-correct first rung; online early-abort (kill a trace once a
  sliding-window confidence floor is breached) is the token-saving rung and wants
  a careful threshold so it doesn't prune correct-but-hesitant chains.
- **Min HW:** any (the demo runs on an 8 B model; our version targets a 10 M
  nanogpt checkpoint on CPU). **Ideal:** a real reasoning model + vLLM where the
  84.7 % token cut becomes a real $/query win.
- **Source:** [39]  ·  **Harvest:** ✅ shipped + measured → `nanogpt-edu`
  (`tools/bench_deepconf.py` — offline confidence-weighted **and** confidence-
  filtered vote + online early-abort; pure helpers unit-tested in
  `tests/test_deepconf.py`). **Measured** on a verifiable char-level addition toy
  (`configs/tiny_add3.py`, every answer checked vs ground truth):
  [`examples/deepconf_addition.md`](../nanogpt-edu/examples/deepconf_addition.md).
  **What held up at 10 M scale:** confidence robustly **tracks correctness**
  (correct traces +0.68–0.90 nats more confident than wrong), and **online
  early-abort trades tokens for accuracy on a clean curve** (~10 % fewer tokens
  at −0.8 % acc, gentle floor → 20 % at −6.7 %, aggressive). **What didn't:** the
  offline *vote* lift **ties** majority (±1 %) — expected and diagnostic, not a
  bug. DeepConf's headline lift needs **large k (256–512) on long reasoning
  traces**; a 6-token numeric answer at k=16 gives majority nothing to beat and
  the sliding window almost nothing to filter. The vote-lift is a **sizing fact**
  (real reasoning model), exactly like FP8/FA-3; the token-savings curve is the
  transferable, model-agnostic result. `frontier-platform` confidence-gated
  inference strategy remains 🔜. No training-time change — decode/eval-time only.

### § 14 — Verifier-free RLVR (RLPR) + GSPO going to production — **shipped + measured (frontier) / track**

Two RL-recipe deltas worth recording against our existing GRPO stack:

- **RLPR — RLVR *without* a verifier (Jun 2025).** RLVR's reach is gated by
  *"is there an executable checker?"* RLPR replaces the rule-based verifier with
  the policy's **own probability of the reference answer** (mean decoding
  probability) as a dense, domain-agnostic reward — extending reasoning RL to
  **general domains** (not just math/code), and reportedly **beating
  model-based-verifier RL** on both math and general-reasoning benchmarks. This
  is the concrete, buildable answer to the "process / explanation-scoring rewards"
  Tier-3 bet we've carried since May: a reward you can compute with *only the
  model*, no sandbox.
- **GSPO is now the default MoE-RL recipe, not a paper.** Qwen's **sequence-level**
  importance ratio + clipping (vs GRPO's noisier token-level ratio) is behind
  Qwen3's RL and is being adopted broadly (TRL exposes it via
  `importance_sampling_level="sequence"`); the practical note is that its clip
  ranges are ~2 orders of magnitude tighter than GRPO's because sequence-level
  ratios live on a different numeric scale. Promotes our May "track GSPO" item to
  "the recipe to implement for any MoE-RL in `frontier-platform`."
- **Win:** RLPR removes the verifier bottleneck (broadens RLVR to chat/general
  reasoning); GSPO removes the MoE-RL instability bottleneck (the exact failure
  mode our MoE recipe would hit under token-level GRPO).
- **Min HW:** same as our existing GRPO path (generation-heavy; 0.5–7 B with
  async gen). **Ideal:** multi-GPU async rollout for either at scale.
- **Source:** [40], [41]  ·  **Harvest:** ✅ shipped + measured → `frontier-platform`
  (`ProbabilityRewardVerifier` implementing RLPR behind the existing
  `make_verifier("probability", ...)` protocol in `rl/verifiers.py`; a
  `GRPOConfig.importance_sampling_level="sequence"` GSPO option with a
  length-normalized sequence ratio in `rl/grpo.py` — both CPU-tested in
  `tests/test_rl.py`, default-off so the token-level GRPO path is unchanged).
  **Measured on one 5060 Ti** (`tools/bench_grpo_gspo.py`, dense + MoE policies):
  [`examples/grpo_gspo_rlpr.md`](../frontier-platform/examples/grpo_gspo_rlpr.md).
  **GSPO's sequence ratio is ~4× lower-variance** than GRPO's token ratio
  (std 0.077 vs 0.296 dense; 0.085 vs 0.293 MoE) — the stability mechanism,
  measured — and **on the MoE model GSPO reaches higher reward** (+0.944 vs GRPO
  +0.875), its target regime. **RLPR sharpens the policy verifier-free**
  (answer-probability 0.44 → 0.70, emit-rate ~0.98) — with two real RLVR caveats
  it surfaced: RLPR needs an SFT warm-start (a cold policy has no advantage
  variance) and a KL anchor (`beta=0` lets the verifier-free reward be hacked
  into collapse). A device bug in the shipped verifier (CPU tensor on a GPU
  model) was fixed in passing. track → `coder-finetune` (GSPO via TRL once we
  exercise an MoE base). Both additive to the GRPO objective we already ship.

---

## Tier 1 — open-weight frontier wave (north-star + sizing evidence)

Not techniques to *build* so much as **evidence that recalibrates the roadmap**:
a remarkable run of capable **open-weight** models landed since the May catalogue,
several validating techniques we already track. They matter here as sizing facts
and north stars for `frontier-platform`, not as minimal-box builds.

| Model | Shape | What it validates for us | Source |
|---|---|---|---|
| **Kimi K2 Thinking** (Moonshot) | 1 T total / 32 B active MoE, 256 K ctx, **native INT4 via QAT** | Open frontier *reasoning agent*: SOTA on HLE (44.9 % w/ tools), τ²-Bench Telecom (93 %), **200–300 sequential tool calls**, ~$4.6 M train. **QAT-to-INT4 → ~2× decode, ½ the file size (594 GB)** — the serving-precision endgame, and a north star for the hill-climb recipe we harvested | [42] |
| **DeepSeek-V3.2(-Exp)** | MoE + **DSA** sparse attention (lightning indexer) | Confirms the NSA/DSA Tier-2 bet: ~½ long-context cost at matched quality; open training operator for the DSA warmup indexer now exists | [43] |
| **Qwen3-Next** | 80 B total / 3 B active, **hybrid 3:1 Gated-DeltaNet : full-attention** + ultra-sparse MoE + MTP | Promotes "hybrid linear-attention" from Tier-3 watch toward Tier-2: extreme activation sparsity (1/27) + linear-attention majority is now a shipping, vLLM-supported recipe | [44] |
| **NVIDIA Nemotron Nano 2** | 9 B **hybrid Mamba-Transformer**, reasoning | Same signal at the small/edge scale: up to **6× throughput** at on-par accuracy by replacing most attention with Mamba-2 — the legible end of the hybrid-SSM bet | [45] |
| **gpt-oss-120b / 20b** (OpenAI) | MoE, **native MXFP4** MoE weights, Apache-2.0 | 120 B runs on a single 80 GB GPU, 20 B in 16 GB — **MXFP4-from-release** is the open-weight echo of our FP8/NVFP4 precision policy | [46] |
| **GLM-4.6 / MiniMax-M2** | 200 K-ctx coder / 230 B-10 B-active agentic MoE | Open *coding/agentic* frontier at ~10 B active — the live north star for `coder-finetune`'s target capability | [47] |
| **Olmo 3** (AI2) | 7 B/32 B, **fully open "model flow"** (every stage, checkpoint, datum) | The reproducibility north star: a complete, traceable pretrain→think pipeline — the public analogue of what `frontier-platform` documents as a *system* | [48] |
| **nanochat** (Karpathy) | ~$100, single 8×H100 node, full ChatGPT stack | The educational north star one rung above `nanogpt-edu`: tokenizer→pretrain→mid-train→SFT→RL→web-UI in one hackable repo — a reference for where the `midgpt`→`coder-finetune` bridge could go | [49] |

**Harvest read:** none of these is a "port it this week" item — they're the
sizing/north-star column for the roadmap. The two that most change *our* plans:
**native low-bit-via-QAT serving** (Kimi K2's INT4, gpt-oss's MXFP4) makes the
quantization-aware-training rung concrete for the `frontier-platform` precision
policy and a future `midgpt` export, and **hybrid linear-attention + ultra-sparse
MoE** (Qwen3-Next, Nemotron Nano 2) is now shipped-in-the-wild enough to graduate
from Tier-3 watching toward a Tier-2 `distgpt`/`frontier` design note.

---

## Tier 2 — scale- or hardware-gated wins

> Recommended at the right scale — gates are sizing facts, not blockers. June
> deltas marked.

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| **FP8 / NVFP4 training** | 8-/4-bit matmul w/ per-tensor scaling | ~1.5–1.6× throughput, ~2× memory | Hopper/Blackwell; pays off >1B | [10] | 🟡 (frontier precision policy) | midgpt, distgpt, frontier |
| **FlashAttention-3** | Hopper-optimized attention kernel | ~1.5× over SDPA on H100 | Hopper+; long sequences | [11] | ideal | midgpt, distgpt |
| **FlexAttention** | `torch.compile` lowers custom masks/biases to fused kernels | FA-like speed with custom masks | PyTorch 2.5+; long ctx, packed docs | [24] | ✅ **shipped this month** — `attention_backend: sdpa\|flex` switch in **midgpt + nanogpt-edu** (CPU/grad guards, mask-rebuild cost documented) | nanogpt-edu, midgpt |
| **Multi-head Latent Attention (MLA)** | Low-rank KV compression | 5–10× smaller KV cache | long-ctx + any serving | [3] | ✅ frontier | distgpt, serving, frontier |
| **Native / DeepSeek Sparse Attention (NSA/DSA)** | Trainable sparse attention | near-linear long-ctx cost; ~½ long-ctx API cost (V3.2) | long-ctx (≥32K) train + serve | [35],[36],[43] | 🔜 track→planned — **DeepSeek-V3.2 shipped DSA + open warmup-indexer operator** | midgpt, distgpt, frontier |
| **Hybrid linear-attention + ultra-sparse MoE** | Gated-DeltaNet/Mamba majority + few full-attn layers; 1/27-class activation sparsity | big long-ctx throughput (Nemotron Nano 2 ~6×); cheap inference at large total params | ≥~10B total for the MoE win; PyTorch/vLLM support now exists | [44],[45] | 🔜 **promoted Tier 3→2** — track→design note | distgpt, frontier |
| **Native low-bit via QAT (INT4 / MXFP4)** | Quantization-aware *training/post-training* so weights serve losslessly at 4-bit | ~2× decode, ~½ file size (Kimi K2 INT4 594 GB); 120B on one 80 GB GPU (gpt-oss MXFP4) | Blackwell/Hopper for native kernels; pays off at serve scale | [42],[46] | 🔜 planned — complements FP8/NVFP4 train policy | midgpt, frontier |
| **Mixture-of-Experts** (fine-grained + shared) | Sparse FFN, more params @ ~const FLOPs | large quality/$ win | ≥3B active-equiv; expert parallelism | [3] | ✅ frontier; **MAI zero-init-attn** de-risks early MoE routing | distgpt, frontier |
| **3D/5D parallelism + comms overlap** | FSDP2 + TP + PP + EP + SP | near-linear scaling to 1000s of GPUs | multi-node + fast interconnect | [12] | 🟡 → **measured** — genuine 2-GPU FSDP2 calibrated on PCIe (§ below); the interconnect, not the code, is the ceiling | distgpt, frontier |
| **Unsloth fast-path** | Custom autograd kernels for LoRA/QLoRA | ~2× faster, ~70% less memory | single-GPU PEFT | [13] | ✅ coder-finetune | coder-finetune |
| **Single-node DDP (data-parallel)** | Replicate model, shard batch, all-reduce grads | linear-ish on a healthy fabric; the on-ramp to multi-GPU PEFT | ≥2 GPUs; model fits one card | [12] | ✅ **shipped this month** — `coder-finetune/cf_dist`, real-NCCL 2× 5060 Ti proof | coder-finetune |
| **Spectrum / targeted full-FT** | Full-FT high-SNR layers, freeze rest | beats LoRA at similar memory | mid-size SFT | — | 🔜 planned | coder-finetune |

### § The systems headline: flipping FSDP2 from negative to positive scaling — **measured (distgpt)**

The §"3D/5D parallelism" entry got its first honest in-repo number this month —
and it started *embarrassing*. A genuine 2-GPU FSDP2 run on a **no-NVLink** pair
(PCIe `PHB`, P2P unsupported, `NCCL_P2P_DISABLE=1` → every collective routes
through host memory) was, naively, **slower than one GPU**:

| Config (416 M, same seed) | Step time | Aggregate tok/s | Scaling | Per-GPU MFU |
|---|---:|---:|---:|---:|
| 1-GPU baseline | 2.84 s / 32 k tok | 11.5 k | 1.00× | 16.2 % |
| 2-GPU **naive** FSDP | 8.20 s / 65 k tok | 8.0 k | **0.69×** ❌ | 5.6 % |
| 2-GPU **optimized** | 4.44 s / 65 k tok | **14.8 k** | **1.28×** ✅ | 10.4 % |

Two changes did it, both halving collective volume: **(1)** `reshard_after_forward
=false` (keep params resident → no backward re-gather; costs ~0.8 GB/GPU,
optimizer state stays sharded), and **(2)** **gating gradient-sync to the last
micro-step** (`set_requires_gradient_sync`) so the reduce-scatter fires **once**
per optimizer step instead of 8×. Loss matched the naive run to four decimals —
the math is unchanged. The full 4 500-step run landed at **val ppl 41.6** (295 M
tokens), dead on the single-GPU forecast.

The honest read: **1.28× is 64 % efficiency, and per-GPU MFU (10.4 %) is *below*
the single GPU's 16.2 %** — that gap is the PCIe tax. You buy wall-clock at the
cost of per-device efficiency, and the entire lesson of `distgpt` is that this
gap is an **interconnect** property: the identical trainer on an NVLink node
reclaims most of those points. (A telling corollary: a bigger micro-batch — the
obvious comm-amortization lever — **OOMs**, because Fix 1's resident params eat
exactly the headroom it needs. On this fabric the comm-light setting wins.)
Full calibration: [`distgpt/examples/5060ti_416m_fineweb.md`](../distgpt/examples/5060ti_416m_fineweb.md#going-multi-gpu-genuine-2-gpu-fsdp2-over-pcie).

---

## Tier 3 — research bets (track, don't build yet)

Carried forward from May (still the right call to watch, not build):

- **Diffusion / next-byte / tokenizer-free LMs**, **long-context position
  schemes** (YaRN / NTK-by-parts / NoPE), **muP / HP transfer at frontier
  scale** — unchanged.
- **Hybrid Mamba/SSM-Transformer** — **graduating.** With Qwen3-Next (hybrid
  Gated-DeltaNet) and Nemotron Nano 2 (hybrid Mamba-2) now shipping as
  open-weight, vLLM-supported models, this moves from "research bet" to the new
  Tier-2 *hybrid linear-attention + ultra-sparse MoE* row above. Still track the
  pure-SSM / tokenizer-free variants here.
- **Process / explanation-scoring rewards beyond final-answer RLVR** —
  reinforced by MAI-Thinking-1's RL findings, and now with a concrete buildable
  instance: **RLPR** (verifier-free probability reward, §14) is the first version
  of this we'd actually implement. Promoted from "track" to a planned
  `frontier-platform` harvest.
- **🆕 Test-time scaling as a first-class lever (DeepConf, §13).** Beyond
  majority-vote self-consistency: confidence-gated parallel thinking gets *more
  accuracy for fewer tokens* with no training. Tracked here as the broader theme;
  the concrete DeepConf harvest is the Tier-1 §13 entry.
- **🆕 Staged long-context extension** (MAI Appendix B): pre-train short
  (16K) → mid-train (64K) → a short, cheap **256K extension (140B tokens)**.
  A progressive 32K→256K checkpoint *matches* a full 128K run on code NLL, and
  adaptation is **remarkably fast** (most gains in the first 1–10 % of steps),
  implying the model is recalibrating positional/attention behavior, not
  learning new capability. Extends to 1M+ tokens at modest cost. **The cheap,
  concrete recipe to adopt the moment we train past block_size 1024** — pairs
  with the FlexAttention work that shipped this month.
- **🆕 From-scratch reasoning ("hill-climbing")** as a *methodology* (vs the
  components we already harvested): specialists → distill → climb at real scale
  is an ideal-tier program for `frontier-platform`, not a minimal-box build.

---

## Roadmap by project

Carried forward from May; **bold = changed this month**. Everything is sized by
hardware tier — nothing is "blocked."

| Project | Next harvest | Tier | Notes |
|---------|--------------|------|-------|
| nanogpt-edu | full-module MTP; **long-ctx via FlexAttention**; staged long-ctx demo | minimal→ideal | **FlexAttention backend shipped** (`attention_backend: flex`); **DeepConf bench done** (§13 — `tools/bench_deepconf.py` + measured on a verifiable addition toy); next is exercising FlexAttention on a packed-doc / long-ctx config |
| midgpt | FP8 matmul; FA-3; **finish vLLM export path**; **QAT/INT4 export** | minimal→ideal | **HF-export validator shipped** (vLLM smoke); **llamafied A/B done (B wins 16.8 %)**; **zero-init-attn + FlexAttention landed**; FP8/FA-3 light up on 8× H100; native-low-bit-via-QAT export (Kimi K2/gpt-oss) is the serving-precision rung |
| distgpt | **Muon (distributed) — knobs landed**; FP8; MoE + expert parallelism; MLA; **hybrid linear-attn note** | ideal | **2-GPU FSDP2 calibrated (1.28× on PCIe)**; **distributed-Muon weight-decay/update-scale shipped**; validate the rest at multi-node + NVLink; hybrid Gated-DeltaNet/Mamba MoE is a new design-note target |
| coder-finetune | **Spectrum**; full-FT 7B; multi-node scale-out; r=256 to convergence on a 100k+ mixture | minimal→ideal | **DAPO knobs + SimPO/KTO shipped**; **single-node DDP shipped** (`cf_dist`, real-NCCL proof); **LoRA-Without-Regret done** (§12 — `lora_hicap.yaml` + pins + 3 measured A/Bs; budget-not-size is the binding constraint); next is targeted full-FT (Spectrum), training r=256 to convergence on a bigger mixture, and beyond one node |
| frontier-platform | sparse attention (1M ctx, NSA/DSA); real vLLM/SGLang backend; hardware MoE-RL run for GSPO at scale | ideal | **MAI-Thinking-1 harvested (10 components)**; **self-play loop shipped**; **GSPO + RLPR measured** (§14, `tools/bench_grpo_gspo.py` — GSPO ~4× lower-variance, wins on MoE; RLPR sharpens reward); **staged long-ctx note written**; remaining work is real backends, a data org, trainable sparse attention, and a frontier-scale MoE-RL run |

---

## What shipped this month

Config-gated, default-off where it touches existing behavior; full test coverage.

### The two empirical runs (the month's headline)

- **`midgpt` — llamafied vs GPT-2 350 M A/B** (`48a6a2b`, pickup doc `8c7de47`).
  Iso-param (354.60 M vs 353.51 M) / iso-token (131 M) / same 2-GPU DDP harness.
  **B wins 48.1 vs 57.8 val ppl (−16.8 %)**, leads at all 19 evals, reaches A's
  final quality ~37 % sooner. New 2-GPU iso-token configs (`gpt2_350m_fweb_5060ti_2gpu.yaml`,
  `..._llamafied_..._2gpu.yaml` with `micro_batch 2 / grad_accum 8`),
  `tools/run_llamafied_AB.sh` orchestrator, comparison plot, sampled completions.
  Writeup: [`examples/5060ti_350m_llamafied_AB.md`](../midgpt/examples/5060ti_350m_llamafied_AB.md).
- **`distgpt` — genuine 2-GPU FSDP2 over PCIe** (perf `1c0949b`, calibration
  `89397f4`, baseline propagation `29a8bab`, full-run doc `3b55b2d`, log dedup
  `cbf4275`). `reshard_after_forward` flag threaded through `parallel/fsdp.py`;
  gradient-sync gating in `training/trainer.py`. **Naive 0.69× → optimized
  1.28×**; full 4 500-step run **val ppl 41.6** (295 M tokens). `clean_log.py`
  keep-last dedup removed the phantom LR-dip artifact from the rewind loop.
  Writeup: [`examples/5060ti_416m_fineweb.md`](../distgpt/examples/5060ti_416m_fineweb.md).

### MAI-Thinking-1 harvest

- **`frontier-platform`** (`4a19593` Tier 1 RLVR, `bb300bf` Tier 2+3,
  `4d71eab` test-count fixes): ten components across `rl/verifiers.py`,
  `rl/grpo.py` (`EntropyController`, dual-clip), `rl/reward.py`,
  `rl/hillclimb.py`, `eval/reasoning_rubric.py`, `eval/long_context.py`,
  `safety/gates.py` (Pareto gate), `model/config.py` (`zero_init_attn_output`).
  **72 new tests; suite → ~409 passing.**
- **`midgpt`** (`5f08856`): cross-pollinated **zero-init residual projections**
  (`GPTConfig.zero_init_proj`, default-off) — parity with nanogpt-edu/distgpt.

### The four planned-harvest landings (shipped this pass, two measured on the 2× 5060 Ti)
  (r=256, all-linear, rsLoRA), `configs/ab_lora_r{16,256}{,_3ep}.yaml` iso-rank
  A/B configs, `examples/lora_without_regret_ab.md`, README recipe section, and
  `tests/test_lora_without_regret.py` (+15) pinning all four findings. **Measured:**
  r=256 **ties** r=16 at 3 epochs (token-acc 0.8476 vs 0.8469); r=16 wins at
  1 epoch because the big adapter is undertrained — compare at iso-convergence.
- **`nanogpt-edu` — DeepConf** (§13). `tools/bench_deepconf.py` (offline
  confidence-weighted + confidence-filtered vote + online early-abort),
  `prepare_addition.py` + `configs/tiny_add{,3}.py` (a verifiable task),
  `examples/deepconf_addition.md`, and `tests/test_deepconf.py` (+11). **Measured:**
  confidence tracks correctness (+0.68–0.90 nats); online early-abort cuts ~10 %
  tokens at −0.8 % acc; the offline vote-lift is a large-k/long-trace sizing fact.
- **`frontier-platform` — RLPR + GSPO** (§14). `ProbabilityRewardVerifier`
  (RLPR, verifier-free probability reward) behind `make_verifier("probability", …)`
  in `rl/verifiers.py`; `GRPOConfig.importance_sampling_level="sequence"` (GSPO,
  length-normalized sequence ratio) in `rl/grpo.py`. Default-off; CPU-tested in
  `tests/test_rl.py`. **Measured on GPU** (see follow-up runs below).
- **`docs/research` — staged long-context extension note** (Tier-3). The MAI
  Appendix-B recipe (16K→64K→256K, 140B-token tail) written up as
  `staged-long-context-extension.md`, cross-linked from the deep dive.

### Follow-up measurement runs (turning the planned harvests into numbers)

- **LoRA Without Regret — three A/Bs** (`coder-finetune`,
  [`examples/lora_without_regret_ab.md`](../coder-finetune/examples/lora_without_regret_ab.md)).
  8 k-Python at 1/3 epochs + 30 k×9-language at 2 epochs, one arm per GPU. r=256
  **ties** r=16 at convergence but **loses at every fixed epoch budget** (slow
  warmup of the 16× adapter); **binding constraint is training budget, not
  dataset size.** Adds `ab_lora_r{16,256}{,_3ep,_sep}.yaml` + a `dataset.shuffle`
  knob in `cf_data` (so a capped subset samples all 9 languages).
- **GSPO vs GRPO + RLPR — GPU measurement** (`frontier-platform`,
  `tools/bench_grpo_gspo.py`, [`examples/grpo_gspo_rlpr.md`](../frontier-platform/examples/grpo_gspo_rlpr.md)).
  GSPO sequence ratio **~4× lower-variance** than GRPO's token ratio (and **wins
  on the MoE policy**, +0.944 vs +0.875); RLPR verifier-free reward **sharpens
  the policy 0.44 → 0.70** with an SFT warm-start + KL anchor. Fixed a device bug
  in `ProbabilityRewardVerifier` (CPU tensor on a GPU model) + a regression pin
  (`tests/test_rl.py`, +1).

### Closing May's roadmap (`dd3bc03` + follow-ups)

- **`coder-finetune`:** **SimPO** (reference-free, routed through TRL
  `DPOTrainer`) + a pure-tensor **SimPO/KTO** loss reference + binary-feedback
  adapter (`cf_pref/{objectives,binary}.py`); **DAPO**-style **overlong reward
  shaping** (smooth ramp) + **dynamic-sampling mask** + `epsilon_high` clip-higher
  knob (`cf_rl/{reward,grpo_train}.py`). Plus **single-node DDP** (`cb86e55`,
  `cf_dist.py`) with launch scaffold (`370dc38`), CPU/gloo topology contract test
  (`d9a2ecd`), and a **real-NCCL 2-GPU proof** (`3b42c72`,
  [`examples/5060ti_2gpu_ddp.md`](../coder-finetune/examples/5060ti_2gpu_ddp.md)).
  Tests 74 → ~85.
- **`midgpt` / `nanogpt-edu`:** **FlexAttention** backend switch
  (`attention_backend: sdpa|flex`, CPU/grad guards, documented mask-rebuild
  cost). **`midgpt`** also gained `tools/validate_hf_export.py` (dependency-light
  HF-export validator + optional vLLM smoke + CI test).
- **`distgpt`:** opt-in **distributed-Muon** `weight_decay` + `update_scale`
  (defaults preserve the exact prior update, pinned by test);
  qk_norm/zero-init-proj surfaced in README (`5bdf2e1`).
- **`frontier-platform`:** deterministic **self-play / evolutionary loop**
  (`rl/selfplay.py`, AlphaEvolve/SPIN-shaped, no LLM-backend assumption).

### Housekeeping

- CI bumped to `checkout@v6` / `setup-python@v6` (Node 24) (`9b20a96`); distgpt
  distributed-test teardown hardened against gloo/FSDP2 SIGABRT (`db0ac47`,
  `48f8022`); DPO bf16/fp16 gated on actual GPU support (`03b4a60`); personal
  paths + GPU UUIDs scrubbed before public push (`5efbf24`); benchmark suite +
  examples runbook fully migrated **RTX 3050/P100 → RTX 5060 Ti** (`3697664`,
  `101981b`, `c7c0e00`).

Both core repos: ruff-clean, end-to-end train/resume/sample smoke-verified.

> **Hardware note.** All this month's runs are on **1–2× RTX 5060 Ti 16 GB**
> (Blackwell, sm_120, native bf16). The 3050/P100 era is fully retired from the
> benchmark tables and the examples runbook.

---

## Sources

New and changed for June; the May edition carries the full catalogue (sources
1–36), referenced by number above.

37. The Microsoft AI Team, *MAI-Thinking-1: Building a Hill-Climbing Machine* —
    35B-active/1T-total from-scratch MoE reasoning model; hill-climb RL recipe
    (specialists → distill → climb), adaptive entropy control, zero-init
    attention output, staged long-context extension, reasoning-trace taxonomy.
    `https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf`
    (deep dive: [`docs/research/mai-thinking-1-deep-dive.md`](./research/mai-thinking-1-deep-dive.md)).

**Researched this month — planned / tracked harvests (sources [38]–[49]):**

38. Thinking Machines Lab (Schulman et al.), *LoRA Without Regret* — LoRA matches
    full fine-tuning at ~⅔ the compute when applied to *all* linear layers with
    sufficient rank (≈256 SFT / 1–32 RL) and a higher, rank-independent LR;
    effective batch < 32. <https://thinkingmachines.ai/blog/lora/>
    (TRL reproduction: <https://huggingface.co/docs/trl/main/lora_without_regret>).
39. Fu, Wang, Tian, Zhao (Meta AI / UCSD), *Deep Think with Confidence (DeepConf)*
    — model-internal confidence filtering of reasoning traces (online early-abort
    + offline confidence-weighted vote); up to 99.9 % AIME 2025 with −84.7 %
    tokens, no training; arXiv 2508.15260. <https://jiaweizzhao.github.io/deepconf/>
40. Yu et al. (OpenBMB), *RLPR: Extrapolating RLVR to General Domains without
    Verifiers* — uses the policy's own mean decoding probability of the reference
    answer as a verifier-free reward; arXiv 2506.18254.
41. Zheng et al. (Qwen Team), *Group Sequence Policy Optimization (GSPO)* —
    sequence-level importance ratio + clipping; now the de-facto MoE-RL recipe
    behind Qwen3, exposed in TRL via `importance_sampling_level="sequence"`;
    arXiv 2507.18071. (Extends May src [29].)
42. Moonshot AI, *Kimi K2 Thinking* — 1 T-total / 32 B-active MoE reasoning agent,
    256 K ctx, **native INT4 via QAT** (~2× decode, 594 GB file), 200–300
    sequential tool calls; SOTA HLE-with-tools 44.9 %, τ²-Bench Telecom 93 %.
    <https://moonshotai.github.io/Kimi-K2/thinking.html>,
    <https://huggingface.co/moonshotai/Kimi-K2-Thinking>.
43. DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language
    Models* — DSA (lightning-indexer sparse attention) at production scale; open
    DSA warmup-indexer training operator; arXiv 2512.02556. (Extends May src [36].)
44. Qwen Team, *Qwen3-Next: Towards Ultimate Training & Inference Efficiency* —
    80 B-total / 3 B-active, hybrid 3:1 Gated-DeltaNet : full-attention, ultra-
    sparse MoE (10–11 of 512 experts) + MTP; vLLM-supported.
    <https://qwen.ai/blog> · <https://vllm.ai/blog/2025-09-11-qwen3-next>.
45. NVIDIA, *Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer
    Reasoning Model* — 9 B hybrid (most attention → Mamba-2), up to ~6×
    reasoning throughput at on-par accuracy; arXiv 2508.14444.
46. OpenAI, *Introducing gpt-oss* — gpt-oss-120b / 20b open-weight MoE, Apache-2.0,
    **native MXFP4 MoE weights** (120 B on one 80 GB GPU, 20 B in 16 GB).
    <https://openai.com/index/introducing-gpt-oss/> · <https://github.com/openai/gpt-oss>.
47. Z.ai, *GLM-4.6* (200 K-ctx agentic/coding) <https://z.ai/blog/glm-4.6>;
    MiniMax, *MiniMax-M2* (230 B-total / 10 B-active agentic-coding MoE)
    <https://github.com/MiniMax-AI/MiniMax-M2> — open-weight coding/agentic
    north stars for `coder-finetune`.
48. Team Olmo (AI2), *Olmo 3* — fully-open 7 B/32 B family releasing the entire
    "model flow" (every stage, checkpoint, datum); arXiv 2512.13961.
    <https://allenai.org/blog/olmo3>.
49. Karpathy, *nanochat* — full ChatGPT-style stack (tokenizer→pretrain→
    mid-train→SFT→RL→web UI) trainable for ~$100 on one 8×H100 node.
    <https://github.com/karpathy/nanochat>.

**Reproducible references (our own runs this month):**

R1. **midgpt llamafied A/B** — [`midgpt/examples/5060ti_350m_llamafied_AB.md`](../midgpt/examples/5060ti_350m_llamafied_AB.md)
    (configs, log.jsonl, comparison plot, `tools/run_llamafied_AB.sh`).
R2. **distgpt 2-GPU FSDP2 calibration + full run** — [`distgpt/examples/5060ti_416m_fineweb.md`](../distgpt/examples/5060ti_416m_fineweb.md)
    (1c0949b perf fix, 2-GPU config, both loss charts, DCP checkpoints).
R3. **coder-finetune real-NCCL 2-GPU DDP proof** — [`coder-finetune/examples/5060ti_2gpu_ddp.md`](../coder-finetune/examples/5060ti_2gpu_ddp.md)
    (`cf_dist`, `scripts/run_5060ti_2gpu_ddp.sh`, NCCL evidence log).

> **Methodology note.** This edition is run-led: the two A/Bs (R1, R2) and the
> DDP proof (R3) are primary evidence executed on-machine this month; numbers
> are extracted from the committed `log.jsonl`/console logs, not estimated.
> The MAI-Thinking-1 harvest (src 37) is sourced from the company technical
> report — **not peer-reviewed**, with cross-model competitor numbers pulled
> from official cards under their own harness, so treat cross-model claims with
> mild skepticism; the components we harvested are recipe-level and validated by
> our own CPU tests, independent of the paper's headline benchmarks. **The
> planned / tracked entries (sources [38]–[49]) are a literature/release scan,
> *not* on-machine runs** — every such entry is tagged `🔜 planned` / `track` /
> `ideal` and its numbers are the originators' reported figures (vendor blogs,
> arXiv preprints, and independent harnesses like Artificial Analysis), carried
> at the sources' confidence, not ours. They become `✅ shipped` only once
> harvested behind an in-repo test, like every prior entry. Hardware envelopes
> remain aspirational sizing targets, not a constraint from any one machine.
> Flag anything stale for the July edition.
