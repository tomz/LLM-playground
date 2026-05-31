# 17a — Frontier Model Gap Research (v2, code-grounded)

> **Version note.** This is a second pass of `17-frontier-model-gap-research.md`.
> v1 was written from the design docs; v2 was written after reading the actual
> implementation (`platform/**/*.py`), running the simulator at frontier scale
> (`1t` and `2t` presets), and enumerating the exact `NotImplementedError` and
> toy-vs-production boundaries in the code. Where v1 said "the repo has X",
> v2 names the file and says how *real* the X actually is.
>
> **Premise (unchanged from `docs/14`).** Assume a 10k–100k-GPU cluster, a $100M+
> compute budget, a data org, and a 100+ person team are available. The question
> is *"if we executed this blueprint exactly as written, would the resulting model
> sit next to the leading 2025-2026 frontier systems — and if not, why not?"*
>
> **Methodology note.** No reliable public technical documentation exists for
> "GPT-5.5", "Claude Opus 4.8", or "Gemini 3.5" by those exact version strings.
> Exact recipes for any closed frontier model are not public. Named-model gaps
> are inferred from the **publicly documented direction of the frontier** —
> DeepSeek-V3 (arXiv 2412.19437), DeepSeek-R1 (arXiv 2501.12948), DeepSeekMath /
> GRPO (arXiv 2402.03300), and the LLM-reasoning survey (arXiv 2504.09037, v4
> 2026-03) — not from leaked specs. Where the repo is genuinely strong (data
> hygiene, infra discipline, RSP gating, sim discipline), this document says so.

---

## TL;DR

If the cluster arrived tomorrow, `frontier-platform` could plausibly produce a
**credible 2025-class reasoning MoE base+SFT+RLVR model** — competitive with
strong open-weights releases (DeepSeek-V3/R1 class) — but **not** a true
frontier-flagship rival. The blocker is no longer architecture (the vocabulary
is right) and no longer pretraining compute (the simulator says ~$35-77M and
~9 days suffice for a 1T-2T MoE run on GB200/B300 — see §0). It is six things,
in order:

1. **A real distributed runtime.** The current MoE forward is a Python
   `for e in range(self.n_experts)` loop on one device
   (`platform/model/transformer.py:314`), and `ParallelEngine` raises
   `NotImplementedError` for any `tp>1` or `pp>1`
   (`platform/training/parallel.py:35-42`). No expert parallel, no
   tensor/pipeline parallel, no FSDP. **You cannot run the 1T MoE on the
   cluster the simulator prices.**
2. **A real post-training compute plant.** GRPO + sandboxed verifier +
   async actor-learner all exist as toy-functional skeletons
   (`platform/rl/grpo.py`, `sandbox.py`, `async_rollout.py`), but the
   simulator's RLVR phase costs $35k against a $35M pretrain — that is two
   orders of magnitude smaller than the o1/R1-class post-training spend the
   field has moved to, and the code lacks the throughput backbone (no vLLM
   wrapper — `serving/engine.py:48` raises `NotImplementedError`) to scale
   rollouts.
3. **A synthetic + reasoning-trace data factory.** The classical pipeline is
   strong (`platform/data/{dedup,decontaminate,filter}.py` are real, not
   stubs), but `platform/data/synthetic.py` is **literally a 16-line random
   word generator for tests** — the largest single capability gap in the repo.
4. **Native multimodality, not adapter multimodality.** The VLM
   (`platform/model/vision.py`) is a LLaVA-style late-fusion adapter; there
   is no audio, no video, no document/chart/OCR pipeline, and no multimodal
   data path.
5. **Real safety/eval harnesses for agentic and reasoning surfaces.** The
   red-team suite is **5 hardcoded prompts and a refusal regex**
   (`platform/safety/redteam.py:25-39`); the classifier is a **keyword
   counter** (`platform/safety/classifiers.py:13-24`); the eval harness falls
   back to perplexity on 5 small tasks if `lm-evaluation-harness` is missing
   (`platform/eval/harness.py:62-81`). The 2026 frontier suite numbers in the
   simulator come from **closed-form sigmoid predictors**, not real
   benchmark runs.
6. **Real FP8 / sparse-attention / 1M-context.** Precision policy is wired
   end-to-end and TE-ready (`platform/training/precision.py`), but the *real*
   FP8 numerics path needs Transformer Engine on Hopper/Blackwell; sparse
   attention for the 1M-context tier (DeepSeek-V3.2) is open.

Items 3 and 4 cannot be solved by compute alone. Items 1, 2, 5, 6 mostly can,
given the team.

---

## 0. Simulator-grounded cost numbers (run today, December 2025)

Before listing gaps it's worth fixing what the *blueprint itself says* a
frontier run costs, because that frames everything below. I just ran:

```bash
# 1T-total MoE (357B active), FP8, 32k GB200, RLVR post-training
python scripts/simulate.py --size 1t --moe-experts 256 --moe-top-k 8 \
    --precision fp8 --gpu-type GB200 --gpus 32768 --reasoning-rl \
    --mla-serving --spec-decode

# 2T-total MoE (713B active), NVFP4, 64k B300, larger RLVR
python scripts/simulate.py --size 2t --moe-experts 256 --moe-top-k 8 \
    --precision nvfp4 --gpu-type B300 --gpus 65536 --reasoning-rl \
    --rl-prompts 500000 --rl-steps 5000 --mla-serving --spec-decode
```

| Run | Active | Tokens | Cluster | Wall | Pretrain $ | RLVR $ | **Total $** | Simulated MMLU / HumanEval / GSM8K / ELO | Safety |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| `1t` MoE / fp8 | 357 B | 20 T | 32,768 × GB200 | 9.4 d | $35.3 M | $35 k | **$36.3 M** | 87.6 / 84.2 / 90.1 / 2268 | **BLOCK** (cbrn/cyber/persuasion/autonomy) |
| `2t` MoE / nvfp4 | 713 B | 30 T | 65,536 × B300 | 8.3 d | $76.1 M | $75 k | **$77.2 M** | 96.1 / 98.6 / 99.3 / 2641 | **BLOCK** (same) |

Two things to read out of this:

- **The pretraining bill for a 1T-active MoE on next-gen hardware is now
  modeled at ~$35M, not $500M+.** That is the active-param + FP8 + 50% MFU
  Blackwell economics talking. If the simulator is right within a factor of
  2-3, **money is no longer the gating constraint at the pretraining stage**.
- **The RLVR line is $35-75k against a $35-76M pretrain — ~0.01-0.02%.**
  That is a *toy* RLVR phase. The o1/R1-class post-training shift the field
  has made implies post-training compute that **rivals or exceeds**
  pretraining for reasoning models. The simulator's RLVR cost is calibrated
  to the toy GRPO loop in `platform/rl/`, not to a frontier production run.
  The gap doc (`docs/14` §1) is correct that this is the largest missing
  *subsystem*; the simulator numbers above are correct on the toy budget,
  not on the frontier budget.

Also: the simulator's safety predictor (a closed-form `cap * 0.55 + noise`
function, `platform/sim/eval_sim.py:91-97`) hits BLOCK on CBRN, cyber,
persuasion, and autonomy at the 1T/2T scale — which is the *intended*
behavior of the RSP gate, not a regression. But it tells you something real:
**a frontier-scale model coming out of this blueprint requires real
red-teaming and dangerous-capability evals before it can be deployed**, and
the current red-team harness (5 hardcoded prompts, §5 below) is nowhere near
sufficient for that gate.

---

## 1. What's actually real in the repo (file-by-file)

The point of this section is to be honest about the toy↔production boundary,
not to dunk on the skeleton. Skeleton is appropriate for a blueprint.

### 1.1 Architecture — production-correct, single-device only

| Subsystem | File | What's real | What's still toy |
|---|---|---|---|
| Decoder transformer (RoPE, RMSNorm, SwiGLU, GQA) | `platform/model/transformer.py` | Real PyTorch modules, runs on CPU/GPU. SDPA used for attention. | Single device. No FlashAttention call, no fused kernels. |
| Multi-head Latent Attention (MLA) | `platform/model/transformer.py:139-220` (`MLAttention`) | Real low-rank KV latent + decoupled RoPE, real incremental KV cache. KV-byte accounting via `ModelConfig.kv_bytes_per_token()`. | Same single-device caveat. |
| MoE FFN (fine-grained + shared + aux-free) | `platform/model/transformer.py:230-330` (`MoEFFN`) | Real top-k routing, real aux-free bias update, real z-loss, expert-count monitoring. | **Forward is a Python `for e in range(self.n_experts)` loop, line 314.** No all-to-all dispatch, no expert parallel, no capacity-factor token dropping. Correctness reference, not a runtime. |
| Multi-Token Prediction heads | `platform/model/transformer.py:355-364, 401-417` | Real train-time auxiliary heads + loss, gated on `self.training` so inference is unchanged. | No speculative-decode wiring in the serving engine (that uses the priced model only). |
| QK-norm | `GQAttention.__init__` (`qk_norm` flag) | Real per-head RMSNorm on Q/K before SDPA. | — |
| LLaVA-style VLM | `platform/model/vision.py` | Real ViT + projector + image-token prepend; loss masking on text positions; `VisionEncoder.from_pretrained` loads SigLIP via HF transformers when available. | Falls back to random-init in-house ViT when HF unavailable. No audio, video, doc/chart pipeline. |
| KV cache | `platform/model/kv_cache.py` + `Transformer.forward_with_cache` | Real incremental decode for GQA and MLA. Bitwise-close vs. full re-encode in tests. | — |

### 1.2 Training — single-process correctness

| Subsystem | File | What's real | What's still toy |
|---|---|---|---|
| AdamW + cosine LR + grad-clip | `platform/training/optim.py` | Real. | — |
| Muon optimizer | `platform/training/muon.py` | Real 5-step Newton-Schulz, real param partitioning (2D hidden vs. IO/embed via name markers). | Single-GPU path only (no distributed all-gather variant). |
| Precision policy | `platform/training/precision.py` | Real `PrecisionPolicy.create("fp8")` resolves to `transformer_engine` + `te.fp8_autocast` *when* TE is installed; otherwise bf16/fp32 with a one-time warning. The trainer's `forward_backward` wraps every forward in `self.precision.autocast()`. | The *fallback* path is exercised today; the *real* FP8 numerics path is exercised only on Hopper/Blackwell with TE installed. So FP8 is "interface-ready", not "validated". |
| Parallel engine | `platform/training/parallel.py` | Real DDP wrap if `dp>1`. | **`tp>1` or `pp>1` raises `NotImplementedError` (line 39-42).** No FSDP, no ZeRO. The doc (`docs/05`) describes Megatron-Core / NeMo / DeepSpeed; the code wraps DDP. |
| Trainer.fit | `platform/training/trainer.py` | Real loop with spike monitor, rewind controller, checkpoint manager, eval hooks. | Synthetic / single-rank. |
| Checkpointing | `platform/training/checkpoint.py` | Real, async save. | Single-rank shards. |
| Spike monitor + rewind | `platform/training/stability.py` | Real rolling-stat detector, rewind via checkpoint mgr. | — |

### 1.3 Post-training — algorithmically right, productionally thin

| Subsystem | File | What's real | What's still toy |
|---|---|---|---|
| SFT (assistant-token loss mask) | `platform/alignment/sft.py` | Real loss-masked SFT. | — |
| Reward model (Bradley-Terry) | `platform/alignment/reward_model.py` | Real BT loss + scalar head. | — |
| DPO/IPO/KTO | `platform/alignment/dpo.py` | Real DPO sigmoid + variants. | — |
| PPO (GAE, KL-to-ref, value head) | `platform/alignment/ppo.py` | Real PPO update. | Synchronous, in-process, single-GPU. |
| GRPO learner | `platform/rl/grpo.py` | **Real per-token clipped surrogate, real k3 KL estimator, real DAPO-style decoupled `clip_higher`, real group-relative advantage, real correct treatment of behavior-vs-current logp ratio under async skew.** This is the cleanest part of the post-training stack. | Single-process, ~10 steps in tests. |
| Verifiers (math/regex/code) | `platform/rl/verifiers.py` | Real sympy-based symbolic equivalence; `MathExactVerifier` tries boxed → numeric → symbolic → string; `CodeUnitTestVerifier` runs candidates against held-out tests. | Verifier *coverage* is math + regex + Python code. No formal proof checkers, no SQL/JSON validators in code, no browser/CAS/theorem verifiers. |
| Sandbox | `platform/rl/sandbox.py` | **Real subprocess + POSIX `RLIMIT_CPU` / `RLIMIT_AS` / `RLIMIT_FSIZE` + wall-clock timeout + scrubbed env + `os.setsid()`.** Honestly documents itself as "not a complete jail". | No gVisor, no Firecracker, no nsjail, no network namespace. Read-only host FS is reachable. |
| Async rollout | `platform/rl/async_rollout.py` | Real `asyncio` actor-learner over the in-process `TorchEngine`, with weight-version tracking + `RolloutBuffer` queue. The interface is the same one a vLLM/SGLang actor would implement. | The backing Engine is the in-process Torch one (the `vllm` branch in `serving/engine.py:43-48` raises `NotImplementedError`). So this is "right shape, wrong scale". |
| Reasoning SFT cold-start | `platform/rl/coldstart.py` | Real R1-style cold-start loop. | Toy scale. |
| Reward shaping | `platform/rl/reward.py` | **Real `CompositeReward` with format reward + soft length penalty + repetition penalty + `answer_spam_guard` (anti-shotgun) + reward clipping + per-component `.breakdown()` for monitoring.** Better than most public OSS RLHF stacks. | — |
| Agentic env | `platform/rl/agentic.py` | Real `ToolEnv` + `<\|tool_call\|>` JSON parsing + balanced-brace JSON extraction + terminal sparse reward + `Trajectory` accounting. | Built-in tools are `calculator` and `lookup`. No browser, no shell, no real code-edit env, no SWE-bench harness. |

### 1.4 Data — pipeline is genuinely real; sources and synthetic are not

| Subsystem | File | What's real | What's still toy |
|---|---|---|---|
| Acquire | `platform/data/acquire.py` | Real `LocalFilesSource`, real `JsonlSource`. | **`CommonCrawlSource`, `GitHubSource`, `ArxivSource`, `WikipediaSource` all raise `NotImplementedError`.** Web/code/papers/wiki ingestion is a stub. |
| Extract | `platform/data/extract.py` | (Skeleton; the doc calls out `trafilatura`/`resiliparse`/`nougat`/AST extractors.) | — |
| Filter (Gopher + classifier + tox) | `platform/data/filter.py` | **Real Gopher heuristics, real FineWeb-Edu-style composite score, real keyword-based toxicity scoring, real fasttext lazy-load hook.** | Quality classifier is heuristic, not a trained fastText model. Detoxify hook is detection-only (no model loaded). |
| Dedup (exact / MinHash-LSH / suffix-array) | `platform/data/dedup.py` | **Real exact SHA-1 dedup, real MinHash-LSH (`MinHashDeduper`), real toy substring dedup.** Production-shaped API, single-process scale. | `substring_dedup` is O(N²) in chars (the docstring says use the Rust suffix-array impl for real scale). MinHash is single-process. |
| Decontamination | `platform/data/decontaminate.py` | **Real 13-gram blake2b-hashed contamination detector with per-eval-set attribution and a reportable hit count.** Production-shaped, scales to ~10s of millions of n-grams in a `set[int]`. | Single-process set; production would use a Bloom filter / sharded index. |
| **Synthetic** | `platform/data/synthetic.py` | **Nothing real.** **The entire file is a 16-line deterministic word-bag generator for tests.** | This is the biggest data gap in the repo. |
| Loader / shard / mix | `platform/data/{loader,shard,mix}.py` | Real streaming, real sharded layout, real mix weighting. | — |

### 1.5 Eval — falls back to perplexity

| Subsystem | File | What's real | What's still toy |
|---|---|---|---|
| Eval harness | `platform/eval/harness.py` | Lazy-loads `lm-evaluation-harness` if installed; otherwise falls back to numpy cross-entropy / perplexity. Real contamination report via §1.4 decontamination. | The "fast" path is 5 zero-shot tasks (`FAST_TASKS`); the "full" path needs `lm_eval` installed (not a default dep). |
| Arena ELO | `platform/eval/arena.py` | Real Bradley-Terry ELO from pairwise judgments. | — |
| Contamination | `platform/eval/contamination.py` | Real, see §1.4. | — |

### 1.6 Safety — RSP shape is right, content is heuristics

| Subsystem | File | What's real | What's still toy |
|---|---|---|---|
| Pre-deployment gate | `platform/safety/gates.py` | **Real `preflight(card, thresholds)` that reads a JSON red-team report and returns PASS/BLOCK with per-category failed list. This is the right RSP gate shape.** | Reads a report; doesn't run the eval itself. |
| Classifiers (input/output sandwich) | `platform/safety/classifiers.py` | Shape is right (input + output `_score` with max). | **Implementation is `tokens-in-bad-word-list / token-count * 10`.** No Llama-Guard, no trained model. |
| Red-team harness | `platform/safety/redteam.py` | Shape is right (suite → probes → refusal-detect → rate). | **5 hardcoded prompts across 5 named suites, refusal detected by a regex of "I can't / sorry but / unable to".** No HarmBench, no AdvBench, no AnyMisuse, no METR runtime, no Cybench, no multi-turn. |

### 1.7 Serving — single-process TorchEngine + KV cache + simulator

| Subsystem | File | What's real | What's still toy |
|---|---|---|---|
| Engine API | `platform/serving/engine.py` | Real `Engine` dispatch with `torch` backend wired. | **vLLM branch raises `NotImplementedError` (line 48). TRT-LLM, SGLang likewise.** |
| TorchEngine | `platform/serving/torch_engine.py` | **Real prefill + incremental decode using `Transformer.forward_with_cache`. Real temperature / top-p / logprob return (matches what vLLM/SGLang return). The actor's importance ratios in GRPO are correct against these logprobs.** | Single process, no continuous batching, no paged KV, no prefix cache across requests. |
| Router | `platform/serving/router.py` | Real tier selector with TTFT budget + cost-min tiebreaker. | — |

### 1.8 Simulator — by far the most mature subsystem

| Subsystem | File | What's real |
|---|---|---|
| Orchestrator | `platform/sim/orchestrator.py` | Real end-to-end glue: data → tokenizer → pretrain → align → reasoning_rl → agentic_rl → eval → safety → serving, with measured-TFLOPs calibration hook. |
| Pretrain pricing | `platform/sim/pretrain_sim.py` | Real chinchilla-loss steps + MFU + active-param FLOPs + precision speedup + node-failure sampling + downtime accounting. |
| Reasoning-RL pricing | `platform/sim/reasoning_rl_sim.py` | Real rollout-FLOPs + update-FLOPs + verifier-CPU + cold-start label $. The *reasoning_quality* lift feeds the eval predictors (this is the unique part). |
| Agentic-RL pricing | `platform/sim/agentic_rl_sim.py` | Real turn × tokens × tool-CPU pricing + saturating-lift model that depends on base + reasoning capability. |
| Serving pricing | `platform/sim/serving_sim.py` | Real MLA KV-throughput multiplier + truncated-geometric speculative-decode multiplier (matches the nanogpt-edu MTP benchmark). |
| Eval predictors (2024 + 2026) | `platform/sim/scaling.py` | Closed-form sigmoid predictors for MMLU/HumanEval/GSM8K/SWE-bench/ARC-AGI-2/HLE/MMMU, with post-training and modality lifts. Calibrated to public 2025 reports. |

**Bottom line of §1:** the **shapes** are right almost everywhere, the
**MoE/MLA/MTP/Muon/QK-norm/precision/GRPO/sandbox/rollout/agentic** code is
real-but-toy, and a handful of specific things — distributed runtime, vLLM
backend, synthetic data, real red-team and safety classifiers, real eval
harness, gVisor jail — are the load-bearing missing pieces.

---

## 2. Gap-by-gap against the public direction of the frontier

Numbering follows the in-repo `docs/14` so cross-references stay easy. Severity:
🟥 capability-defining · 🟧 significant · 🟨 catch-up. Cost type: **$** = compute/eng ·
**R** = research bets · **D** = data/labeling org · **O** = organizational.

### 2.1 Real distributed runtime ($/O) — 🟥 the hidden show-stopper

The blueprint's design (`docs/05`) describes the right composition (DP × FSDP ×
TP × PP × SP × EP × CP), but `platform/training/parallel.py:39-42` explicitly:

```python
if cfg.tp > 1 or cfg.pp > 1:
    raise NotImplementedError(
        "tensor/pipeline parallel not implemented in torch_native engine"
    )
```

And the MoE forward (`platform/model/transformer.py:314`) is:

```python
for e in range(self.n_experts):
    mask = (top_i == e)
    ...
```

This is fine for tests with 8-128 experts on one device. It will not run the
1T MoE the simulator prices on a 32k-GPU cluster. The closest thing in the
repo is `platform/sim/real_train.py` (a calibration probe), which is
explicitly a *single-device* TFLOP/s measurement.

**What this looks like as work.** Pick a backend — Megatron-Core, NeMo,
DeepSpeed, or a stitched PyTorch-native (FSDP2 + DTensor TP + Pipe). Add
expert-parallel all-to-all (DeepSpeed-MoE / Megatron-MoE / Tutel /
NVIDIA-Megatron-LM EP), capacity-factor token dropping, expert-imbalance
recovery, reshardable expert checkpoints, communication overlap, straggler
mitigation. This is **6-12 engineer-months for a competent infra team** and
is mostly engineering, not research — but money does not skip it.

### 2.2 Production-scale RLVR plant (R/$) — 🟥 algorithmically right, scale-wrong

The algorithm is right (see §1.3 — `grpo.py` has correct importance ratios
against the *behavior* policy, k3 KL, decoupled clip ranges). The
infrastructure is shaped right (async rollout engine, weight sync,
verifier-CPU separated from learner GPU). What's missing:

- **Inference backend that scales.** `serving/engine.py` raises
  `NotImplementedError` on the vLLM branch. The async rollout therefore
  drives the in-process `TorchEngine` — fine for tests, useless for the
  ~10⁵-10⁷ rollouts a frontier reasoning RL phase needs.
- **Sandbox jail.** `rl/sandbox.py` uses POSIX rlimits + a scrubbed env, but
  doesn't namespace the filesystem or block syscalls (the file says so
  itself). Wrap in gVisor / Firecracker / nsjail + a network namespace
  before pointing it at untrusted model code at scale.
- **Verifier coverage.** Only math (sympy) and Python unit-tests are wired.
  Formal proof checkers (Lean / Isabelle), SQL/JSON schema validators,
  symbolic-CAS verifiers, and browser/computer-use task verifiers are not.
- **Reasoning-trace data.** Cold-start exists; the *factory* that produces
  millions of high-quality long-CoT traces with verified correctness does
  not (this is §2.3 below).
- **Compute scale.** The simulator's RLVR phase is ~0.01% of pretrain. The
  o1/R1-class direction is for post-training to **rival** pretraining
  compute. Re-pricing the simulator at that scale would be a useful
  honesty exercise.

The good news: when each of these lands as an interface-compatible
swap (vLLM in `EngineConfig.backend`, gVisor wrap around `sandbox.py`, new
verifier classes), the GRPO learner code does not change.

### 2.3 Synthetic + reasoning-trace + agentic-trajectory data factory (D/R) — 🟥 biggest single gap

`platform/data/synthetic.py` is **16 lines of word-bag generator for tests.**
This is the largest single gap in the repo. The classical data pipeline
(`acquire`, `extract`, `filter`, `dedup`, `decontaminate`, `mix`, `shard`,
`loader`) is real and good (§1.4). The thing that has actually changed about
frontier data programs in 2025-2026 is **the dominance of model-generated
data** — distillation, rephrasing, textbook-generation, self-improvement,
reasoning traces, agentic trajectories.

**Minimum factory shape** (no equivalent in repo today):

- `platform/data/synthetic/`:
  - teacher orchestration (calls one or many models),
  - generation policies (rephrasing, textbook, story-grounding, reasoning),
  - rejection sampling against verifiers (re-uses `rl/verifiers.py`),
  - diversity / coverage tracking (n-gram, embedding, topic),
  - contamination index integration (re-uses `eval/contamination.py`),
  - data lineage + license tracking,
  - per-source budget and cost accounting.
- Reasoning-trace pipeline:
  - prompt curation across math/code/STEM/logic,
  - long-CoT sampling with verifier filtering,
  - difficulty curriculum (easy → hard, gated by base-model success rate),
  - decontamination vs. math/code/reasoning benchmarks.
- Agentic-trajectory pipeline:
  - browser / code / shell / tool sessions,
  - success and informative-failure capture,
  - environment-state logging,
  - sparse terminal reward labels,
  - trajectory dedup and curation.
- Multimodal pipeline (none today — see §2.4).

**Why GPUs alone cannot fix this.** Data distribution, verifier coverage,
contamination control, task curricula, and licensing are research + ops
problems. They require people, not just GPU-hours.

### 2.4 Native multimodality vs. adapter multimodality ($/R/D) — 🟥

`platform/model/vision.py` is honestly described in `docs/16` as **MM-1**: a
LLaVA-style adapter (ViT/SigLIP → projector → image-tokens-prepended). That
is a defensible *baseline*; it is not what a Gemini- or GPT-5-class flagship
does. The current implementation is also missing:

- The tokenizer contract for image patches / VQ codes (the bytes tokenizer
  is text-only; the special token vocabulary doesn't reserve image
  placeholders beyond `<|image|>`/`<|image_end|>` as conventions).
- An interleaved image-text data pipeline (no `platform/data/multimodal.py`).
- Audio / video / document / chart / OCR encoders and pipelines.
- A multimodal eval suite that runs on real datasets (MMMU/MathVista/
  ChartQA/DocVQA/RealWorldQA/video QA). The simulator predicts MMMU above
  chance when `multimodal=True`, but that is a sigmoid, not a benchmark run.
- Multimodal serving (variable-resolution image tiling, image preprocessing
  in the engine, multimodal KV economics).

This is a whole second platform's worth of work. Adapter-MM-1 is appropriate
for a research demo; native multimodality is what the flagships ship.

### 2.5 Agentic capability is a 2-tool toy env (R/D) — 🟧

`platform/rl/agentic.py` is well-shaped: terminal sparse reward,
tool-call/tool-result JSON, balanced-brace JSON extraction, agent↔env
multi-turn loop, `Trajectory` accounting with `malformed` count. The two
built-in tools are `calculator` and `lookup`. There is no real coding
sandbox env, no browser env, no shell, no file-editing env, no SWE-bench
harness, no long-horizon planning env. The agentic-RL simulator
(`platform/sim/agentic_rl_sim.py`) prices the cost shape correctly
(turns × tokens × tool-CPU) and computes an `agentic_quality` lift, but
again — that lift drives a *closed-form predictor*, not a real SWE-bench
run.

Closing this requires: real sandboxed coding env (re-use `rl/sandbox.py`),
real browser env (Playwright / browser-use), real shell env (firejail or
similar), real trajectory mining (expert traces from coding sessions),
SWE-bench / Terminal-Bench / GAIA harnesses.

### 2.6 MoE as default vs. MoE as option ($) — 🟧, mostly addressed

`docs/03` now says MoE is the frontier default; `MoEFFN` implements
fine-grained + shared + aux-loss-free. The reference shapes still list the
1B/7B/70B/400B dense tiers as the primary table with **MoE-1T** as the
frontier-shaped tier. The simulator presets include `1t` and `2t` MoE.
**What's missing is operational**: real expert-parallel dispatch (§2.1),
plus making the MoE config the default for any large-tier run.

### 2.7 MLA + sparse attention + 1M context ($/R) — 🟧, half-addressed

`MLAttention` is real with a real incremental KV cache (§1.1). What's still
open:

- **Sparse attention for 1M context** (DeepSeek-V3.2 direction). Not in the
  model or the serving simulator.
- **Long-context evaluation** beyond perplexity: RULER, needle-in-haystack
  variants, long-doc QA, long-context coding, score-vs-context-length curves.
- **Long-context curriculum**: midtraining with long docs, position-
  interpolation stability under extension, retrieval-augmented memory.

### 2.8 FP8/NVFP4 numerics ($) — 🟧

`platform/training/precision.py` is the model citizen here: a single
`PrecisionPolicy` resolves `fp8` to `transformer_engine`'s `fp8_autocast`
under `te.fp8_autocast` when TE is installed, falls back to bf16 / fp32
otherwise, and the trainer wraps every forward in `self.precision.autocast()`
(`platform/training/parallel.py:60-66`). The fallback is exercised; the FP8
path needs Transformer Engine on Hopper/Blackwell — at which point you also
need the per-tile/per-block scaling, accumulation policy, loss-scaling,
checkpoint conversion, communication-precision policy, and the regression
tests that DeepSeek-V3 wrote a whole report about. The simulator already
prices FP8 / NVFP4 throughput (`precision_speedup()` in
`platform/sim/scaling.py`); the *numerics* are interface-ready, not
validated.

### 2.9 Optimizer & training tricks ($/R) — ✅ ported

`platform/training/muon.py` is a real port of the modded-nanogpt Muon
optimizer with the right param-partitioning rules (`split_muon_params`
excludes embeddings, lm_head, MTP heads, and MoE gates). MTP is in the
model and gated to train-only. QK-norm is in `GQAttention`. The
`docs/14` Update column on §6 is honest about this being done.

### 2.10 Real eval harness vs. simulated predictors (D/O) — 🟧

`platform/eval/harness.py` will hand the suite to `lm-evaluation-harness`
**if installed**; if not, it falls back to 5 zero-shot tasks
(HellaSwag/ARC-Easy/PIQA/BoolQ/OpenBookQA) for the fast path and
perplexity for the full path. The 2026 frontier suite (SWE-bench Verified,
ARC-AGI-2, HLE, MMMU) is **only in the simulator**, where each is a
closed-form sigmoid predictor with `reasoning_quality` / `agentic_quality`
/ `multimodal` modifiers (see `platform/sim/scaling.py:124-180`). A real
frontier program needs these wired into the actual harness as gating
checkpoint evals.

### 2.11 Safety harness vs. RSP discipline (O/D) — 🟧

The *gate* (`platform/safety/gates.py`) is the right shape: a `preflight`
function reads a model card + red-team report, compares per-category scores
to per-category thresholds, and returns PASS/BLOCK. This is the RSP
discipline that distinguishes a real frontier program from a research
project. But the *content* is:

- 5 hardcoded prompts in 5 named suites (`platform/safety/redteam.py:25-39`).
- Refusal detection by regex.
- Input/output classifier as a `bad-word-token-count / total-tokens * 10`
  heuristic (`platform/safety/classifiers.py:13-24`).

For a frontier deployment that gate needs HarmBench / AdvBench / Cybench /
WMDP / METR-agent / multimodal-jailbreak / chain-of-thought-monitor /
scheming-eval / agentic-autonomy / cyber-range probes, plus real
Llama-Guard-class classifiers, plus a trusted external red team.

### 2.12 Serving as a product platform ($) — 🟧

`TorchEngine` does real prefill + KV-cache decode + temperature/top-p +
logprobs. The vLLM/TRT-LLM/SGLang branches are stubs. Missing pieces for a
frontier product platform: paged KV cache implementation, continuous
batching, prefix caching across requests, real speculative-decode loop
(the simulator prices it, but the engine doesn't run a draft-and-verify
loop), tool-call runtime, multimodal input handling, hidden/visible
thinking budget, safety filters in the path, prompt-injection defenses,
per-user personalization, distillation cascade, and the full observability
suite (token latency, cache hit rate, tool failures, eval drift, safety
incidents, cost attribution).

---

## 3. What's NOT a gap (give credit where due)

It's easy to make this read as a long list of failings. Several parts of
the blueprint are genuinely strong and would not be the bottleneck.

- **Architectural vocabulary.** MoE (fine-grained + shared + aux-free), MLA
  with real incremental KV, MTP, Muon, QK-norm, RoPE-extension hooks — all
  named and implemented at toy-functional scale. A new team would not have
  to argue about whether to use these.
- **Algorithmic correctness of GRPO.** `platform/rl/grpo.py` is cleaner than
  many public OSS RLHF stacks: per-token importance ratio against the
  *behavior* policy (correct under async actor-learner skew), k3 KL
  estimator (always ≥0), decoupled clip ranges (DAPO-style), broadcast
  group-relative advantage, monitored clip-frac and KL each step. This is
  the part of the post-training plant that wouldn't need to be rewritten.
- **Reward shaping discipline.** `CompositeReward` with `answer_spam_guard`,
  repetition penalty, soft length penalty, format reward, clipped total,
  per-component breakdown. The anti-reward-hacking instincts are present.
- **Classical data pipeline.** Real MinHash-LSH, real 13-gram blake2b
  contamination index with per-eval-set attribution, real Gopher heuristics,
  real composite quality classifier shape. The data pipeline is the part of
  the repo most ready to scale.
- **Simulator discipline.** A discrete-event simulator that prices MoE
  active params, FP8/NVFP4 throughput, MLA KV-throughput, speculative
  decoding, RLVR rollout + verifier CPU + label $, agentic-RL turns × tools,
  GPU failures, downtime accounting, RSP-style safety gating. The simulator
  is more honest than the docs about the post-training cost shift.
- **RSP gate shape.** A `preflight(model_card)` that blocks promotion on a
  per-category threshold report. The right discipline; the content needs
  filling in.
- **Observability + reliability math.** `docs/11` does the MTBF math
  correctly (1 GPU failure / 21h on 4096 GPUs at 10y MTBF); the simulator
  enforces it (`platform/sim/cluster.py:55-65`).
- **Numerical/precision separation.** `PrecisionPolicy` is a clean
  abstraction that makes the FP8 swap a one-liner when the hardware arrives.

---

## 4. Closing-the-gap plan, in dependency order

Assume team and GPUs are available. Sequence matters because some items
unlock others.

### Phase 0 — Productionize the runtime substrate (months 1-6)

1. Pick a distributed backend (Megatron-Core is the lowest-risk choice for
   MoE at scale; NeMo / DeepSpeed / Megatron-LM are alternatives). Replace
   `platform/training/parallel.py` with a real wrapper.
2. Real expert-parallel MoE dispatch (Megatron-MoE / DeepSpeed-MoE / Tutel
   / NVIDIA-NeMo MoE). Reshardable expert checkpoints.
3. Real FP8 numerics validation on Hopper/Blackwell with Transformer
   Engine: per-tile scaling, accumulation policy, loss-scaling, regression
   tests, checkpoint conversion.
4. vLLM / SGLang backend wired into `Engine` (replace the
   `NotImplementedError` on `serving/engine.py:48`). This unlocks both
   serving and scaled RLVR rollouts.
5. gVisor / Firecracker / nsjail wrap around `rl/sandbox.py`. Network
   namespace with no routes.

**No model training yet.** This is enabling work.

### Phase 1 — Real data factory (months 1-12, parallel)

1. Real source connectors: replace `acquire.py` `NotImplementedError`s
   with `warcio` over S3 (CommonCrawl), GHArchive (GitHub), arxiv OAI-PMH,
   wikiextractor. Multilingual via mC4/CulturaX.
2. Replace `platform/data/synthetic.py` (the 16-line word generator) with
   a real synthetic-data factory: teacher orchestration, rejection
   sampling, verifier filtering, contamination integration, diversity
   tracking, lineage, licensing.
3. Reasoning-trace pipeline: curated math/code/STEM/logic prompts, long-
   CoT sampling with verifier filtering, difficulty curriculum, RL
   prompt-pool decontamination.
4. Agentic-trajectory capture: instrumented coding/browser/shell sessions,
   successful + informative-failure trajectories.
5. Multimodal data: interleaved image-text, document/chart/OCR, captions,
   perceptual-hashing dedup. Optional video/audio.
6. Real quality classifier: trained fastText / small transformer on
   FineWeb-Edu-style labels. Real Detoxify or Llama-Guard for toxicity.

### Phase 2 — Scaling-law sweep + ablations (months 4-8)

1. 50M / 150M / 500M / 1.5B / 5B sweep on identical data to fit L(N, D).
2. Ablations: MoE routing (top-k, expert count, fine-grained vs. coarse,
   aux-free vs. aux-loss), MLA vs. GQA, MTP weights, Muon vs. AdamW on
   hidden weights, FP8 vs. bf16, data mixtures, long-context curricula.
3. Pick a "house" config from the sweep.

### Phase 3 — Real pretraining (months 6-12)

1. Start with a 30B-70B-active MoE run, not the largest possible. This
   validates the runtime and the data factory on something that fails
   recoverably.
2. Midtraining stages: code-heavy, math-heavy, long-context (RoPE-extend
   + curriculum), high-quality annealing.

### Phase 4 — Real post-training (months 10-18)

1. Reasoning SFT cold-start at scale (R1-style).
2. RLVR/GRPO at frontier scale: many actors, learner, verifier fleet, KL
   monitor, length monitor, reward-component logging. The `platform/rl/`
   code is the right starting point; the rollout backend is vLLM/SGLang
   (Phase 0).
3. Agentic RL: real sandboxed coding env, real browser env, real shell
   env, expert-trajectory SFT, GRPO over multi-turn episodes.
4. Preference alignment: SFT + DPO/RLHF for tone, refusal, persona,
   safety.
5. Distillation: smaller production models from the large reasoning
   teacher.

### Phase 5 — Real eval + safety + product (months 12-24)

1. Wire real `lm-evaluation-harness` + SWE-bench Verified +
   LiveCodeBench + Terminal-Bench + GAIA + GPQA Diamond + HLE +
   ARC-AGI-2 + FrontierMath + RULER + MMMU/MathVista/ChartQA/DocVQA into
   the actual `Evaluator`, not just the simulator.
2. Replace `safety/redteam.py` (5 prompts) with HarmBench / AdvBench /
   Cybench / WMDP / METR-agent / chain-of-thought monitor / scheming
   eval / multimodal jailbreak suites. Trusted external red team.
3. Replace `safety/classifiers.py` (keyword heuristic) with Llama-Guard-
   class trained classifiers per category.
4. Production serving: continuous batching, paged KV, prefix cache,
   spec-decode loop, multimodal input, tool runtime, reasoning-budget
   API, safety filters, prompt-injection defenses.

---

## 5. Ranked gap table

| # | Gap | Severity | Cost type | Where it lives in code | What closes it |
|---:|---|---|---|---|---|
| 1 | Distributed runtime (TP/PP/EP, expert-parallel MoE) | 🟥 | $/O | `parallel.py:39-42` raises; `MoEFFN.forward` is a Python for-loop | Megatron-Core/NeMo/DeepSpeed wrap + real expert parallel |
| 2 | Synthetic + reasoning-trace + agentic-trajectory data factory | 🟥 | D/R | `synthetic.py` is 16 lines of test data | A real data org + pipeline + verifier-filtered generation |
| 3 | Production RLVR plant (vLLM rollout + jail + verifier coverage + post-training compute budget) | 🟥 | R/$ | `serving/engine.py:48`, `rl/sandbox.py`, `rl/verifiers.py` | vLLM/SGLang backend, gVisor/Firecracker, Lean/SQL/browser verifiers, repriced RLVR phase |
| 4 | Native multimodality (audio, video, doc, chart, OCR) | 🟥 | $/R/D | `model/vision.py` (LLaVA adapter), no `data/multimodal.py` | Whole second platform |
| 5 | Agentic training environments + SWE-bench harness | 🟧/🟥 | R/D | `rl/agentic.py` (2 toy tools) | Real coding/browser/shell envs + SWE-bench/Terminal-Bench/GAIA |
| 6 | Real safety/red-team harness and classifiers | 🟧 | O/D | `safety/redteam.py` (5 prompts), `safety/classifiers.py` (keyword count) | HarmBench/AdvBench/Cybench/WMDP/METR + Llama-Guard |
| 7 | Real eval harness wired to 2026 benchmarks | 🟧 | O/D | `eval/harness.py` falls back to perplexity if `lm_eval` missing | Wire `lm_eval` + SWE-bench + RULER + MMMU + LiveCodeBench |
| 8 | FP8/NVFP4 numerics validation | 🟧 | $ | `precision.py` is interface-ready; numerics need TE on Hopper/Blackwell | DeepSeek-V3-style FP8 recipe + regression tests |
| 9 | Long-context (1M) + sparse attention | 🟧 | $/R | MLA done; sparse-attn open | DeepSeek-V3.2-style sparse attention + RULER eval |
| 10 | Production serving stack (continuous batching, paged KV, spec-decode loop) | 🟧 | $ | `serving/engine.py:48` stub | vLLM/SGLang backend + multimodal/tool-call paths |
| 11 | Organizational discipline (data lineage, contamination ops, eval gating, RSP) | 🟧 | O | Shape is in `safety/gates.py`; content thin | A safety/data/eval org of 30-100 people |

---

## 6. Bottom line, second pass

The blueprint **as written** would not produce a 2026 frontier-flagship model.
But the reason is not what v1 (and many casual readings of the gap doc) imply.
The reason is *not* "the architecture is dense and 2024-era" — that is no
longer true. The reasons are, in order:

1. **The model code is single-device.** Until `parallel.py` learns TP/PP and
   `MoEFFN` learns expert-parallel all-to-all, the simulator's 1T-MoE bill
   is fiction in the sense of "we couldn't physically run it" — though it
   is honest fiction, because the simulator says so explicitly.
2. **The post-training plant is a skeleton at toy compute.** The algorithm
   (GRPO) is right; the throughput backbone (vLLM) is stubbed; the verifier
   fleet's coverage is shallow; the RLVR compute budget priced by the
   simulator is two orders of magnitude smaller than the o1/R1 direction.
3. **The data factory is half a factory.** Classical pipeline is real;
   synthetic + reasoning-trace + agentic-trajectory + multimodal data
   is essentially absent.
4. **The safety/eval harnesses are placeholders.** The *gates* are the
   right shape; the *content* inside the gates is 5 hardcoded prompts and
   a regex. That is fine for a blueprint; it is not fine for promoting a
   frontier model to production.

Items 1, 3 (parts), and 4 are mostly **engineering and organization** —
solvable with the team and budget the premise assumes. Item 2 partially and
items 3 (synthetic data, reasoning traces) and 4 (real dangerous-capability
evals) are **research + data-organization** problems where unlimited GPUs
do not move the needle. **That last category is where a frontier-flagship
program actually differentiates.**

If I had to write the v3 line for this: the blueprint is now a
**directionally-correct toy-functional skeleton of a 2025 frontier program**.
Closing the gap to a 2026 flagship is roughly **18-24 months of focused
engineering + a real data and safety org**, not a research program. The
parts that *need* research bets are: multimodal native architecture beyond
adapters, agentic RL beyond toy tools, reasoning-trace data curation at
scale, and the dangerous-capability evals that actually drive RSP gates.
Everything else is execution.

---

## Sources

- DeepSeek-AI, **DeepSeek-V3 Technical Report**, arXiv 2412.19437 — sparse MoE
  671B/37B, MLA, aux-loss-free balancing, MTP, FP8 training, 14.8T tokens,
  SFT/RL. Cited for the public direction of frontier architecture.
- DeepSeek-AI, **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
  Reinforcement Learning**, arXiv 2501.12948 / *Nature* 645:633-638 (2025) —
  RL on verifiable rewards as the path to emergent reasoning (self-reflection,
  verification, dynamic strategy adaptation).
- Shao et al., **DeepSeekMath: Pushing the Limits of Mathematical Reasoning
  in Open Language Models**, arXiv 2402.03300 — Group Relative Policy
  Optimization (GRPO), the PPO variant the post-training community has
  largely adopted.
- Ke et al., **A Survey of Frontiers in LLM Reasoning: Inference Scaling,
  Learning to Reason, and Agentic Systems**, arXiv 2504.09037 v4 (2026-03,
  TMLR Survey Certification) — the shift from inference-scaling to learning-
  to-reason, the rise of agentic workflows.
- Internal docs: `docs/14-gap-analysis-vs-frontier.md` (the in-repo gap doc
  this builds on), `docs/15-reasoning-rl-rlvr.md`, `docs/16-multimodality.md`.
- Code referenced inline by path and line number throughout §1 and §2 so
  every claim is verifiable.
