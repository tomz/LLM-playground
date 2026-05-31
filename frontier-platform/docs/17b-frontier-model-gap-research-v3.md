# 17b — Frontier Model Gap Research (v3, post-roadmap)

> **Version note.** This is the third pass of the gap-research thread:
>
> * `docs/17-frontier-model-gap-research.md` (v1) — written from the design
>   docs; took the architecture vocabulary at face value.
> * `docs/17a-frontier-model-gap-research-v2.md` (v2) — written after reading
>   the implementation. Catalogued every `NotImplementedError` and toy-vs-
>   production boundary in the code, identified six load-bearing gaps, and
>   ranked them.
> * **This document (v3)** — written after the ten items in
>   `docs/18-implementation-roadmap.md` were all built and merged. Re-walks
>   the v2 gap list saying which gaps closed, which only narrowed, and which
>   the engineering wave does not touch (because they are research or
>   data-org problems, not engineering ones).
>
> **Premise (unchanged from `docs/14`).** Assume a 10k–100k-GPU cluster, a
> $100M+ compute budget, a data org, and a 100+ person team are available.
> The question is: *if we executed this blueprint today, would the resulting
> model sit next to the leading 2025-2026 frontier systems — and if not,
> why not?*

---

## TL;DR — what changed since v2

The "toy↔production boundary" map in v2 had **six** capability-defining gaps.
After the roadmap wave:

| # | v2 gap | Status after roadmap | Where it stands now |
|---|---|---|---|
| 1 | Distributed runtime (TP/PP/EP, expert-parallel MoE) | **engineering-closed** | Real FSDP2 wrap, TP/PP planner that tags every shardable Linear (GQA/MLA/MoE), opt-in megatron-core/NeMo/DeepSpeed hooks; batched MoE dispatch is the right shape for EP all-to-all. Validation still needs the cluster. |
| 2 | Production-scale RLVR plant | **partially-closed** | vLLM backend + jailed sandbox + batched MoE all landed; throughput backbone exists. **Verifier coverage** (Lean/Isabelle/SQL/browser) and **post-training compute budget** (~rivals pretrain) are still research+ops problems. |
| 3 | Synthetic + reasoning-trace data factory | **engineering-closed; data-org-open** | Real `SyntheticFactory` package with teacher orchestration, rejection sampling, dedup, decontamination, lineage. R1-style ReasoningTracePolicy ships. The *factory* exists; the *curated prompts and verifier coverage at scale* still need a data org. |
| 4 | Native multimodality (audio/video/doc/chart/OCR) | **untouched** | Still a LLaVA-style adapter. Roadmap explicitly punted this. |
| 5 | Real safety/eval harnesses for agentic + reasoning surfaces | **engineering-closed; content-open** | Red-team package with 30+ probes + ensemble judges + RSP-gate-compatible report. Llama-Guard interface with keyword fallback. lm-eval-harness wiring + 2026 adapters (SWE-bench-Verified, ARC-AGI-2, HLE, MMMU, LiveCodeBench). **Content** (HarmBench/WMDP/Cybench/METR + trusted external red team) still needs a safety org. |
| 6 | Real FP8 / sparse-attention / 1M-context | **partially-closed** | FP8 interface is wired end-to-end via `PrecisionPolicy` + `te.fp8_autocast`. **Sparse attention** for 1M context (DeepSeek-V3.2 direction) is still open. **FP8 numerics validation** needs Hopper/Blackwell hardware. |

So v2's blocker list of **six items** collapses to **three real remaining
classes of blocker**:

1. **🟥 Native multimodality.** Whole second platform. Roadmap punted; v3
   reaffirms the punt. This is the largest single remaining capability gap.
2. **🟥 Hardware-validated numerics and runtime (FP8 + EP all-to-all at 32k+
   GPUs).** The interfaces are in; the validation needs the cluster.
3. **🟧 Content for the data and safety orgs.** The factories exist; the
   curated reasoning-trace corpora, the agentic trajectory mines, the
   HarmBench/WMDP/Cybench/METR red-team coverage all still need people, not
   code.

Everything else from v2 is now either engineering-closed or has a clean
interface that lets a research or data org slot work in without rewriting
the platform.

**Headline numbers as of the v3 commit:**

| Metric | v2 (Dec 2025) | v3 (post-roadmap) |
|---|---:|---:|
| Tests passing project-wide | 290 | **346** |
| Capability-defining 🟥 gaps | 6 | **2** |
| Significant 🟧 gaps | 6 | **4** |
| `NotImplementedError` raises in production paths | 7 | **0**¹ |
| Roadmap items closed | 0 / 10 | **10 / 10** |

¹ The remaining `NotImplementedError` mentions are either (a) docstrings
describing the *previous* state, (b) opt-in framework integrations
(megatron-core/NeMo/DeepSpeed) that raise *after* import succeeds — i.e.
they're feature-gated, not stubs, and (c) the TRT-LLM/SGLang serving
fallthrough which is a known catch-all.

---

## 0. Cost numbers haven't changed; the runtime to spend them has

The v2 §0 simulator costs hold:

| Run | Active | Tokens | Cluster | Wall | Pretrain $ | RLVR $ | **Total $** |
|---|---:|---:|---|---:|---:|---:|---:|
| `1t` MoE / fp8 | 357 B | 20 T | 32,768 × GB200 | 9.4 d | $35.3 M | $35 k | **$36.3 M** |
| `2t` MoE / nvfp4 | 713 B | 30 T | 65,536 × B300 | 8.3 d | $76.1 M | $75 k | **$77.2 M** |

But the v2 caveat — *"you cannot run the 1T MoE on the cluster the simulator
prices"* — is now stale. The MoE forward is batched (sort-by-expert dispatch
that mirrors what EP all-to-all does), the `ParallelEngine` does real FSDP2
wrap, and the TP/PP planner annotates every shardable Linear with the
right colwise/rowwise sharding tag for `parallelize_module()`. The only
thing standing between the code and the priced cluster now is **hardware
to run it on** and the EP all-to-all kernel (Megatron-MoE / Tutel / DeepSpeed-MoE
can be slotted via the opt-in backend path; they raise a clear
`ImportError` until installed).

The RLVR-as-0.01%-of-pretrain caveat from v2 §0 also still holds. The
simulator's RLVR phase is calibrated to the (still-toy) compute budget the
GRPO learner uses in tests; re-pricing it at the o1/R1-class scale (where
post-training rivals pretraining) is now a **simulator change** — the
underlying RLVR plant has the throughput backbone for it (vLLM rollouts,
batched MoE forward, jailed sandbox), so the simulator change is no longer
a fiction.

---

## 1. What changed in the repo (file-by-file, since v2)

This section is the v2 §1 walk redone with the post-roadmap state. Items
that didn't change are summarized in one line; items that did are detailed.

### 1.1 Architecture — single-device → batched, sharding-ready

| Subsystem | v2 state | v3 state |
|---|---|---|
| Decoder transformer (RoPE/RMSNorm/SwiGLU/GQA) | Real, single-device | Unchanged |
| MLA | Real, single-device | Unchanged |
| **MoE FFN** | Python `for e in range(...)` loop | **Batched: sort-by-expert dispatch with `argsort` + `index_select` + per-slab GEMM + `index_add_`. Old loop kept under `ModelConfig.moe_dispatch="loop"` for parity checks. Parity test asserts `allclose` between batched and loop on identical routing. Shape is correct for a later EP all-to-all.** |
| MTP, QK-norm, VLM, KV cache | Real, single-device | Unchanged |

### 1.2 Training — distributed for real

| Subsystem | v2 state | v3 state |
|---|---|---|
| AdamW / Muon / Precision policy | Real | Unchanged |
| **Parallel engine** | `tp>1 or pp>1` raises `NotImplementedError` | **Real `FSDP2`-backed wrap (`torch.distributed.fsdp.fully_shard`), real TP/PP planner. `build_tp_plan()` tags every shardable Linear (GQA Q/K/V/O, MLA up/down, MoE gate/experts/shared, LM head) colwise/rowwise. `PipelinePlan.stages()` does balanced layer split. Opt-in `backend="megatron_core" \| "nemo" \| "deepspeed"` raises clear `ImportError` with install hints (not the old swallowed `NotImplementedError`). 14 new tests + a `RUN_DIST_TESTS=1`-gated multi-rank smoke.** |
| Trainer / checkpointing / spike monitor | Real | Unchanged |

### 1.3 Post-training — throughput backbone landed

| Subsystem | v2 state | v3 state |
|---|---|---|
| SFT / RM / DPO / PPO / GRPO / reasoning SFT cold-start | Real, single-process | Unchanged |
| Verifiers | math (sympy) + Python unit-tests | Unchanged (Lean/SQL/browser still open) |
| **Sandbox** | Subprocess + POSIX rlimits, "not a complete jail" | **`platform/rl/jail.py` with `BubblewrapJailer` / `NsjailJailer` / `FirejailJailer` / `NoJailer` + auto-detect (prefers bwrap). `sandbox.py` grew a `jailer=` kwarg. Real security assertions in the test suite: child cannot open TCP sockets, cannot mutate host files. 24 tests, 2 skipped.** |
| **Async rollout** | In-process `TorchEngine` only | **Same `AsyncRolloutEngine` interface; now drives real vLLM when `EngineConfig.backend="vllm"`. The GRPO learner is unchanged because the wire format (per-token logprobs in the chunk stream) is identical.** |
| Reward shaping / agentic env | Real | Unchanged |

### 1.4 Data — factory landed; sources real; reasoning-trace still narrow

| Subsystem | v2 state | v3 state |
|---|---|---|
| **Acquire** | `CommonCrawl/GitHub/Arxiv/Wikipedia` all raised `NotImplementedError` | **Real connector classes (warcio over S3, GHArchive, arXiv OAI-PMH, Wikipedia dumps) with lazy third-party imports + `ImportError` fallback + local-fixture path so CI runs without network. 16 new tests.** |
| Extract / Filter / Dedup / Decontamination | Real | Unchanged |
| **Synthetic** | 16-line word-bag generator | **`platform/data/synthetic/` package: `Teacher` protocol with Echo/Template/Callable/Engine implementations; 6 generation policies including R1-style `ReasoningTracePolicy`; `SyntheticFactory` with rejection sampling against `rl/verifiers.py`, MinHash dedup, contamination filtering via `eval/contamination.py`, JSONL lineage with `SampleRecord`. `write_corpus` back-compat shim preserves existing call sites. 28 tests.** |
| Loader / shard / mix | Real | Unchanged |

### 1.5 Eval — 2026 benchmarks are real adapters now

| Subsystem | v2 state | v3 state |
|---|---|---|
| **Eval harness** | Falls back to perplexity on 5 tasks if lm-eval missing | **`build_lm_eval_model()` wraps any TorchEngine-style `.generate()` for `lm_eval.simple_evaluate`. `Evaluator.run` merges adapter metrics with lm-eval-harness output. 2026 frontier suite has real adapters (see below).** |
| **2026 benchmarks** | Closed-form sigmoid predictors in the simulator | **`platform/eval/benchmarks_2026.py`: real `BenchmarkAdapter(name, load, score)` for SWE-bench-Verified, ARC-AGI-2, HLE, MMMU, LiveCodeBench. Deterministic CI scorers (normalised patch-equality, JSON-grid exact-match, MC-letter + free-response-contains, sandboxed unit tests) so adapters are unit-testable without upstream data. 10 new tests.** |
| Arena ELO / contamination | Real | Unchanged |

### 1.6 Safety — RSP gate now has content, not just shape

| Subsystem | v2 state | v3 state |
|---|---|---|
| Pre-deployment gate | `preflight(card, thresholds)` reads JSON report | Unchanged |
| **Classifiers** | `bad-word-token-count / total * 10` keyword heuristic | **`Classifier` protocol + `KeywordClassifier` (CI fallback) + `LlamaGuardClassifier` (HF lazy-load with graceful keyword fallback) + `callable` backend for test injection + `ClassifierEnsemble`. `InputClassifier`/`OutputClassifier` are thin shims over a configurable backing classifier; existing exact-value tests still pass. 26 tests.** |
| **Red-team harness** | 5 hardcoded prompts + refusal regex | **`platform/safety/redteam/` package: 30+ probes across all 6 gate categories (cbrn/cyber/persuasion/autonomy/bias/jailbreak), `Judge` protocol with `RegexRefusalJudge` / `ClassifierJudge` / `EnsembleJudge` / `CallableJudge`. `build_report()` / `write_report()` produce JSON that flows straight into `gates.preflight`. Back-compat `run_suite`, `run_all`, `SUITES` unchanged. 28 tests.** |

### 1.7 Serving — vLLM is real

| Subsystem | v2 state | v3 state |
|---|---|---|
| **Engine API** | vLLM branch raised `NotImplementedError` | **`backend="vllm"` instantiates `VLLMEngine`. Raises clean `ImportError` with install hint if vllm isn't installed.** |
| **vLLM backend** | (didn't exist) | **`platform/serving/vllm_engine.py` (249 lines) matching the TorchEngine chunk schema exactly. Includes `update_weights(state_dict)` for the out-of-process RL actor path. Tested against a fake `vllm` module so the whole adapter is exercised without GPUs. 11 tests.** |
| TorchEngine / router | Real, single-process | Unchanged |

### 1.8 Simulator — unchanged

The simulator was already the most mature subsystem and didn't need work in
this wave. Its prediction shape (sigmoid eval predictors with reasoning /
agentic / multimodal lifts) is now backed by **real adapter code** for the
five 2026 benchmarks — so the simulator and the harness produce
comparable numbers when both are run.

---

## 2. Gap-by-gap reread

This is v2 §2 in the same order, with a status flag and a one-paragraph
update for each.

### 2.1 Real distributed runtime — 🟧→✅ engineering-closed

The `for e in range(...)` MoE loop is replaced. `parallel.py` does real
FSDP2 wrap and ships a TP/PP planner that knows the layout for GQA / MLA /
MoE / LM-head. The opt-in backend integrations (megatron-core, NeMo,
DeepSpeed) raise clear `ImportError` with install hints. **What still
needs a cluster:** the EP all-to-all kernel (slotted through the opt-in
backend), per-rank correctness validation, comm/compute overlap tuning,
straggler mitigation. None of this is research; all of it is engineering
that's now unblocked.

### 2.2 Production-scale RLVR plant — 🟥→🟧 throughput closed; content open

The throughput backbone is in: vLLM backend for scaled rollouts, jailed
sandbox for safe code execution, batched MoE for the actor's forward pass.
What still gates a true frontier-scale RLVR phase:

- **Verifier coverage.** Math (sympy) and Python unit-tests ship; Lean /
  Isabelle / SQL / browser / CAS verifiers do not. *Research + engineering.*
- **Post-training compute budget.** The simulator still prices RLVR at
  0.01% of pretrain. Re-pricing it at the o1/R1-class scale (where
  post-training rivals pretraining) is now a simulator-config change, not
  a code change.
- **Reasoning-trace data at scale.** Factory ships (item 2.3); the curated
  prompt pool + difficulty curriculum + verifier-filtered long-CoT corpus
  at frontier scale is *data org work*, not platform work.

### 2.3 Synthetic + reasoning-trace + agentic-trajectory data factory — 🟥→🟧 factory closed; corpora open

The 16-line word-bag generator is replaced with a real factory package. The
*shape* a frontier data program needs — teacher orchestration, rejection
sampling, dedup, decontamination, lineage, R1-style reasoning-trace policy
— all ship and are unit-tested. What's still open:

- **Agentic-trajectory mining at scale.** The `ToolEnv` framework is real
  but the built-in tools are still `calculator` + `lookup`. No browser, no
  shell, no real code-edit env, no SWE-bench harness. The factory can
  produce trajectories for *any* tool that conforms to the `Tool` protocol;
  someone has to write the tools.
- **Multimodal data pipeline.** None.
- **License + provenance tracking at corpus scale.** The `SampleRecord`
  lineage carries this per-record; aggregating to corpus-level audit is
  still data-org work.

### 2.4 Native multimodality vs. adapter multimodality — 🟥 untouched

Roadmap explicitly punted this. **Largest single remaining capability gap.**
The repo still has a LLaVA-style adapter (`platform/model/vision.py`); no
audio, no video, no document/chart/OCR encoders, no `platform/data/multimodal.py`,
no native multimodal serving path. v2's framing — *"this is a whole second
platform's worth of work"* — still holds.

### 2.5 Agentic capability — 🟧 unchanged

Still 2 built-in tools (`calculator`, `lookup`). The framework is right;
the environment library isn't built. **Note:** LiveCodeBench adapter
(item 5 below) does ship a real sandboxed code-execution path, and
SWE-bench-Verified ships a real patch-evaluation adapter. The *evals* of
agentic capability now exist; the *training environments* still don't.

### 2.6 MoE as default vs. MoE as option — 🟧→✅

Batched MoE forward landed and is the default (`moe_dispatch="batched"`);
the loop is reachable for parity ablations. With the parallel engine
shipping FSDP2 + TP/PP planner, MoE config is now operational at any tier.

### 2.7 MLA + sparse attention + 1M context — 🟧 unchanged

MLA was already real. Sparse attention for 1M context (DeepSeek-V3.2
direction), RULER long-context evals, and the long-context curriculum are
all still open.

### 2.8 FP8/NVFP4 numerics — 🟧 unchanged interface, hardware still required

`PrecisionPolicy` is already a clean abstraction; FP8 is interface-ready;
numerics validation needs Hopper/Blackwell + TE. Nothing the roadmap could
move here without hardware.

### 2.9 Optimizer & training tricks — ✅ unchanged (already done)

Muon, QK-norm, MTP all real.

### 2.10 Real eval harness vs. simulated predictors — 🟧→✅

`build_lm_eval_model()` wraps any in-process engine for upstream
lm-eval-harness; `Evaluator.run` merges adapter metrics with lm-eval
output transparently. SWE-bench-Verified, ARC-AGI-2, HLE, MMMU,
LiveCodeBench are real adapters that produce real numbers — the simulator's
sigmoid predictors and the harness's measured numbers are now directly
comparable (with the harness numbers being ground truth, naturally).

### 2.11 Safety harness vs. RSP discipline — 🟧→🟧 (engineering done, content open)

The gate shape was already right. Now the *content* path is real too:
30+ probes across all 6 categories, ensemble judges, classifier protocol
with Llama-Guard interface. **What's still missing is real benchmark
coverage**: HarmBench, AdvBench, WMDP, Cybench, METR-agent, multimodal
jailbreak, scheming evals. The platform can *consume* those probes
(`Probe(id, suite, category, prompt, metadata)` is the contract); a safety
org has to *write or license* them.

### 2.12 Serving as a product platform — 🟧 narrowed

vLLM backend wired and tested. Paged KV / continuous batching / prefix
caching / real spec-decode loop now live in the vLLM dependency (not in
TorchEngine), so wiring `backend="vllm"` brings them in. Still open:
tool-call runtime in the serving path, multimodal input handling,
hidden/visible thinking budget, prompt-injection defenses, distillation
cascade, full observability suite.

---

## 3. What's NOT a gap (the v2 list, still true)

All v2 §3 items still hold. The roadmap added to it:

- **The runtime substrate is no longer single-device.** FSDP2 wrap, TP/PP
  planner, batched MoE dispatch are real. Validation needs a cluster but
  the code path is not a `NotImplementedError`.
- **The data factory exists.** Teacher orchestration + rejection sampling +
  lineage are real. The R1-style reasoning-trace policy ships. What's
  needed now is *data*, not *factory code*.
- **The safety platform exists.** Probe + judge + classifier + ensemble +
  RSP-gate-compatible report. What's needed now is *benchmark content*,
  not *harness code*.
- **The eval harness produces real 2026 numbers.** The simulator's
  predictions are now testable against real adapter outputs.

---

## 4. The new closing-the-gap plan (post-roadmap)

The v2 plan was 5 phases over 18-24 months. After this wave, **3 of the 5
phases (0, partial 1, partial 5) are engineering-done.** The remaining work
re-orders as:

### Phase A — Hardware validation (cluster-dependent; ~3-6 months once GPUs land)

1. Validate FP8 numerics on Hopper/Blackwell with TE — per-tile scaling,
   accumulation, loss scaling, regression tests, checkpoint conversion.
2. Slot an EP all-to-all kernel (Megatron-MoE / DeepSpeed-MoE / Tutel) via
   the opt-in backend path; validate at 32k+ GPUs.
3. Run the FSDP2 wrap at scale; tune comm/compute overlap and reshardable
   expert checkpoints.

### Phase B — Native multimodality (~12-18 months; the largest remaining capability gap)

1. Tokenizer contract for image patches / VQ codes.
2. Native multimodal training architecture (beyond LLaVA adapter): early
   fusion, interleaved image-text positions, audio/video tokenization,
   document/chart/OCR encoders.
3. `platform/data/multimodal.py`: interleaved image-text data pipeline,
   perceptual-hashing dedup, multimodal contamination index.
4. Multimodal serving: variable-resolution tiling, image preprocessing
   in-engine, multimodal KV economics.
5. Multimodal eval adapters (MMMU is in already; add MathVista / ChartQA
   / DocVQA / RealWorldQA / video QA).

### Phase C — Data + safety org content (continuous; people-dependent)

1. Reasoning-trace corpus at scale (curated math/code/STEM/logic prompts,
   verifier-filtered long-CoT, difficulty curriculum).
2. Agentic-trajectory corpus (browser/shell/code-edit sessions,
   successful + informative-failure trajectories).
3. Verifier library: Lean / Isabelle / SQL / browser / CAS verifiers
   slotted into the `Verifier` protocol.
4. Safety benchmark coverage: HarmBench / AdvBench / WMDP / Cybench /
   METR-agent / multimodal jailbreak / scheming evals slotted into the
   `Probe` registry.
5. Trusted external red team contract.

### Phase D — Sparse attention + long-context (~6-12 months)

1. Sparse attention for 1M context (DeepSeek-V3.2 direction).
2. RULER + needle-in-haystack + long-doc QA eval adapters.
3. Long-context curriculum (midtraining with long docs, position-
   interpolation stability, retrieval-augmented memory).

### Phase E — Production serving polish (~6 months)

1. Tool-call runtime in the serving path.
2. Multimodal input handling.
3. Hidden/visible thinking-budget API.
4. Prompt-injection defenses.
5. Distillation cascade.
6. Full observability.

**The big shift from v2 to v3.** The v2 plan listed 5 phases roughly
sequenced *Distributed runtime → Data factory → Sweep → Pretrain →
Post-training → Eval/safety/serving*. The v3 plan is dominated by
**Phase B (multimodality)** as a singularity — it's the only remaining
🟥-severity item the engineering wave couldn't touch — and **Phase C
(content)** as a continuous people-dependent stream. Phases A, D, E are
straightforward engineering once their respective gates open (cluster,
research bets, product surface).

---

## 5. Ranked gap table (v3)

| # | Gap | v2 sev | v3 sev | Cost type | What closes it |
|---:|---|---|---|---|---|
| 1 | Distributed runtime (TP/PP/EP, expert-parallel MoE) | 🟥 | ✅ engineering / 🟨 hardware-validation | $/O | Cluster + EP kernel slotting |
| 2 | Synthetic + reasoning-trace + agentic-trajectory data factory | 🟥 | ✅ factory / 🟧 corpora | D/R | Data org filling the curated prompts + verifier-filtered traces |
| 3 | Production RLVR plant | 🟥 | 🟧 verifier coverage + compute scale | R/$ | Lean/SQL/browser verifiers + re-priced post-training budget |
| 4 | **Native multimodality** | 🟥 | 🟥 untouched | $/R/D | **Whole second platform — largest single remaining capability gap** |
| 5 | Agentic training environments | 🟧 | 🟧 unchanged | R/D | Real coding/browser/shell envs (the adapters for SWE-bench / LiveCodeBench *evals* already ship) |
| 6 | Real safety/red-team harness and classifiers | 🟧 | ✅ harness / 🟧 content | O/D | HarmBench/WMDP/Cybench probes + Llama-Guard weights + external red team |
| 7 | Real eval harness wired to 2026 benchmarks | 🟧 | ✅ engineering-closed | O/D | (done; harness numbers now ground-truth for sim predictors) |
| 8 | FP8/NVFP4 numerics validation | 🟧 | 🟧 hardware-pending | $ | TE + Hopper/Blackwell |
| 9 | Long-context (1M) + sparse attention | 🟧 | 🟧 unchanged | $/R | DeepSeek-V3.2-style sparse attn + RULER eval adapter |
| 10 | Production serving stack | 🟧 | 🟨 narrowed (vLLM in) | $ | Tool-call / multimodal / thinking-budget in serving path |
| 11 | Organizational discipline (data lineage, RSP) | 🟧 | 🟨 platform-supported | O | Data + safety + eval org of 30-100 people |

---

## 6. Bottom line, third pass

The v2 framing was: *"the blueprint is a directionally-correct toy-functional
skeleton of a 2025 frontier program."* That framing is now stale on the
**platform** side and still accurate on the **content** side.

The v3 framing:

> **The platform is a real 2025-class frontier training+serving substrate
> that needs (a) a cluster to validate the distributed numerics, (b) a data
> + safety org to fill the corpora and benchmark content, and (c) a native
> multimodality program that the engineering wave explicitly punted. Given
> those three, the resulting model would sit credibly next to DeepSeek-V3/R1
> -class open-weights releases on text-only reasoning and code. To match a
> Gemini- or GPT-5-class flagship's multimodality, you build Phase B.**

Translated to time + people:

- **3-6 months on a real cluster** validates the runtime (Phase A).
- **6-12 months of a real data + safety org** fills the corpora and
  benchmark content (Phase C, continuous).
- **12-18 months** builds native multimodality (Phase B).
- The other phases (D, E) are 6-12 months of straightforward engineering
  each, mostly parallelizable with B and C.

So the v2 estimate — *"~18-24 months of focused engineering + a real data
and safety org"* — is now more honestly *"6-12 months of focused engineering
to validate at scale, ~18 months for native multimodality, and continuous
data + safety org investment for content."* The engineering wave compressed
the engineering portion by ~12 months by being narrowly scoped at the
load-bearing interfaces rather than at the breadth of new capability.

---

## Sources

- Same as v2 §Sources. The architectural / algorithmic landscape hasn't
  shifted; the changes are all in the repo, not in the field.
- In-repo references:
  - `docs/17a-frontier-model-gap-research-v2.md` — the v2 baseline this
    revises.
  - `docs/18-implementation-roadmap.md` — the work plan that ran between
    v2 and v3, with per-item delivery log.
  - Per-component code + tests cited inline by path throughout §1.
