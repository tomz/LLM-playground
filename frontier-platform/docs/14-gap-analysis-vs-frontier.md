# 14 — Gap Analysis: `frontier-platform` vs. the 2025–2026 Frontier

> **Premise.** Assume the resource constraint is lifted — a 10k–100k-GPU
> cluster, a $100M+ compute budget, a data org, and a 100+ person team are all
> available. The question is no longer *"can we afford a run?"* but *"if we
> executed this blueprint exactly as written, would the resulting model sit
> next to GPT‑5.x, Claude Opus 4.x, and Gemini 3.x — and if not, why not?"*
>
> **Verdict up front.** This blueprint is an excellent **2024-era** dense-LLM
> training platform. It would produce a credible **GPT‑4-class / Llama‑3‑class**
> model. It would **not**, as written, produce a 2025–2026 frontier model,
> because the frontier moved from *"pretrain a big dense transformer + RLHF"* to
> *"sparse MoE + large-scale RL on verifiable rewards + inference-time reasoning
> + native multimodality + agentic tool-use,"* and those are exactly the parts
> this repo either stubs, omits, or models with 2024 assumptions.
>
> Sources are listed at the end. Named models (GPT‑5.x, Opus 4.x, Gemini 3.x)
> are referenced via their publicly documented lineage and the 2025–2026
> technique literature; exact frontier recipes are not public, so gaps are
> stated against the *direction of travel*, not leaked specs.

---

## How to read this

Each gap is tagged with where it lives and how hard it is to close **given
unlimited compute** (because some gaps are compute-solvable and some are
research/data/organizational and money alone will not fix them):

- 🟥 **Capability-defining** — without this you are a generation behind, full stop.
- 🟧 **Significant** — measurable quality/cost gap; closeable with known methods.
- 🟨 **Catch-up** — modern hygiene the blueprint predates; cheap to add.

Difficulty: **$** = mostly compute/eng, **R** = needs research bets, **D** =
needs a data/labeling org, **O** = organizational/process.

---

## 1. Post-training is a generation behind 🟥 (R + D)

**What the blueprint says** (`07-alignment-sft-rlhf.md`): SFT → Bradley-Terry
reward model → PPO/DPO/IPO/KTO, with RLHF framed as "the stage where models
silently degrade." The entire alignment doc is a 2023–2024 **preference-alignment**
pipeline. The skeleton code matches: SFT, BT reward model, DPO, PPO with GAE.

**Where the frontier is.** The single biggest shift of 2025 — *"the year of
reasoning, RLVR, and GRPO"* — is essentially absent:

- **RLVR (RL with Verifiable Rewards).** DeepSeek-R1 showed reasoning behavior
  *emerges* from RL against deterministic correctness signals (unit tests,
  exact-answer math, compilers, formal checkers) instead of a learned reward
  model. Every frontier lab now ships a "thinking" variant trained this way.
  The blueprint mentions verifiers only as a one-line *data source* for the RM,
  not as the **central post-training compute sink** they have become.
- **GRPO and successors.** Group-Relative Policy Optimization (DeepSeekMath)
  drops PPO's value network, cutting RL memory ~2× and making large-scale RL
  practical. The blueprint's policy-optimization menu is PPO/DPO/IPO/KTO — it
  predates GRPO entirely.
- **Long-CoT / inference-time reasoning as a training target.** Frontier models
  are explicitly trained to spend test-time compute (long reasoning traces,
  self-verification, backtracking). There is **no notion of a reasoning trace,
  thinking budget, or test-time scaling** anywhere in the model, training, eval,
  or serving docs.
- **RL infrastructure.** Modern RLVR needs a generation-heavy actor/learner
  loop: thousands of rollouts per step, an async inference engine feeding a
  trainer, reward workers running sandboxed code/math verifiers at scale. The
  blueprint's alignment code is a synchronous single-process loss function. This
  is the **largest missing subsystem** in the entire repo.

**Why money alone doesn't close it.** RLVR reward design, verifier coverage
beyond math/code, and reasoning-data curation are active research + data
problems. Unlimited GPUs help you *run* RLVR; they don't tell you *what to
reward*.

**To close:** add a `platform/rl/` subsystem (async rollout engine + reward
workers + GRPO/PPO-with-verifier learner); add reasoning traces to the data and
tokenizer contract; make "thinking mode" a first-class model + serving feature.

> **Update:** a *toy-functional* `platform/rl/` now exists and has been deepened
> beyond the basic loop (see `docs/15-reasoning-rl-rlvr.md`): verifiers + group
> rollout + **GRPO learner** + **reasoning-SFT cold-start** (`coldstart.py`,
> R1-style) + **composite reward shaping** (`reward.py` — format/length rewards
> plus repetition/answer-spam reward-hacking guards). It proves the
> cold-start→sample→verify→shape→advantage→update loop on CPU. The production
> gap — async vLLM/SGLang rollout, sandboxed code verifiers, reasoning-data
> curation at scale — is still open.

---

## 2. Default architecture is dense; the frontier is sparse MoE 🟥 ($ + R)

**What the blueprint says** (`03-model-architecture.md`): "Default: decoder-only
transformer," dense. MoE is an *"Optional"* section — Mixtral-style, 8 experts,
top-2, capacity factor 1.25. Reference shapes (1B/7B/70B/400B) are **all dense**.
The code implements a basic top-k MoE with z-loss + load-balance loss.

**Where the frontier is.** Every leading 2025 model is a **large sparse MoE**
(DeepSeek-V3: 671B total / 37B active; Llama-4, Qwen3, and the frontier
proprietary models are MoE). The frontier default *is* sparse; dense is the
exception. Specific deltas:

- **Fine-grained experts + shared expert(s)** (DeepSeek-style): dozens-to-hundreds
  of small experts with one always-on shared expert. The blueprint's 8-expert
  top-2 is the 2023 Mixtral recipe.
- **Auxiliary-loss-free load balancing** (bias-based routing): DeepSeek-V3 moved
  off the aux-loss the blueprint still uses, because aux-loss trades quality for
  balance.
- **MoE as the *primary* scaling axis.** The whole cost/scaling doc reasons in
  dense `6·N·D` FLOPs; it never models active-vs-total parameter economics,
  which is *the* lever frontier labs pull to get 600B-param quality at 37B-param
  training/inference cost.

**To close:** make MoE the default, not optional; implement fine-grained +
shared-expert routing and aux-loss-free balancing; rewrite the scaling/cost
model around active parameters.

> **Update:** the *simulator* models MoE economics — `moe_active_params()`
> in `platform/sim/scaling.py` plus `--moe-experts/--moe-top-k` flags route the
> cost/throughput model through active (not total) parameters, so a 1T-total /
> ~357B-active run prices like its active size. **The model code now also
> implements the frontier recipe**: `MoEFFN` supports fine-grained experts
> (`moe_expert_d_ffn`), always-on shared expert(s) (`moe_shared_experts`), and
> aux-loss-free bias-based load balancing (`moe_balance="aux_free"`, the
> default), with `ModelConfig.active_param_count()` exposing the per-token cost.
> See `configs/model_moe_1t.yaml` (128 experts, top-8, 1 shared, ~1T/~37B). What
> remains is making MoE the *default config* for large tiers (the dense 1B/7B
> baselines stay) and real expert-parallel dispatch — the single-device loop is
> a correctness reference, not a throughput one.

---

## 3. No Multi-head Latent Attention / modern KV compression 🟧 ($ + R)

**What the blueprint says** (`03`): GQA with 8 KV heads, "for inference
efficiency." That's the Llama-2/3 answer.

**Where the frontier is.** **Multi-head Latent Attention (MLA)** (DeepSeek-V2/V3)
compresses the KV cache 5–10× via a low-rank latent projection at near-equal
quality, and **DeepSeek-V3.2 introduced sparse attention** for long context.
KV-cache size is the dominant cost of long-context serving; GQA is a partial
mitigation, MLA is the current frontier answer. The doc lists MLA nowhere; the
code implements only GQA. (Our own SOTA Watch already flags MLA as "planned" —
the frontier-platform docs haven't caught up to our own research digest.)

**To close:** add MLA (or a documented GQA+sparse-attention alternative) as the
attention default for any long-context tier.

> **Update:** **MLA now exists in the model code** — `MLAttention` in
> `platform/model/transformer.py` (set `attn_kind="mla"`): a shared low-rank KV
> latent (`mla_kv_latent_dim`) is the only cached quantity, with a decoupled RoPE
> key carrying position, giving 3-10x KV-cache compression
> (`ModelConfig.kv_bytes_per_token()` quantifies it). The *serving simulator*
> prices the win: `ServingTier(attn_kind="mla", kv_compression=…)` raises
> effective decode throughput so an MLA tier needs fewer GPUs / lower $/Mtok at
> the same QPS. Still open: incremental MLA decode cache in the real serving
> engine, and sparse attention (DeepSeek-V3.2) for the 1M-context tier.
>
> **Update 2:** **incremental KV-cache decode is now implemented** for both GQA
> and MLA — `Transformer.forward_with_cache` + `platform/model/kv_cache.py`
> (`KVCache`). The serving engine (`TorchEngine`) prefills the prompt then decodes
> one token per step over the cache (O(T)/token), and a correctness test asserts
> the cached path matches the full re-encode logits bitwise-close for GQA, MLA,
> and QK-norm. MLA's cache stores only the compressed latent `c_kv` + shared RoPE
> key, not per-head K/V — the actual memory win, now real in code. Sparse
> attention for the 1M tier remains open.

---

## 4. Multimodality is absent 🟥 ($ + R + D)

**What the blueprint says:** Nothing. The data pipeline is text + code. The
tokenizer is text BPE. The model is a text decoder. `00-overview.md` even lists
"multimodal, agents" only as a *budget line* for a "frontier lab," not as a
subsystem.

**Where the frontier is.** GPT‑5.x, Opus 4.x, and Gemini 3.x are **natively
multimodal** (text + image, often audio/video; Gemini especially). This is not a
bolt-on — it touches the data pipeline (interleaved image-text, vision encoders,
modality-specific dedup/filtering), the tokenizer (image patches / VQ tokens /
audio frames), the architecture (vision encoder + projector, or early-fusion
tokens), training (modality-balanced mixes, modality curricula), and eval
(MMMU, video/audio benchmarks).

**Why this is capability-defining.** A text-only model in 2026 is, by
definition, not at the frontier — the flagships are judged on image/video/audio
understanding and on multimodal agentic tasks.

**To close:** this is a whole second platform. Minimum: vision encoder +
projector, interleaved multimodal data pipeline, multimodal tokenizer contract,
multimodal eval suite. (Audio/video are a further tier.)

> **Update:** a *toy-functional* LLaVA-style adapter now exists in
> `platform/model/vision.py` (`VisionEncoder` + `Projector` +
> `VisionLanguageModel`, runs on CPU; see `docs/16-multimodality.md`). It proves
> the image-token-prepend forward/loss path, but the encoder is randomly
> initialized and the data/tokenizer/eval/serving multimodal paths are still
> open. **The simulator now prices multimodal training**:
> `ProgramSpec(multimodal=True, mm_data_frac=…)` inflates effective training
> tokens (vision tokens are heavy) so multimodal runs cost more wall-clock/$, and
> the eval suite scores **MMMU** above chance only when `multimodal=True`.

---

## 5. No agentic / tool-use training or long-horizon eval 🟧 (R + D)

**What the blueprint says:** Tokenizer reserves `<|tool_call|>` (good instinct).
But there is **no tool-use training data, no agentic RL, no multi-step
trajectory training, and no agentic eval** beyond a one-line METR mention in the
safety doc.

**Where the frontier is.** The 2025–2026 flagships are sold as **agents**:
SWE-bench Verified (Opus 4.5 >80%), long-horizon coding, computer/browser use,
multi-tool orchestration, Deep-Research-style workflows. These require training
on **multi-turn tool-use trajectories** and RL over **long-horizon, sparse-reward
tasks** — a different regime from single-turn SFT/DPO. Eval must measure task
completion over long horizons, not just static QA.

**To close:** add tool-use trajectory data + an agentic RL environment harness;
add SWE-bench-style and computer-use evals to `08-evaluation.md`.

> **Update:** a *toy-functional* agentic harness now exists —
> `platform/rl/agentic.py` (`ToolEnv` + `Tool`/`ToolSpec` + `rollout_episode`):
> a multi-turn agent↔env loop where the agent emits `<|tool_call|>` JSON, the env
> runs the tool and returns `<|tool_result|>`, and a **terminal, sparse**
> verifiable reward scores task completion — the agentic-RL regime, not
> single-turn. The *simulator* prices it: `platform/sim/agentic_rl_sim.py` charges
> long-horizon rollout GPU + tool-execution CPU fleet + trajectory labels and
> returns an `agentic_quality` lift, and the eval suite now scores **SWE-bench
> Verified** (driven mostly by `agentic_quality`). Still open: real sandboxed
> tools (code/browser), expert-trajectory SFT data, and the async rollout engine.

---

## 6. Optimizer & training-efficiency stack is 2024 🟨 ($ + R)

**What the blueprint says** (`04-pretraining.md`): AdamW, β=(0.9,0.95), cosine
warmup→decay, grad-clip 1.0. Solid, standard, *and exactly what the field is
moving past for the from-scratch regime.*

**Where the frontier is.**

- **Muon / matrix-aware optimizers** give ~1.35× sample-efficiency on hidden
  weights and are already shipped in our *own* `nanogpt-edu`. The frontier-platform
  optimizer doc doesn't mention them.
- **Multi-Token Prediction (MTP)** (DeepSeek-V3) densifies the gradient and
  doubles as a speculative-decoding draft head — also already in `nanogpt-edu`,
  absent here.
- **WSD / constant-then-decay schedules** are increasingly preferred over cosine
  for runs where the token budget isn't fixed in advance.
- **QK-norm and related stability tricks** — present in our `nanogpt-edu`, not in
  the frontier-platform model.

**Irony to fix:** our small-scale repos are *ahead* of our frontier blueprint on
training tricks. The blueprint should at minimum cross-reference and adopt them.

> **Update (partial):** **QK-norm now exists in the model code** — `qk_norm=True`
> applies per-head RMSNorm to queries and keys before attention (the
> `nanogpt-edu` stability trick), in `GQAttention`. Muon, MTP, and WSD schedules
> remain to be ported from `nanogpt-edu`.

---

## 7. FP8/low-precision is mentioned but under-committed 🟨 ($)

**What the blueprint says:** "FP8 matmul on Hopper/Blackwell via Transformer
Engine" — one line, and the simulator's headline runs are modeled at "50% MFU ×
spec-sheet," i.e. **bf16 economics**.

**Where the frontier is.** DeepSeek-V3 did a **full FP8 training run** at 671B/14.8T
and documented the recipe (per-tile/per-block scaling, FP32 accumulation,
selective high-precision). NVFP4 (Blackwell) pushes toward 4-bit training. This
is a real ~1.5–2× cost lever the blueprint name-checks but doesn't design for or
price in. The cost tables would shrink materially if FP8 economics were modeled.

**To close:** make FP8 a first-class training-numerics design (not a footnote);
re-baseline the simulator's cost/throughput with FP8 and MoE active-param FLOPs.

> **Update:** the simulator now prices low precision — `precision_speedup()` in
> `platform/sim/scaling.py` (bf16 1.0 / fp8 1.55 / nvfp4 2.2) plus a
> `--precision` flag fold the throughput gain into pretraining wall-clock and
> $. The *training code* doesn't yet implement FP8 numerics; the economics are
> modeled.

---

## 8. Data: strong classical pipeline, missing the 2025 data thesis 🟧 (D + R)

**What the blueprint says** (`01-data-pipeline.md`): genuinely good — Gopher
heuristics, FineWeb-Edu-style classifier filtering, MinHash + suffix-array dedup,
global dedup, decontamination, domain mixing. This is a proper data org's
pipeline.

**Where the frontier is — the gaps are about *what* data, not *how*:**

- **Synthetic & model-generated data at scale** (distillation from stronger
  models, self-generated reasoning traces, rephrasing, textbook-style
  generation) is now a *majority* ingredient for many frontier mixes. The
  blueprint treats synthetic data as one acquisition source among ten, not as a
  primary engine.
- **Reasoning-trace data** (long CoT, verified solutions) — needed for §1 — has
  no pipeline.
- **Multimodal data** — needed for §4 — has no pipeline.
- **Mid-training** as a named, distinct stage (high-quality/long-context/
  domain-up-weighted bridge between pretrain and post-train) is folded into a
  3-phase curriculum but not treated with the rigor the field now gives it.

**To close:** add synthetic-data generation + verification as a first-class data
subsystem; add reasoning-trace and multimodal data paths.

---

## 9. Context length targets are mid-tier 🟨 ($ + R)

**What the blueprint says** (`04`): long-context extension to "32k–128k" via RoPE
base scaling / YaRN.

**Where the frontier is.** Gemini ships **1M+ token** context; the others are
solidly in the **200k–1M** range with strong long-context retrieval. 128k is a
2024 target. Closing this interacts with §3 (MLA/sparse attention is what makes
1M context affordable) and needs long-context eval beyond perplexity
(needle-in-haystack, RULER, long-doc reasoning).

---

## 10. Eval & safety predate the agentic/reasoning era 🟨 (O + D)

**Eval** (`08`): MMLU/GPQA/BBH/MATH/GSM8K/HumanEval/MBPP/IFEval/MT-Bench — a
strong **2024** suite. Missing the benchmarks the frontier is actually judged on
in 2026: SWE-bench Verified, ARC-AGI-2, Humanity's Last Exam, frontier-math,
long-horizon agentic task suites, multimodal (MMMU/video), and **reasoning-cost
curves** (score vs. test-time tokens). GSM8K/MMLU are now near-saturated and
contamination-prone.

> **Update:** the *simulator's* eval now reports the 2026 frontier suite —
> `predict_swebench` (agentic-post-training-driven), `predict_arc_agi2` and
> `predict_hle` (reasoning-driven), and `predict_mmmu` (at chance until
> multimodal) in `platform/sim/scaling.py`, surfaced through `simulate_eval`.
> Unlike the 2024 scores these are **lifted by post-training** (`reasoning_quality`
> / `agentic_quality`) and by modality, so they move when you add RLVR, agentic
> RL, or a vision tower — exactly the levers the static 2024 suite couldn't see.
> The static benchmark *harness* in `08-evaluation.md` still needs the real
> datasets wired in.

**Safety** (`09`): genuinely forward-looking on *categories* (CBRN, cyber,
autonomy, RSP preflight gate). The gap is that the **eval harness and RL loop
don't yet test the reasoning/agentic surfaces** those policies care about
(scheming/deception evals, long-horizon autonomy with real tools, reasoning-trace
monitoring). The policy hooks exist; the measurement machinery for 2026-era risks
doesn't.

---

## 11. The simulator encodes 2024 economics 🟨 ($) — *now largely addressed*

`scripts/simulate.py` and the headline table were internally consistent and a nice
artifact — but they modeled **dense** models at **bf16 spec-sheet MFU** and predicted
eval scores from **pretraining scaling laws only**. They therefore:

- overstated cost vs. an MoE+FP8 frontier recipe (active-param FLOPs + FP8 would
  cut the modeled $ substantially), and
- couldn't represent the **post-training compute** that now rivals or exceeds
  pretraining for reasoning models — there was no RLVR compute line, and eval
  scores were throughput-independent pretraining extrapolations that wouldn't
  move even if you added reasoning RL.

**To close:** add MoE active-param + FP8 to the throughput model; add an RLVR /
post-training compute stage; let reasoning compute affect predicted eval scores.

> **Update — done.** All three are now implemented in `platform/sim/`:
> - **MoE active-param FLOPs** via `moe_active_params()` + `--moe-experts`.
> - **FP8/NVFP4 throughput** via `precision_speedup()` + `--precision`.
> - **A real RLVR/GRPO compute stage** — `reasoning_rl_sim.py` prices rollout +
>   update GPU compute, sandboxed-verifier CPU, and cold-start labels, and its
>   `reasoning_quality` multiplier *moves the predicted eval scores* (GSM8K and
>   arena ELO) the way o1/R1 post-training does. New `1t`/`2t` presets and
>   `GB200`/`B300` GPU rows let you price runs on hardware we don't own.
>
> See `docs/13-simulation.md` §3.2/§3.12 and the frontier recipe in §10. What
> remains is calibration polish, not a missing capability.

---

## Priority ordering (if the cluster arrived tomorrow)

Status legend: ✅ implemented (toy-functional code + sim), 🟡 partially (code or
sim, not both), ⬜ open. "Toy-functional" means it runs end-to-end on CPU and is
priced in the simulator, *not* that it is production-scale.

| # | Gap | Tag | Status | Notes |
|---|-----|-----|--------|-------|
| 1 | RLVR + GRPO + reasoning post-training | 🟥 R/D | ✅ | `platform/rl/`: GRPO + cold-start + reward shaping; sim phase |
| 2 | Multimodality | 🟥 $/R/D | 🟡 | VLM adapter + sim pricing + MMMU; data/tokenizer/serving open |
| 3 | MoE as default (fine-grained + shared, aux-free) | 🟥 $/R | ✅ | `MoEFFN` + `model_moe_1t.yaml` + active-param sim |
| 4 | Agentic / tool-use training + eval | 🟧 R/D | 🟡 | `rl/agentic.py` env + sim + SWE-bench predictor; real tools open |
| 5 | MLA / KV compression + 200k–1M context | 🟧 $/R | 🟡 | `MLAttention` + serving sim; sparse-attn + 1M context open |
| 6 | Synthetic + reasoning-trace data engine | 🟧 D/R | ⬜ | not started |
| 7 | FP8/NVFP4 first-class + re-baselined sim | 🟨 $ | 🟡 | sim prices it; training numerics not implemented |
| 8 | Muon / MTP / QK-norm (adopt from our own repos) | 🟨 $/R | 🟡 | QK-norm done; Muon/MTP/WSD open |
| 9 | 2026 eval + safety surfaces | 🟨 O/D | 🟡 | sim eval (SWE-bench/ARC-AGI-2/HLE/MMMU); real harness + safety open |

**Bottom line.** The blueprint nails the *systems-engineering* half of a frontier
program — data hygiene, distributed training, checkpointing/stability, infra
reliability math, serving tiers, RSP gating. Those are real and hard and mostly
**right**. The *2025–2026 capability stack* that was missing is now present in
**toy-functional + simulated** form: sparse MoE (fine-grained + shared, aux-free),
MLA KV compression, RLVR/GRPO reasoning post-training with cold-start and reward
shaping, agentic tool-use RL, multimodal understanding, and a post-training-aware
2026 eval suite — all runnable on CPU and priced by the simulator on hardware we
don't own (GB200/B300). What remains is the **production-scale** half of each:
async vLLM/SGLang rollout, sandboxed code/browser tools, pretrained vision towers,
FP8 numerics, expert-parallel MoE dispatch, the synthetic/reasoning-trace data
engine, and wiring real benchmark datasets into the eval harness. Those need
**research bets, a data/labeling org, and real GPUs** — which is why the flagship
labs still spend as much on *people and data* as on *compute*.

---

## Sources

- Sebastian Raschka, *The State of LLMs 2025: Progress, Problems, and Predictions*
  (Dec 2025) — "the year of reasoning, RLVR, and GRPO"; mid-training; PRM status.
  <https://magazine.sebastianraschka.com/p/state-of-llms-2025>
- *A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason,
  and Agentic Systems*; arXiv 2504.09037.
- DeepSeek-AI, *DeepSeek-R1* (RLVR, emergent reasoning); arXiv 2501.12948.
- Shao et al., *DeepSeekMath* (GRPO); arXiv 2402.03300.
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (671B/37B MoE, MLA, MTP, FP8
  training, aux-loss-free balancing).
- DeepSeek-AI, *DeepSeek-V3.2* (sparse attention) — see our
  `docs/2026-05-sota-llm-agi.md` and Raschka's "V3 to V3.2" coverage.
- Frontier-model comparisons (Gemini 3 Pro / Claude Opus 4.5 / GPT-5.x), Nov–Dec
  2025: SWE-bench Verified, ARC-AGI-2, Humanity's Last Exam, long context,
  multimodal — Artificial Analysis & Vellum flagship reports.
- Internal: `docs/2026-05-sota-llm-agi.md` (our own SOTA Watch already lists MLA,
  MoE, FP8/NVFP4, RLVR/GRPO, MTP as tracked techniques).

> **Methodology note.** Exact recipes for GPT‑5.x, Opus 4.x, and Gemini 3.x are
> not public. Gaps are stated against the *publicly documented direction of the
> frontier* (DeepSeek's open reports are the best-documented proxy) and against
> 2025–2026 technique literature, not against leaked specs. Where the blueprint
> is genuinely strong (data hygiene, infra, distributed systems, serving, RSP),
> this analysis says so.
