# Deep Dive: *MAI-Thinking-1: Building a Hill-Climbing Machine*

**Source:** The Microsoft AI Team · ~109 pages · `microsoft_paper.pdf`
(`https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf`)
**Type:** Company technical report (not peer-reviewed)
**One-line:** A 35B-active / 1T-total MoE frontier **reasoning** model trained
**from scratch** (no third-party CoT distillation), built via iterative RL
"hill-climbing."

---

## The one-paragraph version

MAI-Thinking-1 is Microsoft AI's first in-house frontier **reasoning** model: a
**35B-active / 1T-total Mixture-of-Experts** trained from scratch (no
distillation from third-party chains-of-thought), reaching the frontier on
agentic coding and competition math — **52.8% SWE-Bench Pro, 97.0% AIME 2025,
87.7% LiveCodeBench v6** — and trading blows with Claude Sonnet/Opus 4.6,
GPT-5.4, Gemini, DeepSeek V3.2/V4, and Kimi K2.6. The paper's framing metaphor
is **"hill-climbing"**: rather than one monolithic training run, they build a
base model and then iteratively *climb* via reinforcement learning, training
domain specialists, distilling them back together, and repeating. It's unusually
candid for a frontier lab report — full of the actual cutoffs, token counts,
failure modes, and infrastructure plumbing that most labs omit.

---

## 1. The architecture & the "model ladder"

- **MAI-Thinking-1** post-trained from **MAI-Base-1**: 34.7B active / 962B total
  params, **78 layers**, hidden dim 6656, MoE with **top-8 of 512 experts**,
  256K-token context window.
- Table 1 defines a **model-size ladder** (L12 → L78) so recipes can be ablated
  on small models and extrapolated — the backbone of their "de-risk cheaply,
  then scale" methodology.
- **Notable trick:** attention output is **zero-initialized** (RMSNorm gains →
  0) to prevent MoE routing imbalance early in training — a concrete, reusable
  architectural detail.

## 2. Data pipeline (the most detailed section)

Trained on **30T tokens**. The appendices are refreshingly specific:

- **Web:** A proprietary crawl of **~1.2T pages** → filtered to 794B → deduped
  to ~190B docs (73.4B English + 116.5B non-English), plus 24.2B from Common
  Crawl. They run a **proprietary AI-generated-content detector** and manually
  blocklist heavily-AI domains — an emerging concern made operational.
- **Quality filtering:** attribute models (educational value, factual accuracy,
  info density, reasoning content) → quality model that **drops the bottom 70%**
  of English docs → Gopher-style heuristics.
- **STEM/Code:** dedicated extraction (MathML/LaTeX → Markdown, LLM
  section-level **keep/remove only — no synthesis**), yielding 680B English STEM
  tokens + 233B code-web tokens.
- **PDFs:** 10B docs → 620M, OCR'd via Azure Document Intelligence → **1.8T
  English + 1.85T multilingual tokens**.
- **GitHub:** 7.4T-token corpus split into **files / commits / PRs** (1.26T /
  4.5T / 1.19T). Decontaminated against SWE-bench Verified.
- **Dedup stack:** exact (MD5/SHA-512) → fuzzy (MinHash LSH @0.8) → **semantic**
  (cosine over Qwen3-Embedding-0.6B).
- **Cutoffs (Table 4):** Web HTML Sept 2025, PDFs Dec 2025, GitHub June 2025,
  Books/journals March 2026.

## 3. Training & precision

- Global batch **134M tokens**; staged LR (cosine peak 2e-5 → constant 1e-6).
- **Mixed precision:** BF16 default, **FP8 E4M3** forward GEMMs, **FP8 E5M2**
  data-gradient, FP32 for sensitive ops.
- **Honest failure reporting:** early loss spikes (coding data + expert
  imbalance) that **self-recovered without intervention** — rarely disclosed by
  other labs.

## 4. Long-context extension (Appendix B — a clean, reusable recipe)

Headline practical lesson: **don't pay for expensive long-context training
throughout.** Pre-train at 16K, mid-train at 64K, then a short, cheap **256K
extension phase (140B tokens)**. Key findings:

- A progressive 32K→256K checkpoint **matches** a full 1T-token 128K run on code
  NLL.
- Adaptation is **remarkably fast** — most gains land in the first 1–10% of
  extension steps, implying the model is just **recalibrating positional/
  attention behavior**, not learning new capabilities.
- They explicitly note this **extends to 1M+ tokens** at modest cost.

## 5. The "hill-climb" — RL recipe (the conceptual core)

The signature methodology:

1. Start from MAI-Base-1.
2. Train **three specialists** via RL: **SWE/Agentic**, **STEM**, **Helpfulness
   & Safety**.
3. **Distill** specialists back into one consolidated model via SFT.
4. Final **RL climb** → MAI-Thinking-1.

Technical ingredients:

- Objective derived from **GRPO** with token-level policy gradient.
- **Adaptive entropy control** — an integral controller targeting a setpoint H*
  to prevent entropy collapse.
- Outer ratio clip to curb gradient spikes.
- Reward = `R_task + w_lang·R_lang − w_len·R_len` (language consistency favoring
  English; **difficulty-aware** length penalty).
- Infra: **YOLO** ("You Only Launch Once") in-house PyTorch distributed
  framework, and **SEE** (Sandbox Execution Environment) for agentic RL.

## 6. Most interesting qualitative section: Evolution of Reasoning Traces (Appendix C)

Because they climb **from scratch**, they get a clean view of how reasoning
emerges. The observed archetypes are genuinely insightful:

- **"Weak models guess, strong models work hard."** On an AIME problem, the weak
  checkpoint fabricates candidate roots and gets 704; the strong checkpoint
  derives all four algebraic candidates, **filters by the domain condition**, and
  gets the correct 240.
- **"Weak models brute force, strong models find invariants"** — strong traces
  identify group-theoretic structure (index-3 subgroups mod powers of 3) instead
  of grinding.
- **"Strong models are skeptics"** — they pause ("*Wait, let's re-examine*") and
  **test their own converse on a small case**.
- **Agentic:** strong checkpoints **write and run unit tests**, do "evidence
  archaeology" (read the repo before patching), and seek the **source of truth**;
  weak ones fixate on edit mechanics and speculate on adjacent code paths.

This is the paper's most transferable insight: these behaviors **emerge from
RL**, they weren't distilled in.

## 7. Infrastructure (Appendices F & K — frontier-scale ops)

- **SWE environment build:** 102M GitHub PRs → 745K passing grading extraction,
  on a **two-pool Ray cluster across ~30,000 CPU cores**, rootless podman +
  BuildKit sidecars sharing NVMe. Bottleneck is **LLM token consumption** (12M
  tokens/min, 83% cache hit) → ~20 graded environments/minute.
- **Training cluster:** heterogeneous **H100 / GB200 / GB300**; main
  pre-training on a single **GB200 NVL72** cluster (72-GPU NVLink domains,
  InfiniBand scale-out).
- **Reliability philosophy:** "a node is useful only when healthy, topologically
  valid, observable, and recoverable." Hierarchical **certification**
  (single-node → rack collectives → cross-rack InfiniBand) caught **multiple
  racks with <16 healthy nodes**. Observability is **part of the control loop** —
  telemetry directly drives admit/drain/remediate decisions, managing the fleet
  by **goodput, not provisioned GPUs**.

## 8. Evaluations & safety

- **STEM (Table 17):** AIME 2025 97.0%, AIME 2026 94.5%, HMMT Feb 2026 84.9%,
  GPQA Diamond 84.2%, LCB v6 87.7%.
- **Agentic:** SWE-bench Verified, **SWE-Bench Pro (52.8%)**, Terminal-Bench 2.0
  — simple ReAct loop, 256K context, just **bash + string-replace** tools.
- **Safety (Appendix I):** a two-stage LLM-judge pipeline (classify request along
  5 dimensions → generate response spec), with **jailbreak unwrapping** so scores
  reflect true intent. Release gating uses a **Pareto frontier** over safety/
  over-refusal/quality, with thresholds at a **fixed percentile of what's
  currently achievable** rather than a static number. Parity with Sonnet 4.6 on
  AIR-Bench; wins on CyberSecEval Autocomplete.
- **Human eval:** 1,276 English tasks (30% multi-turn) from expert prompts +
  filtered Copilot logs.

---

## Why this paper matters

1. **From-scratch frontier reasoning is reproducible-in-principle** — they show
   you don't need to distill from GPT/Claude to reach the frontier.
2. **The "hill-climbing" recipe** (specialists → distill → climb) is a clean,
   named alternative to monolithic training.
3. **Unusual transparency** — real cutoffs, token counts, self-recovering loss
   spikes, dead-end ablations (long-context data mixes that *didn't* help, MRCR
   dropped for overfitting).
4. **The reasoning-trace taxonomy** is the most quotable contribution — concrete
   evidence of *what* improves when a model "gets smarter."

## Caveats

- It's a **company technical report, not peer-reviewed** — competitor numbers are
  pulled from official cards under their own harness, so cross-model claims
  deserve mild skepticism.
- Many of the most interesting components (**YOLO, SEE, proprietary crawl,
  AI-content detector**) are described but **not released**.
- Dates (2026 benchmarks, March 2026 cutoffs, CreationDate Jun 2026) place this
  as a **near-future / forward-dated** document.

---

## Harvest notes for LLM-playground

Threads from this paper that map onto our projects:

- **`frontier-platform`** — the hill-climb recipe (specialist → distill → climb),
  GRPO + adaptive entropy control, and the staged long-context extension recipe
  are all directly relevant to the RLVR and long-context gaps tracked in
  `frontier-platform/docs/17b-frontier-model-gap-research-v3.md`.
- **Long-context extension** (Appendix B) — "mid-train short, extend at the end"
  is a cheap, concrete recipe worth a standalone engineering note.
- **Reasoning-trace archetypes** (Appendix C) — could be distilled into a reusable
  eval rubric for measuring qualitative reasoning improvement.

## Harvest log — what shipped into `frontier-platform`

A first wave of paper→repo harvests, scoped to changes that are pure-Python,
CPU-testable, and slot into an existing protocol (no cluster / FP8 / crawl
infra). Each lands against a "content-open" gap from
`docs/17b-frontier-model-gap-research-v3.md` (gap #3 RLVR verifier coverage,
gap #6/§2.11 RL stability). **Tier 1 (RLVR recipe) — shipped:**

| # | Harvest (paper §) | Where | Status |
|---|---|---|---|
| 1 | IFEval-style **objective constraint verifiers** (§8 IFEval table) | `rl/verifiers.py`: `ConstraintFollowingVerifier`, `check_constraint`, 16-checker `CONSTRAINT_CHECKERS` registry, `make_verifier("constraints", ...)` | ✅ shipped |
| 2 | **Adaptive entropy control** (§5) — PI controller targeting setpoint H* to avert entropy collapse | `rl/grpo.py`: `EntropyController` (PI + anti-windup), `GRPOConfig.target_entropy/entropy_*`, wired through `grpo_step` + `run_grpo`/`run_grpo_async`; `_common.compute_token_logps_and_entropy` | ✅ shipped |
| 3 | **Language-consistency reward** (§5, `R_lang`) — anti language-mixing | `rl/reward.py`: `language_consistency_reward` (reuses `data/filter.detect_language`), `RewardConfig.language_weight/target_lang`, into `CompositeReward.breakdown` | ✅ shipped |
| 4 | **Difficulty-aware length penalty** (§5, `R_len`) — longer CoT budget for harder problems | `rl/reward.py`: `RewardConfig.difficulty_aware/length_target_{easy,hard}` + `length_target_for()`, `CompositeReward.difficulty_fn` | ✅ shipped |
| 5 | **Outer ratio clip** (§5) — dual-clip-PPO floor on negative-advantage tokens | `rl/grpo.py`: `_dual_clip_surrogate`, `GRPOConfig.clip_ratio_c` | ✅ shipped |

Tests: `tests/test_rl_tier1.py` (22) + `tests/test_reward_shaping_tier1.py` (14)
= **36 new, all passing**; full suite green except 4 pre-existing
`test_jail.py` failures that are an environment artifact (bubblewrap cannot
`execvp` a uv-managed Python symlink — unrelated to these changes).

**Tier 2 (eval + safety harness) — shipped:**

| # | Harvest (paper §) | Where | Status |
|---|---|---|---|
| 6 | **Reasoning-trace archetype rubric** (§6/Appendix C) — deterministic detectors for the strong-reasoning archetypes (backtracking, verification, case-analysis/filter, invariant-seeking, self-skepticism, enumerate-then-filter, + agentic unit-testing / evidence-first) | `eval/reasoning_rubric.py`: `ReasoningRubric`, `ReasoningSignal`, `RubricResult`, `TraceJudge` protocol; strong>weak on the paper's own AIME exemplar | ✅ shipped |
| 7 | **Long-context eval adapters** (Appendix B) — Code-NLL (position-bucketed), Retrieval-NLL (needle by LM loss), answer-accuracy-by-depth | `eval/long_context.py`: `CodeNLLAdapter` / `RetrievalNLLAdapter` / `LongContextQAAdapter` (BenchmarkAdapter shape) + `make_needle_record`; wired into `Evaluator.run_long_context` | ✅ shipped |
| 8 | **Pareto-percentile release gate** (§8/Appendix I) — thresholds at a fixed percentile of the *currently-achievable* fleet, on a Pareto frontier over (safety, over-refusal, quality) | `safety/gates.py`: `ReleaseMetrics`, `ParetoGateConfig`, `pareto_preflight`, `pareto_frontier`, `percentile_threshold` (additive; original `preflight` untouched) | ✅ shipped |

Tests: `tests/test_tier2_harvest.py` (**25**, all passing).

**Tier 3 (recipe + architecture) — shipped:**

| # | Harvest (paper §) | Where | Status |
|---|---|---|---|
| 9 | **Hill-climb orchestrator** (§5, the conceptual core) — specialists → distill → climb, with rejection-sampled distillation harvest + lineage | `rl/hillclimb.py`: `Specialist`, `HillClimbConfig`, `run_hill_climb` (ties `run_grpo` + `sample_group` rejection-sampling + `run_coldstart`); CPU end-to-end | ✅ shipped |
| 10 | **Zero-init attention output** (§1) — block starts as identity so early attention noise can't perturb MoE routing | `model/config.py` `zero_init_attn_output` flag + `Transformer.init_weights` zeroes `attn.o_proj` (exact residual analogue of a zero post-attn-norm gain) | ✅ shipped |

Tests: `tests/test_tier3_harvest.py` (**11**, all passing).

**Wave totals.** 72 new tests; full suite **409 passing** (4 pre-existing
`test_jail.py` env failures excluded). All ten harvest items from the deep dive
are now in the repo behind their existing protocols (Verifier / RewardConfig /
BenchmarkAdapter / gate / Transformer init), so a research or data org can fill
content without touching platform code.

**Cross-pollination.** Harvest item #10 (zero-init residual projections) also
landed in **`midgpt`** (`model.GPTConfig.zero_init_proj`, default-off, 4 tests →
87 total) to bring it to parity with `nanogpt-edu` and `distgpt`, which already
shipped the same `zero_init_proj` knob independently. The other nine items are
frontier-platform-specific (GRPO learner, verifier/reward protocols, eval
adapters, RSP gate) and don't have a single-node analogue.

## Sources

1. *MAI-Thinking-1: Building a Hill-Climbing Machine*, The Microsoft AI Team —
   `https://microsoft.ai/wp-content/uploads/2026/06/main_20260602_2.pdf`
   (local copy: `microsoft_paper.pdf`).
