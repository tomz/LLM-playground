# 17 — Frontier Model Gap Research

## Short verdict

Even with enough GPUs, `frontier-platform` is **not blocked primarily by pretraining compute anymore**. The repo's latest design has caught up on many 2025–2026 architectural ideas — sparse MoE, MLA-style KV compression, GRPO/RLVR skeletons, Muon/MTP/QK-norm, multimodal adapter, simulator pricing for FP8/NVFP4, agentic RL toy harness.

But to produce something competitive with the leading closed frontier tier — “GPT-5.5 / Claude Opus 4.8 / Gemini 3.5”-class in the user's phrasing — the remaining gaps are mostly:

1. **Data and post-training organization**, not just GPUs.
2. **Production-scale RLVR / agentic rollout infrastructure**, not just toy loops.
3. **Native multimodality**, especially video/audio/document/web/computer-use.
4. **Real distributed MoE + FP8/NVFP4 training backends**, not correctness references.
5. **Real eval/safety harnesses for reasoning, agency, deception, autonomy, multimodal tasks.**
6. **Synthetic/reasoning-trace data factory** — probably the largest open gap.

> **Methodology note.** No reliable public technical documentation was found for the exact named versions “GPT-5.5”, “Claude Opus 4.8”, or “Gemini 3.5”. Exact closed-model recipes are not public. This comparison is against the public direction of frontier systems, using open technical proxies such as **DeepSeek-V3**, **DeepSeek-R1**, **GRPO / DeepSeekMath**, and current reasoning-agent survey literature.

---

## What the current `frontier-platform` already gets right

Compared to the original 2024-style “dense decoder + SFT/RLHF” blueprint, the repo is now much closer to a modern frontier architecture:

| Area | Current repo status | Frontier relevance |
|---|---:|---|
| Sparse MoE | ✅ Implemented conceptually: fine-grained experts, shared experts, aux-free balancing, active-param accounting | Matches the public DeepSeek-V3 direction: 671B total / 37B active, MoE economics |
| MLA / KV compression | ✅ Model + KV-cache path exist | Matches long-context / inference-cost frontier direction |
| RLVR / GRPO | ✅ Toy-functional code + async rollout skeleton + sandboxed verifier | Matches DeepSeek-R1 / reasoning-model direction |
| Muon / MTP / QK-norm | ✅ Present per gap doc status | Matches modern training-efficiency/stability stack |
| FP8/NVFP4 economics | 🟡 Simulated; production numerics not fully real | Matches DeepSeek-V3-style FP8 economics, but not yet operational |
| Agentic RL | 🟡 Toy-functional tool env + sim pricing | Good conceptual start; not enough for real SWE/computer-use agents |
| Multimodality | 🟡 LLaVA/SigLIP-style adapter path | Useful baseline; not native frontier multimodality |
| Eval simulator | 🟡 Has modern score predictors | Needs real benchmark harnesses/datasets |

So this is no longer merely a “2024 dense LLM training platform.” It is better described as:

> **A CPU-runnable, GPU-ready blueprint for a frontier-style text/reasoning model, with several major production and data gaps before it could plausibly compete with closed frontier systems.**

---

## Main gaps vs. leading frontier models

### 1. Synthetic + reasoning-trace data engine is the biggest missing subsystem

This is the most important non-GPU gap.

The docs already have a strong classical data pipeline: crawl, extract, filter, dedup, decontaminate, mix, shard. That is necessary but not sufficient for 2026-class systems.

Modern frontier systems appear to depend heavily on:

- large-scale **synthetic data generation**,
- distillation from stronger teacher models,
- self-improvement loops,
- reasoning traces,
- code/math/formal verifier-generated data,
- multimodal instruction data,
- agentic trajectories,
- preference and safety data refreshed continuously.

`frontier-platform` currently treats synthetic data as a source, not as a **central production machine**.

**What is missing:**

- `platform/data/synthetic/` style subsystem:
  - teacher orchestration,
  - generation policies,
  - rejection sampling,
  - verifier-based filtering,
  - contamination controls,
  - diversity/coverage tracking,
  - data lineage and licensing.
- Reasoning-trace data:
  - long-CoT,
  - verifier-labeled,
  - difficulty-curriculum controlled,
  - decontaminated against math/code benchmarks.
- Agentic trajectory data:
  - multi-turn browser/code/tool traces,
  - successful and failed plans,
  - environment-state logging,
  - sparse terminal reward labels.
- Multimodal data:
  - interleaved image-text,
  - OCR/doc/chart datasets,
  - video/audio pipelines,
  - perceptual dedup,
  - modality-specific safety filtering.

**Why GPUs do not solve this:**  
A giant cluster can train on data. It cannot invent the right data distribution, verifier coverage, task curricula, contamination controls, or labeling organization.

**Severity:** 🟥 capability-defining.

---

### 2. RLVR exists, but not yet at frontier production scale

The repo's `docs/15-reasoning-rl-rlvr.md` and `platform/rl/` direction are correct: GRPO, verifier rewards, code/math verifiers, cold-start SFT, reward shaping, async rollout.

That matches the public DeepSeek-R1 / GRPO direction. DeepSeek-R1's public abstract explicitly frames reasoning as emerging from reinforcement learning on verifiable tasks, including self-reflection, verification, and dynamic strategy adaptation. DeepSeekMath introduced GRPO as a PPO variant that improves reasoning while reducing PPO memory overhead.

But the current platform is still a skeleton compared to what a frontier reasoning-training system needs.

**Missing production pieces:**

- distributed actor/learner infrastructure:
  - many inference actors,
  - learner workers,
  - rollout buffers,
  - replay / filtering queues,
  - weight sync,
  - backpressure and fault tolerance.
- high-throughput inference backend:
  - vLLM/SGLang integration,
  - speculative decoding,
  - paged KV,
  - prefix caching,
  - multi-node rollout serving.
- real verifier fleet:
  - code sandboxes with gVisor/Firecracker,
  - math symbolic checkers,
  - theorem provers,
  - database/query validators,
  - browser/task verifiers,
  - adversarial hidden tests.
- reward-hacking defenses:
  - verifier fuzzing,
  - held-out tests,
  - adversarial graders,
  - exploit detection.
- long-CoT training controls:
  - budgeted reasoning,
  - length penalties,
  - trace compression,
  - hidden vs. visible reasoning policy,
  - inference-time compute curves.

**In short:** the algorithmic shape is right; the **industrial post-training plant** is not there yet.

**Severity:** 🟥 capability-defining.

---

### 3. Agentic capability is still a toy harness, not a full training environment

Leading frontier models are increasingly judged as **agents**, not just chat models:

- SWE-bench-style software engineering,
- computer/browser use,
- deep research workflows,
- tool orchestration,
- long-horizon planning,
- file editing,
- command execution,
- multi-step recovery from errors.

Your repo has a good start: `ToolEnv`, tool-call JSON, terminal sparse rewards, and simulator hooks. But that is not enough to train or evaluate frontier-grade agents.

**Missing pieces:**

- real browser environments,
- real coding sandboxes,
- realistic repo-editing tasks,
- package installation/build/test loops,
- stateful terminal sessions,
- long-horizon memory,
- curriculum over task length/difficulty,
- trajectory mining from human/AI workflows,
- failure analysis and retry policies,
- agentic safety evals:
  - autonomy,
  - deception,
  - cyber misuse,
  - exfiltration attempts,
  - persistence,
  - unauthorized tool use.

The leading closed models likely benefit from massive internal traces: code-assistant sessions, browser/tool interactions, user feedback, eval-driven self-play, and specialized task environments. `frontier-platform` does not yet have a comparable data or environment factory.

**Severity:** 🟥/🟧 depending on whether the target is “best chat model” or “best agent.”

---

### 4. Multimodality is adapter-level, not native frontier multimodality

`docs/16-multimodality.md` correctly says the current implementation is MM-1: image-understanding adapter with a vision encoder and projector. That is a useful baseline, but leading frontier systems are not merely “text LLM + image adapter.”

A GPT-5.x / Gemini-3.x-class model is expected to be deeply multimodal:

- text,
- images,
- documents,
- charts,
- screenshots,
- audio,
- video,
- possibly computer-use state.

Gemini especially has historically differentiated on long context and multimodal/video capabilities. A late-fusion image adapter is unlikely to match native systems trained on interleaved multimodal corpora at scale.

**Missing pieces:**

- native multimodal tokenizer/sequence contract,
- image/video/audio tokenization strategy,
- variable-resolution image tiling,
- OCR/document layout modeling,
- chart/table reasoning,
- video temporal modeling,
- audio input/output,
- multimodal SFT/RL data,
- multimodal eval harness:
  - MMMU,
  - MathVista,
  - ChartQA,
  - DocVQA,
  - RealWorldQA,
  - video QA,
  - screen/browser tasks.
- serving support:
  - image preprocessing,
  - batching variable image/video inputs,
  - modality-aware safety filters,
  - multimodal KV/cache economics.

**Bottom line:** current repo can produce a multimodal demo; it cannot yet produce a native multimodal frontier model.

**Severity:** 🟥.

---

### 5. Production MoE is not the same as MoE correctness code

The architecture doc is now aligned with the public DeepSeek-V3 direction:

- sparse MoE,
- fine-grained experts,
- shared experts,
- aux-loss-free balancing,
- active vs. total parameter accounting.

DeepSeek-V3's public abstract describes a 671B-parameter MoE with 37B active parameters per token, MLA, auxiliary-loss-free load balancing, MTP, 14.8T pretraining tokens, SFT, and RL. So your design is directionally right.

But to train such a model on 10k–100k GPUs, the missing issue is **distributed systems reality**:

- expert parallel dispatch,
- all-to-all routing,
- load balancing under real token distributions,
- communication overlap,
- straggler mitigation,
- token dropping/capacity policies,
- expert placement,
- pipeline + tensor + data + expert parallel composition,
- checkpointing expert shards,
- recovering from expert imbalance or collapse,
- inference-time expert routing and caching.

The repo currently has a correctness-oriented model path. That is not yet a Megatron/DeepSpeed/DeepSeek-scale MoE runtime.

**Severity:** 🟧 significant.  
**Why not 🟥?** Because this is mostly engineering if you have the right team and cluster, not an unknown research problem.

---

### 6. FP8/NVFP4 is simulated, but real numerics are hard

The simulator prices FP8/NVFP4 speedups. That is good. But “set precision='fp8'” is not equivalent to a stable frontier-scale FP8 run.

Public DeepSeek-V3 reporting emphasizes full FP8 mixed-precision training at scale. That kind of run needs:

- per-tensor/per-block/per-tile scaling,
- accumulation policy,
- selective high-precision paths,
- optimizer state precision design,
- loss-scaling and overflow monitoring,
- attention/MLP kernel support,
- communication precision policy,
- checkpoint conversion,
- numerics regression tests,
- hardware-specific tuning for H100/H200/B200/GB200/B300.

`frontier-platform` has a policy abstraction and simulator economics, but not a validated FP8 training recipe.

**Severity:** 🟧.

---

### 7. Long context target still needs sparse attention + eval discipline

The repo has MLA and KV-cache compression, which is a major improvement. But leading systems now compete on:

- 200k–1M+ context,
- long-document reasoning,
- retrieval across huge contexts,
- multi-file codebase comprehension,
- video/document context,
- long-horizon agent memory.

MLA helps KV-cache memory, but 1M context usually also needs:

- sparse/sliding/global attention patterns,
- chunked prefill,
- retrieval-augmented memory,
- hierarchical attention,
- long-context curriculum,
- position/interpolation stability,
- long-context evals beyond “needle in haystack.”

**Missing evals:**

- RULER-style tasks,
- long-document QA,
- multi-hop across many files,
- long-context codebase modification,
- long video/document tasks,
- score-vs-context-length curves.

**Severity:** 🟧.

---

### 8. Evaluation harness is still mostly simulated

The simulator predicts modern metrics, but a frontier program needs real, continuously running eval infrastructure.

The old static suite — MMLU, GSM8K, HumanEval, BBH — is not enough. Many are saturated or contamination-prone.

A modern eval platform needs:

- SWE-bench Verified,
- LiveCodeBench,
- Terminal-Bench / OSWorld-style agent tasks,
- GAIA / deep research tasks,
- GPQA Diamond,
- Humanity's Last Exam,
- ARC-AGI-style tasks,
- FrontierMath / hard math,
- MMMU / MathVista / DocVQA / ChartQA,
- long-context reasoning,
- safety/autonomy/cyber/CBRN evals,
- evals under different inference budgets,
- pass@k / majority vote / best-of-n,
- cost-quality curves.

Most importantly, evals need to be wired into training decisions:

- checkpoint gates,
- regression dashboards,
- contamination checks,
- automated red-team reports,
- per-domain failure analysis,
- canary tasks,
- model behavior diffs.

**Severity:** 🟧/🟥.  
For a demo model, simulated eval is fine. For a frontier model, real eval is existential.

---

### 9. Safety architecture is policy-aware but not yet reasoning/agentic enough

The safety docs include the right categories: CBRN, cyber, persuasion, autonomy, red-teaming, gates. But frontier risk now lives especially in:

- long-horizon agency,
- tool use,
- cyber automation,
- self-exfiltration,
- deception/scheming,
- hidden reasoning traces,
- model organisms of misalignment,
- multimodal abuse,
- autonomous replication attempts,
- prompt-injection against tool-using agents.

A frontier-grade platform needs safety evals that run in the same kind of environments used for agentic training.

**Missing pieces:**

- tool-use safety sandbox,
- cyber range evals,
- autonomy eval harness,
- dangerous capability scoring,
- chain-of-thought / reasoning-trace monitoring policy,
- jailbreak suites for multimodal and agentic contexts,
- scalable red-team data ingestion,
- deployment gating tied to eval thresholds.

**Severity:** 🟧.

---

### 10. Serving is not yet a frontier product system

The serving design has the right concepts:

- quantization,
- paged KV,
- speculative decoding,
- continuous batching,
- tensor-parallel inference,
- autoscaling.

But leading frontier systems are product platforms, not just inference engines.

Missing or underdeveloped:

- production vLLM/SGLang/Triton backend integration,
- multimodal serving path,
- tool-call runtime,
- browser/computer-use runtime,
- reasoning-budget API,
- hidden/visible thought policy,
- safety filters in the serving path,
- prompt-injection defenses for tools/RAG,
- per-user memory/personalization controls,
- latency/cost routing across model sizes,
- distillation cascade:
  - small fast model,
  - medium model,
  - large reasoning model,
  - specialist tool models.
- observability:
  - token latency,
  - cache hit rate,
  - tool failures,
  - eval drift,
  - safety incidents,
  - cost attribution.

**Severity:** 🟧.

---

## “If the GPUs arrived tomorrow” priority plan

If the goal is to actually produce a frontier-class run, I would not start by launching the biggest pretraining job. I would sequence it like this:

### Phase 1 — Productionize the training substrate

Before spending $100M+ on tokens:

1. Real distributed MoE runtime:
   - expert parallelism,
   - tensor/pipeline/data parallel composition,
   - reshardable checkpoints.
2. Real FP8/BF16/NVFP4 numerics path:
   - Transformer Engine or equivalent,
   - precision audits,
   - stability tests.
3. Real data loader at trillion-token scale:
   - resumable,
   - deterministic,
   - multi-region/object-store friendly.
4. Failure/restart drills:
   - node failures,
   - checkpoint restore,
   - loss spike rewind,
   - expert imbalance recovery.

### Phase 2 — Build the data factory

This is the likely make-or-break phase.

1. Synthetic data generation system.
2. Reasoning-verifier data system.
3. Multimodal data pipeline.
4. Agentic trajectory capture/generation.
5. Contamination and lineage database.
6. Eval-linked data curriculum.

### Phase 3 — Train base and midtrain models

1. Start with a 30B–70B active MoE-scale model, not maximum size.
2. Validate scaling laws.
3. Run ablations:
   - MoE routing,
   - MLA,
   - MTP,
   - Muon,
   - FP8,
   - data mixtures.
4. Midtrain for:
   - code,
   - math,
   - long context,
   - multimodal,
   - tool-use formats.

### Phase 4 — Post-train as a first-class compute program

1. Reasoning SFT cold-start.
2. RLVR with GRPO:
   - math,
   - code,
   - formal,
   - structured tasks.
3. Agentic RL:
   - coding,
   - browser,
   - computer-use,
   - long-horizon tools.
4. Preference/safety alignment:
   - DPO/RLHF/RLAIF,
   - refusal behavior,
   - harmlessness,
   - style/persona.

### Phase 5 — Native multimodal frontier pass

The current adapter path is not enough. You need:

1. pretrained vision tower baseline,
2. interleaved image-text training,
3. document/chart/OCR data,
4. video/audio roadmap,
5. multimodal eval gates,
6. multimodal serving.

### Phase 6 — Eval, safety, and productization

1. Real benchmark harnesses.
2. Continuous model regression testing.
3. Dangerous-capability gates.
4. Agentic safety evals.
5. Reasoning-budget API.
6. Production inference stack.

---

## Gap ranking

| Rank | Gap | Severity | Why it matters |
|---:|---|---:|---|
| 1 | Synthetic + reasoning + multimodal data factory | 🟥 | GPUs without the right data produce an expensive non-frontier model |
| 2 | Production RLVR / GRPO / verifier rollout system | 🟥 | Reasoning models are post-training-compute products, not just pretrained LMs |
| 3 | Native multimodality | 🟥 | Closed frontier systems are multimodal; adapter-only vision is not enough |
| 4 | Agentic training environments | 🟥/🟧 | Frontier models are judged by long-horizon tool use and coding agents |
| 5 | Real distributed MoE runtime | 🟧 | Architecture exists; production expert parallelism does not |
| 6 | Real FP8/NVFP4 training numerics | 🟧 | Major cost/stability lever; simulation is not implementation |
| 7 | Long-context/sparse-attention stack | 🟧 | MLA helps, but 1M-context needs more |
| 8 | Real eval and safety harnesses | 🟧 | Simulated scores are not gates for a frontier program |
| 9 | Production serving/product platform | 🟧 | A frontier model must be deployable as a reasoning/multimodal/agentic system |
| 10 | Org/process maturity | 🟧 | Frontier training is a company-scale operation, not a repo-scale project |

---

## Bottom line

If we had the GPUs, `frontier-platform` could plausibly become the basis for a **strong open-frontier-style text/reasoning model**, especially because it now includes the right architectural vocabulary: MoE, MLA, MTP, Muon, GRPO/RLVR, FP8 economics, agentic hooks, and multimodal adapter support.

But to compete with the leading closed frontier models, the remaining work is not “make the model bigger.” It is:

> **Turn the blueprint into a full data + post-training + multimodal + agentic + eval + safety factory.**

The largest gap is no longer model architecture. The largest gap is the **frontier production loop**:

```text
generate data → verify/filter → train → rollout → evaluate → red-team
→ mine failures → synthesize harder data → retrain/post-train → deploy
```

That loop, at scale and across reasoning, multimodality, and agency, is what separates a GPU-rich training run from a true frontier model program.

---

## Sources and public technical proxies

- DeepSeek-AI, *DeepSeek-V3 Technical Report* — sparse MoE, MLA, aux-loss-free load balancing, MTP, FP8, 14.8T tokens, SFT/RL.
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — RL-induced reasoning, self-reflection, verification, dynamic strategy adaptation.
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* — GRPO and math-data pipeline.
- Ke et al., *A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems* — reasoning, inference scaling, learning-to-reason, agentic workflows, PPO/GRPO, verifiers.
- Internal docs: `docs/14-gap-analysis-vs-frontier.md`, `docs/15-reasoning-rl-rlvr.md`, `docs/16-multimodality.md`.
