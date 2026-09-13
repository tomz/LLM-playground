# SOTA Watch — LLM & AGI · 2026-07

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-07  ·  **Status:** published

> July moved the frontier from better standalone models toward **more efficient,
> persistent, and auditable agent systems**. OpenAI's GPT‑5.6 family pushed the
> quality-per-token and quality-per-dollar frontier, added programmatic tool
> calling, and exposed parallel multi-agent execution as an explicit high-compute
> mode. Two late-month studies showed why the surrounding harness now matters as
> much as the checkpoint: retaining reasoning plus compacting context tripled
> GPT‑5.6 Sol's ARC‑AGI‑3 score while using one-sixth the output tokens, while an
> agent/human audit estimated that roughly 30% of SWE‑Bench Pro tasks are broken.
> GPT‑Red applied self-play to red-team robustness, and a scientific-computing
> field report located the new bottleneck in validation and stewardship rather
> than implementation. **All frontier sources in this edition are from July
> 2026.** Repository harvest remains the secondary lens.

---

## The frontier this month

- **GPT‑5.6 shifted the headline from raw intelligence to useful work per token.**
  OpenAI released Sol, Terra, and Luna on July 9, with reasoning-effort tiers and
  an `ultra` mode that coordinates four agents by default. The vendor reports
  GPT‑5.6 Sol at 53.6 on Agents' Last Exam and 80 on the Artificial Analysis
  Coding Agent Index, while using fewer tokens and less wall time than its cited
  frontier comparison. The field-level signal is that capability, latency,
  token use, and price are now one optimization surface—not separate leaderboards.
  [52]
- **Programmatic tool calling moved orchestration inside the inference loop.**
  GPT‑5.6 can write and run lightweight programs that filter tool output,
  preserve relevant intermediate state, and choose subsequent actions. This can
  reduce model round trips and context growth compared with returning every raw
  tool response to the model. It is a concrete step from fixed tool-call chains
  toward model-written control flow. [52]
- **Parallel agents became a first-class reasoning-effort setting.** GPT‑5.6
  `ultra` trades more tokens for lower time-to-result by coordinating parallel
  workstreams; OpenAI reports that four- and, on selected evaluations, sixteen-
  agent configurations improve the score/latency frontier. The important
  systems question is no longer simply “how long should one model think?” but
  “when should a controller branch, specialize, and reconcile?” [52]
- **Agent memory and context policy changed an AGI benchmark by 3×.** On the
  ARC‑AGI‑3 public set, replacing a harness that discarded private reasoning and
  used rolling truncation with retained reasoning and compaction raised GPT‑5.6
  Sol from **13.3% to 38.3% RHAE** while cutting output tokens by **6×**. This is
  unusually strong evidence that an agent evaluation measures the model *and*
  its state-management policy. [53]
- **Coding-eval quality became a frontier problem of its own.** An audit of the
  731-task SWE‑Bench Pro public split found 200 tasks (27.4%) broken through an
  agent-assisted pipeline and 249 (34.1%) through five-engineer review. Failure
  modes included overly strict tests, underspecified or misleading prompts, and
  low-coverage tests. OpenAI consequently retracted its earlier recommendation
  to adopt the benchmark. Capability claims are only as sound as the task/test
  contract beneath them. [54]
- **Agentic scientific software exposed verification as the binding constraint.**
  A field report covering eight projects found agents useful for maintenance,
  migration, optimization, and GPU-oriented redesign, but unable to reliably
  determine scientific validity. Successful teams used exact parity, known
  answers, simulated data, statistical checks, and staged changes; human work
  shifted from implementation toward specification, verification, and long-term
  stewardship. [55]
- **GPT‑Red brought self-improvement to defensive red teaming.** OpenAI's July
  15 archive describes an automated red-teaming system that uses self-play to
  improve robustness, alignment, and prompt-injection resistance. The primary
  page blocks automated retrieval, so this edition intentionally carries no
  unsupported numerical claim beyond the official archive summary. [56]

---

## TL;DR — what we harvested

- **Retained state + compaction shipped and was measured.** `midgpt/context.py`
  retains reasoning, compacts old turns, and scores fact fidelity; nanogpt adds
  an equal-budget rolling-vs-compaction A/B. On **2× RTX 5060 Ti**, compacting
  the 350M checkpoint's prefill 1,024→256 tokens cut latency **27.81→11.97 ms
  (−57.0%)**; peak VRAM stayed ~1.48 GiB because weights dominate at batch 1.
- **Benchmark QA shipped.** `coder-finetune/eval/task_contract.py` flags strict,
  underspecified, low-coverage, and misleading tasks with evidence and requires
  human adjudication before exclusion.
- **Stateful distributed rollouts gained a backend contract.** `distgpt` now has
  a lease queue with retry, stale-worker rejection, idempotent completion,
  dead-lettering, lineage, and snapshot/restore. It is a tested state machine,
  not a claim of a production Redis/etcd service.
- **Programmatic tools, multi-agent budgets, and held-out red-team gates shipped
  as frontier protocols.** Hard call/output/token budgets, deterministic
  reconciliation, curriculum deduplication, and train/held-out leakage checks
  are CPU-tested; real asynchronous fleets remain ideal-scale adapters.
- **Research search now counts all work.** Nanogpt records every failed/kept
  attempt in schema-versioned JSONL; midgpt's candidate-search benchmark includes
  failed branches and verifier time rather than pricing only the winner.

---

## Hardware envelopes per project

Hardware is a sizing note, never a blocker.

| Project | Scale | Minimal | Ideal | Unlocks at ideal |
|---------|-------|---------|-------|------------------|
| **nanogpt-edu** | 10M–100M | laptop CPU / 8 GB GPU | 1× H100 80 GB | fast context-compaction and self-play ablations |
| **midgpt** | 124M–1.5B | 1× 16 GB GPU | 8× H100 single node | long-context retained-state A/Bs, FP8/FA‑3, realistic token budgets |
| **distgpt** | 1B–70B | 1 node × 8× A100 | 8–64 nodes × 8× H100/B200 + NVLink/InfiniBand | stateful rollout inference, 3D/5D training, parallel-agent serving |
| **coder-finetune** | 0.5B–7B | 1× 8–16 GB GPU | 1–8× H100 plus isolated CPU sandboxes | repository-agent RL/eval and large task-quality audits |
| **frontier-platform** | 1B–500B+ | design docs and CPU protocol tests | heterogeneous training/serving fleet | programmatic tools, learned routing, multi-agent fan-out, automated red teams |

---

## Tier 1 — high-ROI, broadly applicable

| Technique | What it does | Win | Cost / risk | Min HW | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|--------|---------|-----------|
| **Retained reasoning state** | Preserves private reasoning or an equivalent durable state across actions | Avoids re-solving the task after every tool turn; central to the ARC‑AGI‑3 gain | State can contain sensitive data or stale assumptions; APIs differ | Any stateful agent harness | [53] | ✅ shipped (`midgpt`) | midgpt; frontier-platform |
| **Semantic context compaction** | Summarizes old actions and observations instead of dropping the oldest window | Preserves learned task state while lowering context and output cost | A lossy summary can erase constraints or evidence; requires fidelity tests | CPU-testable controller | [53] | ✅ shipped + 2-GPU cost A/B | nanogpt-edu; midgpt; frontier-platform |
| **Benchmark task-contract audit** | Checks agreement among prompt, hidden tests, reference patch, and observed failures | Prevents broken tasks from becoming false capability conclusions | Human review remains necessary; agent auditors can share model blind spots | CPU + code sandboxes | [54] | ✅ shipped | coder-finetune; frontier-platform |
| **Staged agent verification** | Breaks work into small changes with exact, statistical, or simulation-backed acceptance gates | Makes agent-generated scientific/code changes reviewable and reproducible | “Last mile” and scientific validity remain human-intensive | Any CI-capable machine | [55] | ✅ expanded | all five projects |
| **Programmatic tool calling** | Lets the model generate control flow that filters results and selects next actions | Fewer round trips and less irrelevant context on tool-heavy tasks | Generated orchestration needs sandboxing, budgets, and provenance | API or local tool-capable model | [52] | ✅ protocol / ideal backend | coder-finetune; frontier-platform |

---

## Tier 2 — scale- or hardware-gated wins

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| **Parallel multi-agent reasoning** | Fans a hard task into specialist workstreams and reconciles results | Improves score/latency frontier when branches can run concurrently | Multiplies tokens and serving concurrency; needs cancellation and duplicate-work controls | [52] | 🟡 budget/reconciliation protocol; ideal fleet | frontier-platform |
| **Capability/cost model routing** | Chooses Sol/Terra/Luna-class capability and reasoning effort per request | Better successful-work-per-dollar than one flagship default | Requires calibrated evals, prices, latency telemetry, and fallback policy | [52] | planned→ideal | frontier-platform; serving |
| **Large-scale automated red teaming** | Uses self-play to generate attacks and train/evaluate defenses | Continuously adapts robustness tests to the defended model | Must prevent train/test leakage and preserve independent held-out attacks | [56] | 🟡 curriculum + held-out gate shipped; ideal attack fleet | frontier-platform |
| **Long-lived agent memory service** | Retains, compacts, retrieves, and audits state over long trajectories | Enables coherent work beyond one context window | Privacy, poisoning, stale memory, and distributed consistency | [53] | 🟡 state/compaction + durable queue contracts | midgpt; distgpt; frontier-platform |

---

## Tier 3 — research bets (track, don't build yet)

- **Model-written orchestration as a replacement for workflow code.** It can
  reduce round trips, but generated control flow must outperform a deterministic
  planner on reliability, debuggability, and total cost before becoming default.
  [52]
- **Self-improving safety loops.** GPT‑Red makes the direction concrete, but
  defensive self-play can overfit to its own attacker distribution. Independent
  red teams and frozen held-out attack suites remain mandatory. [56]
- **Agent-assisted benchmark governance.** Agents can make exhaustive audits
  affordable, yet benchmark owners need transparent issue taxonomies, human
  adjudication, versioning, and reevaluation when tasks are removed. [54]
- **Fully autonomous scientific-software stewardship.** July's field report
  argues against this today: implementation accelerated, while scientific
  validity, ownership, maintenance, and attribution remained human obligations.
  [55]

---

## Roadmap by project

| Project | Next harvest | Hardware tier | Notes |
|---------|--------------|---------------|-------|
| **nanogpt-edu** | Drive compaction A/B with generated long trajectories | minimal | Equal-budget pure-Python A/B and full attempt/cost provenance shipped; next measure model task success |
| **midgpt** | Task-level retained-state agent evaluation | minimal→ideal | Controller + fidelity tests + 2-GPU prefill A/B shipped; next report downstream success, not latency alone |
| **distgpt** | Redis/etcd adapter for the rollout queue contract | ideal | Lease/retry/idempotency/dead-letter/snapshot state machine shipped and CPU-tested |
| **coder-finetune** | Run contract audit on a versioned real SWE task sample | minimal | Auditor + human-adjudication boundary shipped; structured proof output/checker contract also landed |
| **frontier-platform** | Async backends and independent attack/reviewer corpora | ideal | Programmatic tools, shared multi-agent budget, held-out red-team gate, and research provenance/release gate shipped as tested protocols |

---

## What shipped this month

- **`nanogpt-edu`:** `research/provenance.py` schema-versioned JSONL for every
  keep/discard/crash and complete cost aggregation; `research/context_ab.py`
  equal-budget rolling-vs-compaction comparison.
- **`midgpt`:** `context.py` retained reasoning, compaction, and fidelity;
  `research_search.py` all-branch search accounting; real 2-GPU benchmark in
  [`examples/context_compaction_2gpu.md`](../midgpt/examples/context_compaction_2gpu.md)
  (**75% fewer input tokens, 57.0% lower prefill latency**).
- **`distgpt`:** `eval/rollout_queue.py` durable-work state-machine reference:
  leases, retry, stale-worker rejection, idempotency, lineage, snapshots.
- **`coder-finetune`:** `eval/task_contract.py` conservative SWE task audit and
  `eval/structured_proof.py` theorem/schema/checker/novelty separation.
- **`frontier-platform`:** bounded programmatic tool plans, shared-budget
  multi-agent reconciliation, held-out self-play red-team gates, and complete
  research artifact/cost/reviewer release gates. Design:
  [`docs/15-july-august-2026-harvest.md`](../frontier-platform/docs/15-july-august-2026-harvest.md).
- Focused tests pass across all five projects. Full suites: nanogpt **55 pass**,
  distgpt **125 pass / 4 skip**, frontier **431 pass / 5 skip**; midgpt and
  coder-finetune pass all logic tests but each has one pre-existing torchrun
  smoke blocked by this container's hostname/rendezvous networking.

---

## Sources

Sources continue the cumulative numbering from June (1–51).

52. OpenAI, *GPT‑5.6: Frontier intelligence that scales with your ambition*,
    **9 July 2026** — Sol/Terra/Luna family; programmatic tool calling; `max` and
    parallel-agent `ultra` effort; vendor-reported agent/coding efficiency.
    <https://openai.com/index/gpt-5-6/>
53. Bigio and Sanders, OpenAI, *How enabling two settings tripled our scores on
    the ARC‑AGI‑3 benchmark*, **29 July 2026** — retained reasoning and compaction
    raised public-set RHAE 13.3%→38.3% and cut output tokens 6×.
    <https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/>
54. OpenAI, *Separating signal from noise in coding evaluations*, **8 July 2026**
    — agent-assisted and five-engineer audit of SWE‑Bench Pro; 27.4% and 34.1%
    of the public split respectively labeled broken.
    <https://openai.com/index/separating-signal-from-noise-coding-evaluations/>
55. OpenAI, *Scientific computing in the age of agentic AI*, **28 July 2026** —
    exploratory report on eight agent-assisted scientific-software projects;
    emphasizes external acceptance targets, staged iteration, expert validation,
    and durable stewardship.
    <https://openai.com/index/scientific-computing-agentic-ai/>
56. OpenAI, *GPT‑Red: Unlocking Self-Improvement for Robustness*, **15 July 2026**
    — official archive summary describes automated self-play red teaming for
    robustness, alignment, and prompt-injection resistance. The article URL
    is canonical but blocks automated retrieval, so no numerical result is
    quoted here.
    <https://openai.com/index/unlocking-self-improvement-gpt-red/>

> **Methodology note.** GPT‑5.6 capability and efficiency numbers are vendor-
> reported and are labeled as such. The ARC‑AGI‑3 and SWE‑Bench Pro findings are
> first-party audits with disclosed harness/audit methods, not independent model
> evaluations. The scientific-computing report is retrospective and exploratory,
> not a controlled productivity trial. Source [56] is scoped to the official
> archive text that could be verified. Every source above is dated July 2026.
