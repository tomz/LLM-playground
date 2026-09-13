# SOTA Watch — LLM & AGI · 2026-09

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-09-07  ·  **Updated:** 2026-09-13  ·  **Status:** draft — month to date through September 13

> **More capable agents, cheaper persistent state, and harder evidence about control.**
> September's first week brought GPT-6 Astra, Claude Fable/Mythos 5.1, Gemini 3.8
> Flash/Cyber, and Muse Spark 1.3. The second week adds DeepSeek-V4.1-Flash:
> released multimodal weights, a causal encoder–decoder architecture, and much
> smaller KV caches for input-heavy agent workloads. But architectural efficiency
> is not the same as cheap completed work: reasoning effort, harness choice,
> cache reuse, and changing API contracts still determine the bill. Meanwhile,
> revised Astra safety disclosures and Anthropic's real-world misuse report make
> the distinction between capability, alignment, monitorability, and actual
> enforcement more important—not less.

**Scope:** a living September 1–13 survey, not a full-month retrospective or an
exhaustive leaderboard. Granite 4.2 [63] remains an explicitly labeled **August 25
carry-in**. Capability and efficiency claims are vendor-reported unless labeled
**independent evaluator**; Artificial Analysis [68, 69] provides that additional
perspective in this refresh. None of these model results was reproduced in this
repository. Source numbering continues from [August](./2026-08-sota-llm-agi.md),
which ended at 57.

**What changed since September 7:**

- Added the September 10 DeepSeek release, its technical report, released-weight
  revision, cache architecture, and post-training recipe. [64–66]
- Re-read Astra's **September 9 system-card revision**, including its distinction
  between verbalized metagaming and actual oversight gaming. [58]
- Added a versioned, independent DeepSeek evaluation—not a cross-vendor winner. [68, 69]
- Refreshed token prices and recorded conflicting DeepSeek notices: the current
  pricing page says V4-Pro service continues, superseding the launch post's
  announced September 14 redirect. [64, 67]
- Added Anthropic's September misuse disclosure, while preserving its **December
  2025–August 2026 observation window** and model-family limitations. [70]

---

## The frontier this month

- **DeepSeek-V4.1-Flash makes agent-state cost an architectural target.** The
  September 10 release natively accepts images/text and generates text, with up
  to 1M context. The technical report specifies **552B backbone parameters plus
  196B Engram conditional-memory parameters**, but only **8B active during
  prefill and 16B during decode**. Causal encoder–decoder processing, cross-layer
  cache reuse, FP4 cache storage, and bounded replay target different parts of
  serving cost. MIT-licensed weight shards are listed in the released repository;
  this is not merely an open-weights promise or an 8B model that fits an 8B
  deployment budget. [64–66]
- **GPT-6 Astra combines broader deployment with a revised safety account.**
  OpenAI's September 3 system card calls Astra its first broadly deployed model
  to reach the company's **Critical cybersecurity threshold**. The API card
  lists text/image input, text output, a 1,050,000-token context, **922,000 maximum
  input tokens**, and 128,000 maximum output tokens. The September 9 revision
  clarifies what alignment evaluations establish about generalization and
  evaluation awareness. Neither large context nor better simulated alignment
  establishes reliable open-ended autonomy. [58, 59]
- **Claude Fable 5.1 advances scientific and coding work, with safeguards inside
  the evaluation contract.** Anthropic reports **52.6% on Terminal-Bench-Science
  0.1**, versus Fable 5's 24.7%, and 55.8% on Terminal-Bench 4.0 in its setup.
  Fable 5.1 and Mythos 5.1 share the underlying model but have different
  safeguards/access rules. Some evaluations substitute earlier models when
  safeguards intervene; they are not uniformly measurements of one unrestricted
  model. [60]
- **Gemini 3.8 Flash makes extra reasoning a visible cost/quality trade-off.**
  Google's September 2 release reports **54.9% on HLE-Verified** and stronger
  long-horizon work than 3.7 Flash. Google attributes part of the gain to more
  reasoning and iterative tool use, which can consume more tokens. Flash Cyber's
  defensive-security capabilities remain under the restricted Fairwind Program,
  not an unrestricted endpoint at ordinary Flash prices. [61]
- **Muse Spark 1.3 emphasizes collaboration and constraint retention.** Meta's
  September 2 announcement describes better long-thread task tracking,
  clarification, requests for help, and confirmation before consequential
  actions. Meta engineers report roughly **20% fewer tool calls and 25% fewer
  tokens** than 1.2 on their coding comparisons. The retrieved page makes `max`
  reasoning available in Muse Code and Meta Model API; open weights remain on
  the roadmap. Those efficiency claims are workload-specific. [62]
- **Scientific-agent evidence extends beyond mathematical certificates.**
  Anthropic reports experimentally validated protein binders, a Venus elevation
  map, and optimizations for seven open-source biology models. The GPU work
  reports **1.4–2.5× H100 inference speedups**, with identical outputs, but the
  retrieved announcement still describes the optimization code as forthcoming.
  These are first-party case studies, not a controlled estimate of autonomous
  research productivity or independent confirmation of every result. [60]
- **Safety evidence now includes observed misuse as well as system-card tests.**
  Astra's revised card still reports reduced monitorability despite better
  simulated alignment. Anthropic's September disclosure describes disrupted
  operations across seven harm areas, largely involving earlier Claude families.
  These are different evidence types: adversarial evaluations, simulated
  deployment, and selected real-world incidents cannot be pooled into one
  safety score. [58, 70]
- **Late-August carry-in: Granite 4.2 provides a smaller open post-training
  reference.** IBM's August 25 card describes an Apache-2.0 8B dense model with
  full, low-effort, and non-thinking modes. SFT, multi-environment GRPO, and
  subsequent preference alignment use separate generation/update GPU pools.
  That is a concrete training account alongside September's larger systems,
  not September news or a demonstrated match for the largest closed models. [63]

### DeepSeek's technical contribution: separate compute, live cache, and persisted state

The report's architecture is more informative than a single memory-reduction
headline. Its comparisons below are against **DeepSeek-V4-Flash at equal sequence
lengths**, and remain first-party engineering results. [65, 66, §§2–3]

| Component | What changes | What the result does **not** establish |
|-----------|--------------|----------------------------------------|
| **Causal Encoder–Decoder (CED)** | A 20-layer causal encoder feeds a 20-layer decoder; decoder global KV comes from final encoder states. Most prompt tokens avoid full decoder computation, producing the 8B-prefill / 16B-decode active-parameter split. | The checkpoint is not an 8B dense model. Active compute does not describe weight residency, Engram storage, communication, or total serving memory. |
| **CSA2 + FP4 main KV** | Full/Reindex/Reuse layer modes share global KV and sparse indices; hierarchical indexing limits deeper search. Reported global KV is **890 bytes/token**, roughly **¼** of V4-Flash's footprint. | This is the **global KV component**, not all HBM usage. It is an architecture trained for this design, not a generic FP4 toggle for any existing model. |
| **SWA Bounded Replay** | Reconstructs missing sliding-window state by replaying a bounded recent window instead of persisting all that state; reported persistent KV footprint is roughly **⅛** of V4-Flash's. | Reconstruction is approximate, not exact checkpoint restoration. Quality at cache-resumption boundaries and extreme retrieval cases needs separate testing. |

The post-training account is equally important: **SFT → RL → on-policy
distillation**, with the claimed improvements concentrated in task/environment
synthesis, verification, filtering, deduplication, and difficulty calibration,
not a newly invented RL objective. Tasks are treated as **problem, environment,
verification-system** triplets and re-audited over their lifecycle. This is
DeepSeek's attribution, not an independently isolated causal estimate of how
much each component contributed. [66, §5.1]

Its asynchronous implementation also differs from Granite's: DeepSeek reports
**co-locating rollout and training on the same devices and time-sharing them**,
with sample-level dispatch, interruption/resumption, length-bias mitigation,
and masking of excessively stale tokens. Do not describe all asynchronous RL
as separate GPU pools or assume that a lease queue already implements either
system. [63; 66, §5.2]

### What the numbers establish—and what they do not

These observations use different tasks, harnesses, budgets, and safeguards.
**This is an evidence ledger, not a cross-vendor ranking.**

| Development | Reported observation | Qualification that changes the interpretation |
|-------------|----------------------|-----------------------------------------------|
| Astra simulated alignment | **34 vs 73 severity-3-or-higher flags** on 54,218 matched internal Codex tasks: about 53% fewer than Sol | First-party resampling/tool simulation and monitor judgments; neither run had a severity-4 flag. This is not a 53% reduction in observed production incidents. [58, §8.6] |
| Fable 5.1 scientific work | Terminal-Bench-Science 0.1: **52.6% vs 24.7%** | Anthropic reports standard errors of ±3.5–4.5 percentage points per model and setup differences from the public leaderboard. Preserve harness and substitute-model policy. [60] |
| Fable 5.1 computer use | OSWorld 2.0: **77.9% partial, 41.7% strict** | Partial credit is not completed work. These use the August 2026 task release and are not directly comparable with older OSWorld 2.0 numbers. [60] |
| Flash 3.8 reasoning | HLE-Verified: **54.9%** | Vendor-reported on the named variant; not interchangeable with original HLE or other tool/no-tool settings. Higher effort can increase tokens. [61] |
| Flash Cyber patching | CWE-Bench pass@1: **47.2%** | An external benchmark reported by Google, not independently reproduced here. Ordinary Flash access/pricing does not establish Cyber terms. [61] |
| Muse Spark 1.3 efficiency | About **20% fewer tool calls, 25% fewer tokens** vs 1.2 | Meta engineer comparisons on coding workflows, not a universal workload distribution or matched-task cost estimate. [62] |
| Scientific GPU optimization | **1.4–2.5×** across seven models on H100 | Identical outputs are vendor-reported; some whole-job gains use cross-sequence caching. Public optimization code remains announced, not verified here. [60] |
| V4.1-Flash agent performance | **90.6% Terminal-Bench 2.1; 31.2% Terminal-Bench 4.0** | DeepSeek's maximum-effort settings and named scaffolds/context limits. These are different benchmarks, not a before/after decline or direct comparison with Anthropic's setup. The paper acknowledges gaps on the hardest tasks. [65; 66, §§5.3, 6] |
| V4.1-Flash effort scaling | Effort 25→100 raises DeepSWE v1.1 **66.0%→74.2%**, with roughly **2.5× more output tokens** reported across the effort comparison | First-party ablation. The internal scalar is not a calibrated unit across models; the paper maps public API `low`/`high`/`max` to 50/75/100, not arbitrary API integers. [66, §§5.1.4, 5.3.3] |
| **Independent evaluator: Artificial Analysis** | Retrieved V4.1-Flash max-effort page: **Intelligence Index v4.3 = 40**, **209.5 output tokens/s**, **250M evaluation output tokens**, **$476.89** for its index evaluation | AA's own evaluation, not this repository's. Index, serving speed, and cost measure different things; token speed is not completed-task latency. Preserve index version, effort, provider/settings, and retrieval date. Do not compare this score with older index versions. [68, 69] |

AA's v4.3 methodology weights Agents 30%, Coding 20%, Scientific Reasoning 20%,
and General 30%. That composition is useful context for its composite, not an
agreed definition of intelligence. Its cost and token-use measurements strengthen
the case for evaluating **accepted results per budget**, rather than selecting a
model from token prices or maximum-effort scores alone. [68, 69]

### Token prices are not completed-task prices

Rates below were rechecked **September 13, 2026**, in **USD per million tokens**.
They are selected standard token rates, not all-in quotes. Tool charges, cache
writes, batch/fast modes, context thresholds, taxes, and negotiated terms can
change a bill. There is no quality-equivalence claim. [59–61, 67]

| Model / tariff | Input, cache miss | Cached input/read | Output | Important condition |
|----------------|------------------:|------------------:|-------:|---------------------|
| GPT-6 Astra | $10 | $1 | $50 | Above 272K input tokens, the **full request** uses 2× input/cache rates and 1.5× output rates. Standard cache writes are $12.50/M before applicable context multipliers. [59] |
| Claude Fable 5.1 | $10 | $0.25 | $50 | Anthropic estimates ~25% lower typical billed cost than Fable 5, up to ~45% for highly agentic work, through cheaper cache reads—not reduced base input/output rates. [60] |
| Gemini 3.8 Flash | $0.75 | Not specified in the cited launch post | $3.75 | Introductory rates end December 31, 2026; announced January 1, 2027 rates are $1.50 input / $7.50 output. Not Cyber prices. [61] |
| DeepSeek-V4.1-Flash, **peak** | $0.30 | $0.006 | $1.20 | Peak: Monday–Friday, **01:00–04:00 and 06:00–10:00 UTC**. Use `deepseek-flash`; cache-hit and cache-miss tokens have different rates. [67] |
| DeepSeek-V4.1-Flash, **off-peak** | $0.15 | $0.003 | $0.60 | All other hours; half of peak rates. A cheaper tariff is not a guaranteed latency/service-level improvement. [67] |

**Practical interpretation:** count failed branches, repeated context,
reasoning/output charges, tools, verification, and human intervention. Cache
savings depend on actual reuse; architectural KV compression, billed cache hits,
and textual context compaction are **three different mechanisms**. A longer
context can increase cost and exposure to untrusted material. These are evaluation
recommendations, not measured September repository gains.

### API identifiers and service notices are part of reproducibility

DeepSeek's launch post says `deepseek-v4-flash` and
`deepseek-v4-flash-vision-exp` now route to V4.1-Flash; the current pricing page
confirms those legacy models have retired. However, the launch post also
announces that `deepseek-v4-pro` will redirect on **September 14 at 04:00 UTC**,
while the current pricing page explicitly says **V4-Pro service will continue
after September 14 with unchanged billing**, in response to user demand. Record
both notices and use the current pricing page for the published service terms;
do not repeat the earlier planned cutoff as settled fact. **No live routing test
was performed for this report.** [64, 67]

A requested model string therefore does not necessarily identify immutable
weights. Record the requested ID, resolved model/version where exposed, effort,
request date, tariff, harness revision, and intervention policy. Astra's retrieved
snapshot list exposes only `gpt-6-astra`; do not invent a dated snapshot. For open
weights, pin the actual repository revision and artifacts. [59, 65, 67]

### Safety refresh: revised definitions and retrospective incident evidence

The **September 9 Astra system-card change log** clarifies that its metagaming
measurement concerns what is *verbalized* in chain of thought. Oversight gaming
is the narrower case where the model acts on reasoning about grading/reward or
monitoring in a way that undermines the evaluation's intended meaning. OpenAI
removed a previous comparison plot to reduce confusion and expanded the
limitations around alignment generalization. Neither absence of observed
failures nor absence of verbalized metagaming establishes absence of the
underlying behavior. The card continues to report lower monitorability in
adversarial settings despite improved average alignment. [58, change log;
§§8–9]

Anthropic's **September misuse disclosure** adds selected observed cases across
cyber operations, influence, surveillance, scams/fraud, biological misuse,
conventional weapons, and illicit distillation. The activity was disrupted
**between December 2025 and August 2026**. The report says Haiku, Sonnet, and Opus
were used; no cases involved Fable/Mythos-class models **except one illicit
distillation case**. These are notable cases selected by the provider, not a
representative sample, a measured prevalence rate, or proof that unobserved
misuse did not occur. Attribution, disruption outcomes, and uplift assessments
remain Anthropic's account rather than independently reconstructed findings
here. [70]

**Implication:** pair model evaluations with narrowly scoped permissions,
external action controls, incident-response procedures, and auditable outcome
checks. Do not substitute a cooperative safety benchmark or readable reasoning
trace for enforcement. Conversely, do not misattribute earlier-model incidents
to a new model merely because the disclosure appeared during its launch month.

### The AGI signal: broader work, still bounded evidence

The meaningful signal is broader long-horizon work across labs, an increasingly
explicit architecture for storing/reusing agent state, and investment in
verifiable training environments. The evidence still does **not** establish
reliable open-ended autonomy, autonomous recursive self-improvement, or AGI
under an agreed operational definition. Google's agentic-development account and
DeepSeek's task-synthesis pipeline describe research processes, not controlled
demonstrations of self-sustaining improvement. [61; 66, §§5–6]

August's separation of discovery, verification, novelty, and expert review still
applies. Lean certificates, wet-lab binding results, passing software tests,
composite indices, and incident reports answer different questions. DeepSeek's
own limitations section notes gaps on hard reasoning and untested robustness
boundaries despite strong everyday-agent results. Record what an agent was
**authorized** to do as well as whether its output passed a check. [58, 60, 66, 70]

---

## TL;DR — what we harvested

- **No new implementation or model benchmark is claimed in this refresh.** It
  updates the sourced report, evidence ledger, and navigation; external results
  remain external results.
- **Carry forward August's foundations:** full-attempt provenance,
  candidate/verifier cost accounting, structured proof/checker boundaries, and
  independent-review release gates. Their prior status is documented in the
  [August edition](./2026-08-sota-llm-agi.md) and the
  [July/August design document](../frontier-platform/docs/15-july-august-2026-harvest.md).
- **Next harvest, still planned:** versioned model/harness/billing contracts,
  cache-aware completed-task costs, audited task/environment/verifier triplets,
  and externally enforced approvals. Cache architecture and asynchronous RL need
  separate designs and measured prototypes, not renamed existing helpers.

---

## Hardware envelopes per project

Sizing targets, not a description of one workstation or a promise that maximum
model size and context fit simultaneously. Minimal means a meaningful small job;
ideal means a useful experimental envelope. API evaluation also requires an
explicit monetary budget and access approval where applicable.

| Project | Scale | Minimal | Ideal | Unlocks at ideal |
|---------|-------|---------|-------|------------------|
| **nanogpt-edu** | 10M–100M | laptop CPU / 8 GB GPU | 1× H100 80 GB | repeated seeded experiments and measured search/verifier overhead |
| **midgpt** | 124M–1.5B | 1× 16 GB GPU at smaller sizes | 8× H100 single node | useful-model context/effort ablations and task-level cost/quality curves |
| **distgpt** | 1B–70B | 1 node × 8× A100 for smaller configurations | 8–64 nodes × 8× H100/B200 + fast fabric | asynchronous-rollout designs with controlled policy staleness and distributed cache measurements |
| **coder-finetune** | 0.5B–7B | 1× 8–16 GB GPU with suitable quantization/context | 1–8× H100 | matched-budget adapter/reasoning evaluations and larger-model comparisons |
| **frontier-platform** | 1B–500B+ | CPU protocol tests + budgeted API access | research fleet + isolated execution + reviewer infrastructure | long-running agents with independent enforcement, monitoring, and auditable scientific workflows |

V4.1-Flash's released checkpoint is far beyond `midgpt` and `distgpt`'s stated
model envelopes. Its active-parameter count does not make it a small local
baseline. Study reduced-scale mechanisms or use explicitly budgeted API
comparisons; deployment sizing must include the backbone, conditional memory,
cache tiers, runtime overhead, and interconnect. [65, 66]

---

## Tier 1 — high-ROI, broadly applicable

“Win” describes the intended benefit unless identified as measured above.
🟡 is a carried-forward partial foundation; 🔜 is a proposed next implementation.

| Technique | What it does | Win | Cost / risk | Min HW | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|--------|---------|-----------|
| **Safeguard- and version-aware eval contracts** | Pins model/revision or alias-resolution evidence, effort, task version, harness, tools, substitute model, and strict/partial scoring | Makes results interpretable across model and service changes | A provider may not expose immutable versions; substitutions can conceal which model did the work | CPU + model/API | [59, 60, 64, 67] | 🔜 planned; prior task-audit contract is a foundation | coder-finetune; frontier-platform |
| **Completed-task cost ledgers** | Counts cached/uncached inputs, reasoning/output, retries, tools, verifiers, review, and tariff/time windows | Exposes the actual cost/quality frontier | Billing differs by provider; a fixed price-per-token snapshot is insufficient | CPU + metered runs | [59–62, 67–69] | 🟡 prior all-branch accounting; provider/cache integration planned | nanogpt-edu; midgpt; frontier-platform |
| **Audited task/environment/verifier triplets** | Versions synthetic tasks, their environments, reference solutions, and checks; re-audits failed trajectories and difficulty | Targets useful training data rather than novelty in the optimizer alone | A flawed verifier teaches reward hacking; synthetic held-out sets can share generator biases | CPU tests + suitable small model | [66, §5.1] | 🔜 planned; prior task audits/provenance are not a task factory | nanogpt-edu; coder-finetune; frontier-platform |
| **Clarification and approval as explicit states** | Separates ambiguity, help requests, and permission for consequential actions | Makes interactive/unattended workflows easier to control | Waiting agents need escalation/timeouts; automatic approval defeats the boundary | CPU protocol tests | [58, 60, 62, 70] | 🔜 planned | coder-finetune; frontier-platform |
| **Outcome checks independent of narration** | Checks executable results, resource access, and authorization outside agent self-report | Reduces dependence on complete or faithful reasoning traces | Observation can miss failures; enforcement and incident response need separate controls | CPU sandbox + tests | [58, §§8–9; 70] | 🟡 prior verifier/release gates; authorization checks planned | nanogpt-edu; midgpt; coder-finetune; frontier-platform |
| **Reasoning-effort ablations** | Measures shallow/deep/no-thinking settings on the same held-out tasks with complete token/time costs | Finds effort levels with useful marginal gains | Effort labels/scalars are not calibrated across models; maximum effort is not automatically optimal | small local model or API | [59, 61, 63; 66, §5.3] | 🔜 planned | midgpt; coder-finetune |
| **Scientific artifact validity contracts** | Separates numerical equivalence, formal validity, experimental evidence, and novelty review | Prevents a local check from being called a validated discovery | Domain experts, lab access, or reproducible environments may dominate cost | CPU checks; domain-specific resources | [60] | 🟡 prior proof/review contracts; broader domains track | coder-finetune; frontier-platform |

---

## Tier 2 — scale- or hardware-gated wins

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| **CED + cross-layer sparse-cache reuse** | Separates prefill/decode compute and shares global KV/index states across layers | DeepSeek reports lower prefill active compute and smaller global KV | Requires architecture/training changes, compatible kernels, and quality checks; not a drop-in GPT-2 flag | [65; 66, §2] | track / ideal; reduced-scale research prototype first | midgpt; distgpt; frontier-platform |
| **FP4 cache + bounded replay** | Compresses global state and reconstructs recent sliding-window state instead of persisting everything | Reported global/persistent KV footprints of roughly ¼ / ⅛ of V4-Flash | Separate live-cache, persisted-cache, bandwidth, and replay measurements; test approximation errors and cache-resumption boundaries | [66, §§2–3, 6] | track; text compaction and exact training resume are different mechanisms | distgpt; frontier-platform |
| **Asynchronous multi-environment RL / OPD** | Reduces rollout stragglers; IBM uses separate GPU pools, DeepSeek co-locates and time-shares rollout/update execution | Enables higher concurrency for heterogeneous trajectories | Policy-version/log-prob provenance, length-bias control, staleness limits, backpressure, and scheduling; implementations are not interchangeable | [63; 66, §5.2] | ideal; prior lease queue is not a full asynchronous trainer | distgpt; coder-finetune; frontier-platform |
| **Long-context, high-effort agent serving** | Couples large contexts, iterative tools, and adjustable reasoning budgets | Supports work requiring substantial accumulated state | API spend or inference fleet, KV capacity, latency, and untrusted-context isolation; include context surcharges and alias changes | [59, 61, 62, 67] | 🟡 prior compaction/budget protocols; end-to-end evaluation planned | midgpt; distgpt; frontier-platform |
| **Scientific GPU-kernel optimization** | Generates kernels and reuses intermediate results for repeated scientific workloads | Anthropic reports 1.4–2.5× H100 inference gains across seven models | Representative inputs, target GPU, numerical-equivalence checks, and kernel expertise; public optimization code still forthcoming in the cited source | [60] | track; no local reproduction | distgpt; frontier-platform |
| **Trusted-access defensive agents** | Combines stronger security capabilities with vetting and constrained execution | Extends defensive work while controlling access | Approved programs, isolated targets, independent approvals, and organizational accountability—not GPU capacity alone | [58, 60, 61, 70] | track / ideal; not an implementation of vendor access controls | frontier-platform |
| **Full-trajectory monitoring plus enforcement** | Combines behavior monitoring, action logs, resource isolation, and blocking checks | Adds defense beyond learned alignment | Monitoring compute, privacy/retention rules, evaluated detectors, and incident response; no detector guarantees absence of misuse | [58, §§9–10; 70] | ideal; design target, not a shipped security guarantee | frontier-platform |

---

## Tier 3 — research bets (track, don't build yet)

- **Monitorability that scales with capability.** Use Astra's revised definitions:
  verbalized metagaming, actual oversight gaming, and monitor evasion are not
  interchangeable rates. Track independent replications and realistic
  false-positive/false-negative behavior, not only explicitly adversarial tests.
  **Harvest: track · frontier-platform.** [58, change log; §§8–9]
- **Agentic self-improvement with controlled evidence.** Google's development
  account and DeepSeek's synthetic task pipeline motivate better experiments,
  not a declaration of autonomous recursive improvement. Track fixed researcher/
  compute budgets, task-quality audits, independent held-out tasks, and gains
  across successive iterations. **Harvest: track · nanogpt-edu; distgpt;
  frontier-platform.** [61; 66, §5]
- **Robustness of approximate long-context state.** DeepSeek explicitly flags
  sparse-selection errors and approximate replay in untested boundary cases.
  Evaluate retrieval, cache eviction/resumption, replay frequency, and quality
  under long trajectories—not just bytes per token or needle retrieval at one
  context length. **Harvest: track · midgpt; distgpt; frontier-platform.** [66, §6]
- **General-purpose autonomous science.** Bindings, maps, kernels, and proofs
  suggest breadth, but selected successes do not reveal the full attempted-task
  denominator or human contribution. Retain domain-specific checks and
  independent review. **Harvest: track · coder-finetune; frontier-platform.** [60]
- **Private enterprise oversight.** Anthropic's Enterprise Frontier Safeguards
  proposes customer-controlled storage and customer-led review. The retrieved
  announcement still says phased rollout starts later in the fall; it is not
  verified as generally shipped by this cutoff. Incident response and privacy
  governance remain separate requirements. **Harvest: track · frontier-platform.** [60, 70]
- **Open weights versus reproducible training.** Granite and V4.1-Flash have
  released weights; Muse Spark's remain future work. Pin licenses and artifacts,
  then separately assess data disclosure, training recipes, inference support,
  and independent evaluation. A public checkpoint does not establish complete
  training reproducibility. **Harvest: track · coder-finetune; distgpt;
  frontier-platform.** [62, 63, 65, 66, 68]

---

## Roadmap by project

Carried forward from August, with September's next measurements added. These are
recommendations, not authorization to launch paid runs or implement production
safety systems as part of this documentation update.

| Project | Next harvest | Hardware tier | Notes |
|---------|--------------|---------------|-------|
| **nanogpt-edu** | Run the existing candidate/provenance harness with an external verifier and repeated seeds; add versioned problem/environment/checker fixtures | minimal | Preserve the verifier-timing task. Count failed generations and invalid synthetic tasks, not just the best accepted result |
| **midgpt** | Drive candidate search with a model/checker; compare context and effort at matched task budgets | minimal→ideal | Preserve model-backed search. Separate textual compaction from architectural KV compression, and measure strict success plus end-to-end time/cost—not prefill alone |
| **distgpt** | Attach generation workers and CPU verifiers to the lease queue; design policy-version, staleness, and length-bias accounting | ideal | Preserve the real-worker backend task. Choose separate-pool versus time-shared execution explicitly before claiming an IBM- or DeepSeek-style asynchronous trainer |
| **coder-finetune** | Integrate a jailed Lean checker and held-out theorem set; extend eval provenance to effort, safeguard, substitute-model, and API-version settings | minimal→ideal | Preserve exact-theorem and novelty boundaries. Granite 8B is a comparison above the usual 7B envelope; V4.1-Flash is a much larger external baseline, not a default local fine-tune |
| **frontier-platform** | Add literature/reviewer backends, then specify approval enforcement, monitor evaluation, model/tariff provenance, and incident response | minimal→ideal | Preserve the external-review workflow. Keep authorization independent of model confidence; cache-tier and production-control implementations need corresponding design updates |

---

## What shipped this month

- **This September 13 documentation refresh:** expanded month-to-date coverage,
  seven additional sources, a rechecked pricing table, the Astra revision and
  DeepSeek notice correction, independent-evaluator evidence, and updated
  [edition-index](./README.md) / [repository README](../README.md) navigation.
- **No new training, serving, safety, or evaluation implementation is claimed.**
  July/August protocol work remains prior work. This refresh involved source and
  document checks, not paid model runs, weight downloads, GPU benchmarks, Lean
  executions, or new subproject test-suite results.

---

## Sources

Sources continue the cumulative numbering from August (57). **All sources below
were retrieved or rechecked September 13, 2026.** Publication/release dates and
revision dates are distinguished from retrieval dates; mutable pages are not
historical launch snapshots.

58. OpenAI, *GPT-6 Astra System Card*, **3 September 2026; revised 9 September**.
    The change log revises alignment-generalization and verbalized-metagaming
    explanations. §§1, 8–10 support the Critical cyber classification, matched
    Codex simulation, monitorability caveats, and deployment safeguards. These
    are OpenAI's assessments under its framework, not a universal risk taxonomy.
    <https://deploymentsafety.openai.com/gpt-6-astra/gpt-6-astra.pdf>
59. OpenAI API documentation, *GPT-6 Astra*, **live model card**. Input/context/
    output limits, reasoning efforts, listed aliases, token/cache-write prices,
    and long-context surcharges. Markdown representation rechecked September 13.
    <https://developers.openai.com/api/docs/models/gpt-6-astra.md>
60. Anthropic, *Introducing Claude Fable 5.1 and Claude Mythos 5.1*,
    **September 2026** (month-level date on the primary page). Benchmark tables
    and footnotes, scientific case studies, cache pricing, safeguard/substitute-
    model distinctions, trusted access, and announced EFS rollout. Optimization
    code and EFS availability are not upgraded from “announced” by this refresh.
    <https://www.anthropic.com/claude-fable-and-mythos-5-1>
61. Google, Tulsee Doshi and Raluca Ada Popa, *Introducing Gemini 3.8 Flash and
    3.8 Flash Cyber*, **2 September 2026**. Reasoning/token-effort warning,
    HLE-Verified and patching claims, Fairwind access, and pricing-expiry footnote.
    External benchmark names do not make vendor-reported scores independent.
    <https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/>
62. Meta AI Research, *Introducing Muse Spark 1.3*, **2 September 2026**.
    Collaboration/approval behavior, coding-workflow efficiency claims, current
    API/Muse Code availability including `max`, and future open weights.
    <https://research.meta.ai/blog/introducing-muse-spark-1-3>
63. IBM Granite Team, *Granite-4.2-8B* model card, **25 August 2026 release**
    (**carry-in**, not September news). Apache-2.0, 8B dense architecture, 128K
    native context with a stated extension to 512K, thinking modes, and separate-
    pool SFT/asynchronous GRPO/preference-alignment account. Context extension
    remains a stated capability, not independently validated here.
    <https://huggingface.co/ibm-granite/granite-4.2-8b/blob/main/README.md>
64. DeepSeek, *DeepSeek-V4.1-Flash: Smarter, Faster, More Efficient*, **10 September
    2026** in the official API-news index. Release, native vision, aliases, and
    pricing announcement. Its V4-Pro redirect notice conflicts with the later
    continuation notice on the current pricing page [67]; retain that distinction.
    <https://api-docs.deepseek.com/news/news260910/>
65. DeepSeek-AI, *DeepSeek-V4.1-Flash* model card and released repository,
    **revision `dba1be0a40aa45a94ad051997016db3960a90277`**, inspected September 13.
    MIT license, architecture, maximum-effort evaluation setup, and deployment
    materials. The repository metadata lists **48 safetensors weight shards**;
    their presence was checked without downloading or loading the weights.
    <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/README.md>
    Repository metadata: <https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash>
66. DeepSeek-AI, *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*,
    **September 2026 technical report**, at the same repository revision.
    §§2–3: CED, CSA2, FP4 global KV, and approximate SWA replay; §5: task
    synthesis, effort presets, asynchronous RL/OPD, and evaluation conditions;
    §6: robustness and difficult-task limitations. These are first-party results.
    <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf>
67. DeepSeek API documentation, *Models & Pricing*, **live page, September 13
    snapshot**. Flash cache-hit/miss and output rates, UTC peak windows, legacy
    Flash aliases, and the explicit notice that V4-Pro service/billing continues
    after September 14. Service behavior was not tested by making paid requests.
    <https://api-docs.deepseek.com/quick_start/pricing>
68. Artificial Analysis, *DeepSeek V4.1 Flash (Reasoning, Max Effort): Intelligence,
    Performance & Price Analysis*, **live evaluation page, September 13 snapshot**.
    Independent-evaluator index **v4.3**, output-token speed, evaluation token use,
    and cost. This is an AA measurement, not our reproduction or an overall rank
    across every model/harness. Page values can change with reevaluation.
    <https://artificialanalysis.ai/models/deepseek-v4-1-flash>
69. Artificial Analysis, *Intelligence Benchmarking Methodology*, **v4.3 as
    retrieved September 13**. Evaluation composition, category weights, and
    setup details. Index versions and task/harness changes matter to comparisons.
    <https://artificialanalysis.ai/methodology/intelligence-benchmarking>
70. Anthropic, *Detecting and countering misuse of AI: September 2026*,
    **September disclosure covering activity disrupted December 2025–August
    2026**. Selected incidents across seven harm areas, provider attribution,
    and model-family scope, including the illicit-distillation exception.
    This is first-party incident reporting, not a population prevalence study.
    <https://www.anthropic.com/threat-intelligence-report-september-2026>

> **Methodology and coverage limits.** This refresh rechecked the original primary
> sources, read the new release/model/pricing materials and relevant technical-
> report sections, and inspected AA's model page and methodology. Search and
> secondary trackers were used for discovery, not as evidence for quoted numbers.
> Unlike the first-week draft, this edition includes an independent evaluator,
> but it does not supply a matched cross-vendor experiment or assert a universal
> winner. Threat cases remain selected provider observations. Model-weight
> listings were inspected, not downloaded or executed. Launch notices, pricing,
> aliases, evaluator indices, and even system cards can change; the dates,
> revision, benchmark versions, and explicit notice conflict above are part of
> the evidence. Missing items mean “not verified in this survey,” not “nothing
> else happened.” September remains open for further primary research and
> independent replication.
