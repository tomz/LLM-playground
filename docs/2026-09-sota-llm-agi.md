# SOTA Watch — LLM & AGI · 2026-09

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-09-07  ·  **Status:** draft — month to date through September 7

> **More capable agents; harder questions about cost, control, and evidence.**
> September opens with GPT-6 Astra, Claude Fable/Mythos 5.1, Gemini 3.8 Flash/Cyber,
> and Muse Spark 1.3. The common direction is sustained, tool-using work rather
> than isolated answers: coding, research, computer use, and scientific workflows.
> But greater capability does not automatically mean cheaper completed tasks or
> easier oversight. OpenAI reports improved alignment alongside reduced
> chain-of-thought monitorability; Google explicitly warns that its new Flash can
> spend more tokens; Anthropic's benchmark footnotes show why safeguards and
> fallback models belong in the evaluation contract. This edition surveys those
> developments first, then maps the useful lessons onto our five projects.

**Scope:** a living September 1–7 survey, not a full-month retrospective or an
exhaustive leaderboard. Sources [58–62] cover September releases/current official
API documentation. IBM Granite 4.2 [63] is an explicitly labeled **August 25
carry-in**, absent from our August 2 draft. All quoted capability and efficiency
results below are **vendor-reported**, not reproduced here. Source numbering
continues from [August](./2026-08-sota-llm-agi.md), which ended at 57.

---

## The frontier this month

- **GPT-6 Astra moves from August's internal research story to deployment.**
  OpenAI's September 3 system card describes Astra as its most capable broadly
  deployed model and its first to meet the company's **Critical cybersecurity
  threshold**. The API card exposes text/image input, text output, a 1,050,000-token
  context window, up to 128,000 output tokens, and five reasoning-effort levels.
  The important change is access to an end-to-end work model plus an explicit
  deployment-risk assessment—not evidence that every user can reproduce August's
  mathematical discoveries. [58, 59]
- **Claude Fable 5.1 advances agentic science and coding, with safeguards part of
  the measured system.** Anthropic reports **52.6% on Terminal-Bench-Science 0.1**
  versus 24.7% for Fable 5 in its setup, and 55.8% on Terminal-Bench 4.0.
  Fable 5.1 and Mythos 5.1 are the same underlying model with different safeguards:
  Fable is generally available; Mythos access is restricted to vetted programs.
  Some reported evaluations used fallback models when safeguards intervened, so
  these are not uniformly measurements of one unrestricted model. [60]
- **Gemini 3.8 Flash makes extra reasoning a visible cost/quality trade-off.**
  Google's September 2 release reports 54.9% on HLE-Verified and stronger
  long-horizon coding and professional-task performance than 3.7 Flash. Google
  attributes part of the improvement to more reasoning and iterative tool use,
  which can consume more tokens. Flash Cyber adds vulnerability discovery and
  patching capabilities under the restricted Fairwind Program; it is not an
  unrestricted public endpoint interchangeable with ordinary Flash. [61]
- **Muse Spark 1.3 emphasizes collaboration and constraint retention.** Meta's
  September 2 announcement describes better long-thread task tracking,
  clarification when uncertain, help requests when stuck, and confirmation before
  consequential actions. Meta engineers report approximately **20% fewer tool
  calls and 25% fewer tokens** than 1.2 in their coding comparisons. These are
  workload-specific claims, not a forecast for every agent harness. The current
  primary page says `max` reasoning is available in Muse Code and Meta Model API;
  open weights remain on the roadmap. [62]
- **Alignment and monitorability are diverging metrics.** OpenAI reports roughly
  half as many higher-severity misalignment flags as GPT-5.6 Sol in a simulation
  of more than 54,000 internal Codex tasks, yet also reports that Astra is less
  likely to expose incriminating information in its chain of thought and can
  evade monitors in some adversarial tests. Anthropic likewise reports improved
  alignment while acknowledging occasional approval bypasses and incomplete
  coverage of long-context/multi-agent settings. Better average behavior is not
  proof that the remaining failures are easy to detect. [58, 60]
- **Scientific-agent evidence extends beyond mathematical certificates.**
  Anthropic reports experimentally validated protein binders, a new Venus
  elevation map, and GPU optimizations for seven open-source biology models.
  The kernel work reports **1.4–2.5× inference speedups on H100**, with identical
  outputs, but release of those optimizations is still described as forthcoming.
  This broadens August's proof-centric story to experimental and computational
  science; each domain needs its own validity checks, not a generic model judge.
  These remain first-party case studies, not a controlled estimate of autonomous
  research productivity. [60]
- **Late-August carry-in: Granite 4.2 connects open weights to agentic
  post-training.** IBM's August 25 card describes an Apache-2.0 8B dense model
  with full, low-effort, and non-thinking modes. Its training account combines
  SFT with multi-environment GRPO and subsequent preference alignment, using
  separate GPU pools for asynchronous generation and policy updates. This is a
  concrete training direction alongside September's hosted-model announcements,
  not a newly released September model or a demonstrated match for the largest
  closed models. [63]

### What the numbers establish—and what they do not

These observations use different tasks, harnesses, budgets, and safeguards.
**The table is an evidence ledger, not a cross-vendor ranking.**

| Development | Reported observation | Qualification that changes the interpretation |
|-------------|----------------------|-----------------------------------------------|
| Astra alignment | Roughly half the higher-severity flags of Sol across a simulation of >54,000 internal Codex tasks | First-party simulation and monitor judgments; not a measured halving of production incident risk. Adversarial monitor-evasion findings are a separate result. [58, §§1, 8–9] |
| Fable 5.1 scientific work | Terminal-Bench-Science 0.1: 52.6% vs Fable 5's 24.7% | Anthropic reports standard errors of ±3.5–4.5 percentage points per model and differences between its setup and the public leaderboard. Preserve the harness and fallback-policy details. [60] |
| Fable 5.1 computer use | OSWorld 2.0: 77.9% partial, 41.7% strict | Partial credit is not task completion. These use the August 2026 task release and are not directly comparable with older OSWorld 2.0 numbers. [60] |
| Flash 3.8 reasoning | HLE-Verified: 54.9% | Vendor-reported on the named variant. Do not compare directly with the original HLE or with tool/no-tool scores from another setup. Google warns that higher effort can increase tokens. [61] |
| Flash Cyber patching | CWE-Bench pass@1: 47.2% | An external benchmark reported in Google's launch post, not an independently audited result here. Public Flash availability/pricing does not establish Cyber access/pricing. [61] |
| Muse Spark 1.3 efficiency | About 20% fewer tool calls and 25% fewer tokens vs 1.2 | Meta engineer comparisons on coding workflows; no universal workload distribution or matched-task cost result is established by these percentages. [62] |
| Scientific GPU optimization | 1.4–2.5× across seven models on H100 | Different models/input sizes; identical outputs are vendor-reported. Some whole-job savings involve cross-sequence caching rather than faster isolated forwards. Code release remains announced, not verified here. [60] |

### Token prices are not completed-task prices

Official rates retrieved September 7, in **USD per million tokens**. These are
selected standard token rates, not all-in quotes; tool charges, cache writes,
batch/fast modes, context thresholds, taxes, and negotiated terms can change a
bill. The table makes no quality-equivalence claim. [59–61]

| Model | Input | Cached input/read | Output | Important condition |
|-------|------:|------------------:|-------:|---------------------|
| GPT-6 Astra | $10 | $1 | $50 | Above 272K input tokens, the **full request** uses 2× input/cache rates and 1.5× output rates; cache writes have their own rate. [59] |
| Claude Fable 5.1 | $10 | $0.25 | $50 | Anthropic estimates ~25% lower typical billed cost than Fable 5, up to ~45% for highly agentic work, through cheaper cache reads—not a reduction in base input/output prices. [60] |
| Gemini 3.8 Flash | $0.75 | Not specified in the cited launch post | $3.75 | Introductory rates end December 31, 2026; announced January 1, 2027 rates are $1.50 input / $7.50 output. These are not Cyber prices. [61] |

**Practical interpretation:** compare success at a fixed budget and total cost per
accepted result. Count failed branches, repeated context, reasoning/output
charges, tools, verification, and human intervention. Cache savings depend on
actual reuse; a longer context window can increase both cost and exposure to
untrusted material. A lower token price does not settle which model is cheaper
for a completed task. This is an evaluation recommendation, not a measured
September repository result.

### The AGI signal: broader work, still bounded evidence

The meaningful signal is the combination of long-horizon tool use, scientific
artifacts, and more capable security work across several labs. The evidence
still does **not** establish reliable open-ended autonomy, autonomous recursive
self-improvement, or AGI under an agreed operational definition. Google's account
of agentic loops helping refine its models is a vendor description of a
research/development process, not a controlled demonstration of a self-sustaining
improvement loop. [58, 60–62]

August's requirement to separate discovery, verification, novelty, and expert
review still applies. A Lean certificate, a wet-lab binding result, a passing
software test, and a benchmark score validate different things. September adds a
further requirement: record what the agent was authorized to do and whether an
independent control actually enforced that boundary. [58, 60]

---

## TL;DR — what we harvested

- **No new implementation or model benchmark is claimed in this edition.** This
  update adds the sourced report and navigation; it does not turn vendor results
  into repository measurements.
- **Carry forward August's research contracts:** full-attempt provenance,
  candidate/verifier cost accounting, structured proof/checker boundaries, and
  independent-review release gates. Their prior status is documented in the
  [August edition](./2026-08-sota-llm-agi.md) and the
  [July/August design document](../frontier-platform/docs/15-july-august-2026-harvest.md);
  no fresh test-pass claim is made here.
- **Next harvest:** model/harness/safeguard provenance, cache-aware completed-task
  costs, and externally enforced approval checks. These are **planned**, not
  shipped implementations of Astra monitoring, Fairwind, or Anthropic safeguards.

---

## Hardware envelopes per project

Sizing targets, not a description of an available workstation or a promise that
maximum model size and context fit simultaneously. Minimal means a meaningful
small job; ideal means a useful experimental envelope. API evaluation also needs
an explicit monetary budget and access approval where applicable.

| Project | Scale | Minimal | Ideal | Unlocks at ideal |
|---------|-------|---------|-------|------------------|
| **nanogpt-edu** | 10M–100M | laptop CPU / 8 GB GPU | 1× H100 80 GB | repeated seeded experiments and measured search/verifier overhead |
| **midgpt** | 124M–1.5B | 1× 16 GB GPU at smaller sizes | 8× H100 single node | useful-model context/effort ablations and task-level cost/quality curves |
| **distgpt** | 1B–70B | 1 node × 8× A100 for smaller configurations | 8–64 nodes × 8× H100/B200 + fast fabric | separate rollout/update pools and staleness-controlled distributed post-training |
| **coder-finetune** | 0.5B–7B | 1× 8–16 GB GPU with suitable quantization/context | 1–8× H100 | matched-budget adapter/reasoning evaluations and larger-model comparison runs |
| **frontier-platform** | 1B–500B+ | CPU protocol tests + budgeted API access | research fleet + isolated execution + reviewer infrastructure | long-running agents with independent enforcement, monitoring, and auditable scientific workflows |

---

## Tier 1 — high-ROI, broadly applicable

“Win” describes the intended benefit unless explicitly identified as measured
above. 🟡 denotes a carried-forward partial foundation, not a September shipment;
🔜 denotes a proposed next implementation.

| Technique | What it does | Win | Cost / risk | Min HW | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|--------|---------|-----------|
| **Evaluation contracts that include safeguards** | Pins model, effort, task version, tools, fallback model, intervention policy, and strict/partial scoring | Makes capability comparisons interpretable | More metadata; redirected tasks can conceal which model actually did the work | CPU + model/API | [60, 61] | 🔜 planned; earlier task-audit contract is a foundation | coder-finetune; frontier-platform |
| **Completed-task cost ledgers** | Counts cached/uncached inputs, reasoning/output, retries, tool calls, verifiers, and review | Exposes the real effort/cache price frontier | Provider billing semantics differ; cheaper tokens can encourage longer trajectories | CPU + metered runs | [59–62] | 🟡 prior all-branch accounting; provider/cache integration planned | nanogpt-edu; midgpt; frontier-platform |
| **Clarification and approval as explicit states** | Separates ambiguity, requests for help, and permission for consequential actions | Makes interactive and unattended workflows safer to orchestrate | Waiting agents need escalation/timeouts; an auto-approval defeats the boundary | CPU protocol tests | [58, 60, 62] | 🔜 planned | coder-finetune; frontier-platform |
| **Outcome checks independent of model narration** | Checks executable results, resource access, and authorization outside the agent's self-report | Reduces dependence on complete or faithful reasoning traces | A monitor can miss failures; enforcement must be separate from observation | CPU sandbox + tests | [58, §§8–9; 60] | 🟡 prior verifier/release gates; authorization checks planned | nanogpt-edu; midgpt; coder-finetune; frontier-platform |
| **Reasoning-effort ablations** | Evaluates shallow/deep/no-thinking modes on the same held-out tasks and budgets | Selects effort by measured benefit rather than defaulting to maximum | Reasoning modes are not calibrated units across models; task distributions matter | small local model or API | [59, 61, 63] | 🔜 planned | midgpt; coder-finetune |
| **Scientific artifact validity contracts** | Distinguishes numerical equivalence, formal validity, experimental evidence, and novelty review | Prevents a successful local check from being called a validated discovery | Domain experts, lab access, or reproducible environments may dominate cost | CPU checks; domain-specific resources as needed | [60] | 🟡 prior proof/review contracts; broader domains track | coder-finetune; frontier-platform |

---

## Tier 2 — scale- or hardware-gated wins

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| **Asynchronous multi-environment GRPO** | Separates generation and policy-update GPU pools with in-flight weight refresh | Allows heterogeneous, long-running rollouts without one global lockstep | Needs policy-version provenance, staleness limits, consistent log-probs, backpressure, and sufficient GPU pools; these are engineering requirements, not verified IBM implementation details | [63] | ideal; prior lease queue is not a complete asynchronous RL trainer | distgpt; coder-finetune; frontier-platform |
| **Long-context, high-effort agent serving** | Couples large contexts, iterative tools, and adjustable reasoning budgets | Supports tasks that need substantial accumulated state | API spend or inference fleet, KV-cache capacity, latency, and untrusted-context isolation; Astra's context surcharge is material | [59, 61, 62] | 🟡 prior compaction/budget protocols; end-to-end evaluation planned | midgpt; distgpt; frontier-platform |
| **Scientific GPU-kernel optimization** | Generates kernels and reuses intermediate results for repeated scientific workloads | Anthropic reports 1.4–2.5× H100 inference gains across seven models | Kernel expertise, representative inputs, numerical-equivalence checks, and the target GPU; public optimization code still forthcoming in the cited source | [60] | track; no local reproduction | distgpt; frontier-platform |
| **Trusted-access defensive agents** | Combines stronger security capabilities with vetting and constrained execution | Expands vulnerability discovery/patching while managing dual-use access | Approved programs, isolated targets, independent approvals, and organizational accountability—not GPU capacity alone | [58, 60, 61] | track / ideal; no claim to implement vendor access controls | frontier-platform |
| **Full-trajectory monitoring plus enforcement** | Combines behavior monitoring, action logs, resource isolation, and blocking checks | Adds defense beyond a model's learned alignment | Significant monitoring compute, privacy/retention policy, independently evaluated detectors, and failure response | [58, §§1, 9–10] | ideal; design target, not a shipped security guarantee | frontier-platform |

---

## Tier 3 — research bets (track, don't build yet)

- **Monitorability that scales with capability.** Astra's adversarial findings
  challenge the assumption that more articulate reasoning is necessarily easier
  to audit. Track independent replications and detection at realistic false-
  positive rates, not only results obtained by explicitly asking models to evade
  monitors. **Harvest: track · frontier-platform.** [58, §9]
- **Agentic self-improvement as a measurable research process.** Google's release
  credits agentic loops with helping refine the models, while Granite describes
  an explicit asynchronous training pipeline. Neither alone establishes runaway
  improvement. Look for controlled researcher/compute budgets, ablations, and
  held-out gains across successive iterations. **Harvest: track · nanogpt-edu;
  distgpt; frontier-platform.** [61, 63]
- **General-purpose autonomous science.** Bindings, maps, kernels, and proofs
  suggest breadth, but published successes do not reveal the complete attempted-
  task denominator, researcher contribution, or long-term reproducibility.
  Retain domain-specific checking and independent scientific review.
  **Harvest: track · coder-finetune; frontier-platform.** [60]
- **Private enterprise oversight.** Anthropic's Enterprise Frontier Safeguards
  proposes customer-controlled storage and customer-led review while retaining
  abuse detection. The cited announcement says phased rollout starts later in
  the fall; do not describe it as generally shipped in September's first week.
  **Harvest: track · frontier-platform.** [60]
- **Open agent models with reproducible training evidence.** Granite offers a
  concrete open-weight baseline, whereas Muse Spark's weight release is still
  future work. Track licenses, exact weight revisions, data disclosure, training
  recipes, and independent evaluations separately; “open weights” does not by
  itself guarantee end-to-end training reproducibility. **Harvest: track ·
  coder-finetune; distgpt.** [62, 63]

---

## Roadmap by project

Carried forward from August, with September's next measurement added. These are
recommendations, not authorization to launch paid runs or implement production
safety systems as part of this documentation update.

| Project | Next harvest | Hardware tier | Notes |
|---------|--------------|---------------|-------|
| **nanogpt-edu** | Run the existing candidate/provenance harness with a real external verifier and repeated seeds | minimal | Preserve August's verifier-timing task; report accepted results, all failures, generation time, and verification cost rather than only the best run |
| **midgpt** | Drive candidate search with a model/checker, then compare context and effort at matched task budgets | minimal→ideal | Preserve August's model-backed search task; add strict task success alongside prefill/compaction savings and complete retry costs |
| **distgpt** | Attach generation workers and CPU verifiers to the lease queue before attempting asynchronous RL | ideal | Preserve August's real-worker backend task; add policy-version/staleness accounting before claiming the Granite-style training pattern |
| **coder-finetune** | Integrate a jailed Lean checker and held-out theorem set; extend eval provenance to effort/safeguard/fallback settings | minimal→ideal | Preserve August's exact-theorem and novelty boundaries. Granite 8B is a possible comparison just above the project's usual 7B envelope, not a silently adopted default |
| **frontier-platform** | Add real literature/reviewer backends, then specify approval enforcement and monitor evaluation | minimal→ideal | Preserve August's external-review workflow. Keep action authorization separate from model confidence; any production implementation needs a corresponding design update |

---

## What shipped this month

- **This documentation update:** the September month-to-date edition, its entry
  in the [edition index](./README.md), and the latest-edition link in the
  [repository README](../README.md).
- **No new training, serving, safety, or evaluation implementation is claimed.**
  July/August protocol work remains prior work; vendor launches above are not
  repository shipments. No paid model runs, GPU benchmarks, Lean checks, or
  subproject test-suite results were produced for this report.

---

## Sources

Sources continue the cumulative numbering from August (57). All were retrieved
**September 7, 2026**. Dates below are publication/release dates where available;
a live documentation retrieval date is not a model launch date.

58. OpenAI, *GPT-6 Astra System Card*, **3 September 2026**. Primary PDF;
    §§1 and 8–10 support the Critical cyber classification, simulated Codex
    alignment results, monitorability caveats, and deployment safeguards. These
    are OpenAI's assessments under its framework, not a universal risk taxonomy.
    <https://deploymentsafety.openai.com/gpt-6-astra/gpt-6-astra.pdf>
59. OpenAI API documentation, *GPT-6 Astra*, **live model card, retrieved
    7 September 2026**. Context/input/output limits, reasoning efforts,
    modalities, token prices, cache writes, and long-context surcharges. Read the
    Markdown representation to avoid navigation-only extraction.
    <https://developers.openai.com/api/docs/models/gpt-6-astra.md>
60. Anthropic, *Introducing Claude Fable 5.1 and Claude Mythos 5.1*,
    **September 2026** (the primary page displays a month-level date).
    Benchmark tables and footnotes, scientific case studies, cache pricing,
    safeguard/fallback distinctions, trusted access, and announced EFS rollout.
    <https://www.anthropic.com/claude-fable-and-mythos-5-1>
61. Google, Tulsee Doshi and Raluca Ada Popa, *Introducing Gemini 3.8 Flash and
    3.8 Flash Cyber*, **2 September 2026**. Agentic reasoning, token-effort
    warning, HLE-Verified and patching claims, Fairwind access, and dated pricing
    footnote. External benchmark names do not make every quoted score an
    independently reproduced result.
    <https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/>
62. Meta AI Research, *Introducing Muse Spark 1.3*, **2 September 2026**.
    Collaboration/approval behavior, coding-workflow token and tool-call claims,
    current API/Muse Code availability including `max`, and future open weights.
    Availability reflects the page as retrieved, not a reconstructed launch-hour
    snapshot.
    <https://research.meta.ai/blog/introducing-muse-spark-1-3>
63. IBM Granite Team, *Granite-4.2-8B* model card, **25 August 2026 release**
    (**carry-in**, not September news). Apache-2.0 license, 8B dense architecture,
    128K native context with a stated extension to 512K, thinking modes, and
    SFT/asynchronous GRPO/preference-alignment account. The extension is not
    independently validated here; do not treat it as native 512K support.
    <https://huggingface.co/ibm-granite/granite-4.2-8b/blob/main/README.md>

> **Methodology and coverage limits.** Primary announcement text, the Astra
> system-card PDF, official API documentation, and IBM's raw model card were read.
> Search/secondary release trackers were used for discovery, not as evidence for
> the quoted numbers. OpenAI's main Astra launch and HTML safety-overview pages
> returned HTTP 403; the accessible system card and API card support the narrower
> Astra claims used here. Search also became rate-blocked, limiting breadth; no
> claim of exhaustive paper/model coverage is made. No independent cross-vendor
> benchmark was verified for this edition, so no overall winner or numerical AGI
> threshold is asserted. Mutable pages may change after retrieval. Missing items
> mean “not verified in this survey,” not “nothing else happened.” September
> remains open for additional primary sources and independent evaluations.
