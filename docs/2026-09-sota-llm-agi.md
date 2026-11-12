# SOTA Watch — LLM & AGI · 2026-09

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-09-07  ·  **Updated:** 2026-09-30  ·  **Status:** draft — full-month edition through September 30

> **Longer-horizon agents became cheaper, more open, and harder to integrate
> safely.** September began with GPT-6 Astra, Claude Fable/Mythos 5.1, Gemini 3.8
> Flash/Cyber, Muse Spark 1.3, and DeepSeek-V4.1-Flash. The late-month releases
> add GPT-6 Sol/Luna and 6.1 Sol, Claude Opus/Sonnet 5.5, Grok 4.7, MiMo V2.6,
> Gemini 4 Argon, and Embed 5. The durable signal is not a single leaderboard
> winner. It is the convergence of adjustable reasoning, million-token state,
> mixed-environment RL, cheaper cache reuse, explicit migration contracts, and
> stronger runtime controls. Availability still ranges from MIT-licensed weights
> to generally available APIs to trusted-tester previews, and those categories
> must not be collapsed.

**Scope:** a living September 1–30 survey, not an exhaustive release catalog or
leaderboard. Granite 4.2 [63] remains an explicitly labeled **August 25 carry-in**.
Capability and efficiency claims are vendor-reported unless labeled **independent
evaluator**; Artificial Analysis [68, 69] provides that perspective for the
DeepSeek snapshot only. None of these model results was reproduced in this
repository. Released model files were listed and revision-pinned but not
downloaded or executed; technical reports were read directly. Source numbering
continues from [August](./2026-08-sota-llm-agi.md), which ended at 57.

**What changed since September 13:**

- Added the September 21–29 hosted-model wave: Grok 4.7; GPT-6 Sol/Luna and
  one-week-later 6.1 Sol; and Claude Opus/Sonnet 5.5. [71, 75–81]
- Added MiMo V2.6's MIT-licensed 1.02T-total/42B-active checkpoints, technical
  report, mixed-environment RL recipe, and released 9B SFT starting point. [72–74]
- Added September 30's Gemini 4 Argon as a **restricted, rolling-out preview**—not
  a generally available endpoint—and Embed 5 as a generally available retrieval
  release. [82, 83]
- Expanded the cost ledger and API-contract section with model supersession,
  reasoning-mode restrictions, thinking-block response changes, forced-tool-use
  failures, and cache-price differences. [76, 78–81]
- Updated all five project roadmaps and harvest tiers. No paid API run, model
  execution, GPU benchmark, or implementation change is claimed.

---

## The frontier this month

- **Gemini 4 Argon is the most expansive late-month capability claim—and remains a
  restricted rollout.** Google's September 30 announcement reports 77.9% on
  DeepSWE v1.1, 51.3% on AutomationBench, 91.7% on LVBench, and a 1M-token
  **output** limit. Access initially goes to trusted cyber defenders and testers;
  broader developer, enterprise, and consumer availability is future tense.
  Introductory API pricing is announced at $2/M input and $10/M output before a
  later $4/$20 tariff. Google also describes chain-of-thought/action monitoring,
  sealed evaluation sandboxes, and restricted unguarded cyber access. These are
  first-party claims and a staged deployment plan, not evidence of general
  availability or reliable million-token trajectories. [82]
- **Claude 5.5 makes migration behavior part of model quality.** Anthropic
  released Opus 5.5 on September 22 and Sonnet 5.5 on September 28, both with 1M
  context and 128K ordinary maximum output. Anthropic reports stronger coding and
  knowledge-work results with lower completed-task cost than their predecessors,
  but also documents breaking changes: forced tool choice can error, thinking
  blocks are model/conversation-bound, an older computer-use tool is rejected,
  and between-tool text can disappear from user-visible streams unless display
  behavior is configured. Opus always thinks; Sonnet's sampling controls also
  reject non-default temperature/top-p/top-k. [75–78]
- **OpenAI's middle tier changed twice in eight days.** GPT-6 Sol and Luna shipped
  September 22; GPT-6.1 Sol followed September 29 and is now the documented newer
  Sol. Sol and 6.1 Sol retain 1,050,000-token context and $2/$10 base input/output
  rates, while 6.1 halves cached input from $0.20/M to $0.10/M. It also drops
  `none` reasoning and moves tool calling to Responses only. Luna lists the same
  context/output ceilings at $0.10/M input and $0.50/M output. The 6.1 safety
  addendum assigns Critical cyber and High bio/chemical capability under
  OpenAI's framework. Model-family names do not preserve API behavior. [79–81]
- **MiMo V2.6 publishes the month's most actionable agentic-RL stack.** Xiaomi's
  MIT-licensed Pro/Flash repositories describe an omni-modal hybrid-SWA MoE with
  **1.02T total / 42B active parameters**, 1M context, and a five-layer
  speculative decoder. Its report gives a 25K-trajectory GRPO batch (1,568
  prompts × 16 rollouts), mixed task/harness training, groupwise grading,
  prompt-mean loss aggregation, router freezing, and failure timelines. A
  separate 9B Qwen3.5 SFT checkpoint, environment suite, and RL framework are
  released as smaller research foundations. “Self-improvement” here means a
  designed researcher-run training loop, not autonomous recursive improvement.
  [72–74]
- **Grok 4.7 is a useful price/performance datapoint, not an open recipe.** The
  September 21 release lists $2/M input and $6/M output and reports gains over
  Grok 4.6 after a larger base model and longer RL on harder, multi-hour tasks.
  The benchmark and safety evidence is first-party, and no weights or detailed
  training recipe accompany the hosted release. [71]
- **Embed 5 turns retrieval tiers into one index contract.** Cohere's September
  30 generally available Pro/Fast family shares an embedding space, so an index
  built with Pro can be queried with either tier. Both accept text/images, cover
  100+ languages and 128K input, expose Matryoshka dimensions from 256 to 2,048,
  and support float/int8/binary outputs and private deployment. This is relevant
  to retrieval cost and migration testing; Cohere's quality numbers remain
  first-party. [83]
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

### MiMo's technical contribution: mixed RL, comparative grading, and failure disclosure

MiMo V2.6 scales three coupled resources: rollout compute, task/harness diversity,
and grader compute. Its reported RL mixture is **68% coding, 12% general tool use,
13% visual design, 3% context following, and 4% cybersecurity**. Each update uses
1,568 prompts with 16 rollouts each; asynchronous partial rollouts run at a
reported policy staleness of four. Prompt-mean aggregation prevents long answers
from dominating the objective, while token-level importance ratios and separately
tuned positive/negative clipping bounds respond to policy entropy. These choices
are one integrated system, not independent claims that any single knob will
transfer unchanged to a small model. [73, §§4–6]

| Mechanism | What it contributes | Qualification / local harvest |
|-----------|---------------------|-------------------------------|
| **Groupwise Reward Synthesis (GRS)** | Builds task-specific rubrics offline by contrasting rollouts, adding quality distinctions where binary tests mark several solutions as equally passing | Grader cost and correlated judge bias can rise; use only after executable checks and held-out audits. Prototype on the released 9B/environment suite, not the 1.02T model. |
| **Groupwise Advantage Redistribution (GAR)** | Ranks successful trajectories online and redistributes advantage toward stronger passing solutions | A learned grader becomes part of the reward boundary. Preserve raw outcomes, grader prompts/versions, disagreements, and reward-hacking checks. |
| **Router freezing** | Prevents MoE load collapse during RL; the report's trainable-router run moved from ~6× to 16× peak/mean load and from 0.5% to 22% cold experts by step 20 | First-party two-run ablation at one layer. It motivates expert-load telemetry and a reduced-scale test; it is not proof that every MoE should freeze routing. |
| **MOPD2** | Reuses teacher/SFT prefixes and on-policy student continuations to train decision points, including harder-to-verify domains | Multi-teacher distillation can transfer errors and obscures which teacher supplied a capability. Keep teacher, prefix, continuation, and verifier provenance. |

The failure timeline is unusually useful. Across 30 updates, Pro took **123.1
hours** and Flash **81.8 hours** with interruptions from GPU double-bit errors,
Kubernetes and grader-network failures, rollout-length prediction bias, expert
imbalance, GPU OOM, and host-memory OOM during packing. This supports explicit
restart, staleness, queue, and memory-pressure accounting; it does not make the
large run reproducible on this repository's hardware. The released 9B SFT model
is closer to local research scale but still exceeds `coder-finetune`'s stated 7B
ceiling and reports a **77.4B-token weighted SFT mixture**. [73, §§5–7; 74]

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
| MiMo V2.6 router freezing | At decoder layer 9, trainable routing reached **16× peak/mean load and 22% cold experts** by step 20; frozen routing stayed near 5.5× and 1% while reported benchmark performance grew | First-party ablation on the Pro run and one observed layer. Load balance recovery after restoring router weights supports the diagnosis but does not isolate every training interaction. [73, §5.4] |
| Opus 5.5 coding | Terminal-Bench 4.0: **66.4%** vs Opus 5's 52.3%; FrontierCode v1.1: **54.4%** vs 48.0% | Anthropic-run settings use adaptive thinking and named effort levels. Some benchmark tasks fall back to earlier models when safeguards intervene, so the result is not uniformly one unrestricted model. [75] |
| Sonnet 5.5 coding | Terminal-Bench 4.0: **70.6%** vs Sonnet 5's 10.3%; CursorBench 4.0: **55.5%** vs 34.1% | Vendor-reported, with large model/setup changes and effort-dependent costs. Anthropic says Opus remains stronger on complex open-ended work despite some close benchmark scores. [77] |
| Grok 4.7 coding | CursorBench 4.0: **46.3%** vs Grok 4.6's 40.4%; Terminal-Bench 4.0: **37.6%** vs 20.3% | SpaceXAI-run comparisons, different effort labels, and no released model or detailed recipe. Low token rates do not imply lowest accepted-task cost. [71] |
| Gemini 4 Argon preview | DeepSWE v1.1 **77.9%**, AutomationBench **51.3%**, LVBench **91.7%** | Google-reported on an initially restricted model. The announcement supplies selected scores and deployment examples, not a generally reproducible endpoint or complete attempt denominator. [82] |

AA's v4.3 methodology weights Agents 30%, Coding 20%, Scientific Reasoning 20%,
and General 30%. That composition is useful context for its composite, not an
agreed definition of intelligence. Its cost and token-use measurements strengthen
the case for evaluating **accepted results per budget**, rather than selecting a
model from token prices or maximum-effort scores alone. [68, 69]

### Token prices are not completed-task prices

Rates below were rechecked **September 30, 2026**, in **USD per million tokens**.
They are selected standard token rates, not all-in quotes. Tool charges, cache
writes, batch/fast modes, context thresholds, taxes, and negotiated terms can
change a bill. There is no quality-equivalence claim. [59–61, 67, 71, 75–83]

| Model / tariff | Input, cache miss | Cached input/read | Output | Important condition |
|----------------|------------------:|------------------:|-------:|---------------------|
| GPT-6 Astra | $10 | $1 | $50 | Above 272K input tokens, the **full request** uses 2× input/cache rates and 1.5× output rates. Standard cache writes are $12.50/M before applicable context multipliers. [59] |
| Claude Fable 5.1 | $10 | $0.25 | $50 | Anthropic estimates ~25% lower typical billed cost than Fable 5, up to ~45% for highly agentic work, through cheaper cache reads—not reduced base input/output rates. [60] |
| Gemini 3.8 Flash | $0.75 | Not specified in the cited launch post | $3.75 | Introductory rates end December 31, 2026; announced January 1, 2027 rates are $1.50 input / $7.50 output. Not Cyber prices. [61] |
| DeepSeek-V4.1-Flash, **peak** | $0.30 | $0.006 | $1.20 | Peak: Monday–Friday, **01:00–04:00 and 06:00–10:00 UTC**. Use `deepseek-flash`; cache-hit and cache-miss tokens have different rates. [67] |
| DeepSeek-V4.1-Flash, **off-peak** | $0.15 | $0.003 | $0.60 | All other hours; half of peak rates. A cheaper tariff is not a guaranteed latency/service-level improvement. [67] |
| GPT-6.1 Sol | $2 | $0.10 | $10 | More than 272K input doubles input/cache rates and makes output 1.5× for the full request. `none`/`minimal` effort and Chat Completions tool calling are unsupported. [79–81] |
| GPT-6 Luna | $0.10 | $0.01 | $0.50 | Same long-context surcharge rule; Chat Completions function calling requires `reasoning_effort=none`. Compare accepted-task quality, not tokens alone. [79] |
| Claude Opus 5.5 | $4 | $0.20 | $20 | Five-minute cache writes are $5/M; adaptive thinking cannot be disabled. Batch input/output is 50% off; fast mode uses separate $8/$40 rates. [75, 76] |
| Claude Sonnet 5.5 | $2 | $0.20 | $10 | Five-minute cache writes are $2.50/M. Adaptive thinking defaults to high effort; migration has request/response breaking changes. [77, 78] |
| Grok 4.7 | $2 | Not specified in the cited launch post | $6 | Starting rates; a fast variant doubles output speed and token prices. [71] |
| Gemini 4 Argon, **introductory** | $2 | $0.10 | $10 | Announced for a staged restricted rollout, not generally available at cutoff. Later base rates are $4/$20; no expiry date is stated. [82] |

**Practical interpretation:** count failed branches, repeated context,
reasoning/output charges, tools, verification, and human intervention. Cache
savings depend on actual reuse; architectural KV compression, billed cache hits,
and textual context compaction are **three different mechanisms**. A longer
context can increase cost and exposure to untrusted material. These are evaluation
recommendations, not measured September repository gains.

Embed 5 uses a different contract: $0.12/M text tokens for Pro and $0.08/M for
Fast, with image input at $0.40/M for either. Its shared embedding space can avoid
re-indexing when query traffic changes tier, but storage/search cost still depends
on dimensions, output format, corpus size, recall targets, and update frequency.
The generative-model table above should not be used to compare retrieval quality. [83]

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

The late-month APIs make compatibility testing equally important. `gpt-6-sol`
and `gpt-6.1-sol` are distinct IDs with different cache prices, reasoning modes,
and Chat Completions tool behavior; “upgrade Sol” is not a transparent alias
change. Claude Opus/Sonnet 5.5 can reject previously accepted forced-tool and
sampling requests, bind thinking history to a particular model, and move text
between tool calls into thinking blocks. A stream can therefore appear silent
without the request failing. Pin request/response fixtures and test errors,
streamed event ordering, tool-result continuation, cache reuse, and visible
progress before changing a production default. [76, 78–81]

Argon has no generally available model ID in the cited September 30 announcement.
Do not convert “rolling out soon” into an endpoint, price guarantee, or availability
claim. Conversely, Embed 5 Pro/Fast's shared space is an explicit compatibility
contract, but it still needs corpus-specific recall, dimension/quantization, and
mixed-tier migration tests before reusing a production index. [82, 83]

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

The **September 29 GPT-6.1 Sol addendum** classifies the model as Critical for
cybersecurity and High for biological/chemical capability under OpenAI's
Preparedness Framework and applies Astra's safeguard stack. It reports stronger
results than GPT-6 Sol on several safety sets, but also says displayed comparison
values for earlier models can reflect later reevaluations rather than their launch
values. Preserve evaluation date and model version; do not construct a clean
historical trend from mutable comparison columns. [81]

Anthropic reports its strongest automated behavioral-audit result for Opus 5.5,
external pre-release evaluation, and cyber safeguards/fallbacks for both 5.5
models; Sonnet keeps its predecessor's biology safeguards. Google describes
activation monitoring, action/chain-of-thought monitoring, sealed sandboxes, and
trusted-defender access for Argon. Those are different layers and disclosure
levels. None supplies independent detector false-positive/false-negative rates or
shows that monitoring alone blocks every consequential action. [75, 77, 82]

**Implication:** pair model evaluations with narrowly scoped permissions,
external action controls, incident-response procedures, and auditable outcome
checks. Do not substitute a cooperative safety benchmark or readable reasoning
trace for enforcement. Conversely, do not misattribute earlier-model incidents
to a new model merely because the disclosure appeared during its launch month.

### The AGI signal: broader work, still bounded evidence

The meaningful signal is broader long-horizon work across labs, increasingly
explicit architectures for agent state, and serious investment in verifiable
training environments and grader compute. MiMo shows reported gains across one
mixed RL run and held-out harnesses; Argon shows selected internal deployments
and unusually long trajectories; several hosted models reduce per-task cost. The
evidence still does **not** establish reliable open-ended autonomy, autonomous
recursive self-improvement, or AGI under an agreed operational definition.
Google's development account, DeepSeek's task-synthesis pipeline, and MiMo's
“self-improvement loop” remain researcher-designed processes with supplied tasks,
harnesses, graders, compute, and checkpoints. [61; 66, §§5–6; 73; 82]

August's separation of discovery, verification, novelty, and expert review still
applies. Lean certificates, wet-lab binding results, passing software tests,
composite indices, internal deployment anecdotes, and incident reports answer
different questions. Million-token context or output is a capacity ceiling, not
evidence that million-token work is coherent, economical, safe, or necessary.
Record the full attempted-task denominator, human interventions, safeguard
substitutions, authorization scope, and accepted outcome—not only the longest or
best trajectory. [58, 60, 66, 70, 73, 75, 77, 82]

---

## TL;DR — what we harvested

- **No new implementation or local model benchmark is claimed in this edition.**
  It updates the sourced report, evidence ledger, navigation, and roadmap;
  external results remain external results.
- **Carry forward August's foundations:** full-attempt provenance,
  candidate/verifier cost accounting, structured proof/checker boundaries, and
  independent-review release gates. Their prior status is documented in the
  [August edition](./2026-08-sota-llm-agi.md) and the
  [July/August design document](../frontier-platform/docs/15-july-august-2026-harvest.md).
- **Highest-ROI September additions:** model/API migration fixtures, visible-
  stream and forced-tool regression tests, mixed-tier retrieval evaluation, and
  completed-task cost ledgers that include reasoning, cache, tools, retries, and
  safeguard substitutions.
- **Scale-gated next work:** reduced-scale mixed-harness GRPO with grader
  provenance, expert-load telemetry/router-freeze ablation, and long-trajectory
  serving tests. MiMo's released 9B checkpoint is a research reference above the
  current 7B fine-tuning envelope, not authorization for a large run.
- **Control remains external:** monitor outputs, model narration, and readable
  reasoning do not replace permissions, sandboxing, independent outcome checks,
  incident response, or human approval for consequential actions.

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

MiMo-V2.6 Pro/Flash are also outside the stated envelopes: 42B active does not
remove the 1.02T-weight, MoE communication, omni-encoder, KV, and speculative-
decoder footprint. Its released 9B distill checkpoint is above the normal
`coder-finetune` ceiling and its reported 77.4B-token SFT stage is not a local
recipe. Use the artifacts for inspection and reduced-data experiments only after
explicit sizing and budget approval. [72–74]

---

## Tier 1 — high-ROI, broadly applicable

“Win” describes the intended benefit unless identified as measured above.
🟡 is a carried-forward partial foundation; 🔜 is a proposed next implementation.

| Technique | What it does | Win | Cost / risk | Min HW | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|--------|---------|-----------|
| **Safeguard- and version-aware eval contracts** | Pins model/revision or alias-resolution evidence, effort, task version, harness, tools, substitute model, and strict/partial scoring | Makes results interpretable across model and service changes | A provider may not expose immutable versions; substitutions can conceal which model did the work | CPU + model/API | [59, 60, 64, 67, 75–82] | 🔜 planned; prior task-audit contract is a foundation | coder-finetune; frontier-platform |
| **API migration and stream-contract fixtures** | Replays requests across model versions and asserts supported reasoning/sampling, tool-choice errors, event ordering, thinking continuity, and visible progress | Catches silent stream/UI regressions and incompatible tool workflows before rollout | Provider event schemas and aliases change; fixtures need redaction and pinned expected behavior | CPU + budgeted API | [76, 78–81] | 🔜 planned | coder-finetune; frontier-platform |
| **Completed-task cost ledgers** | Counts cached/uncached inputs, reasoning/output, retries, tools, verifiers, review, and tariff/time windows | Exposes the actual cost/quality frontier | Billing differs by provider; a fixed price-per-token snapshot is insufficient | CPU + metered runs | [59–62, 67–69, 71, 75–83] | 🟡 prior all-branch accounting; provider/cache integration planned | nanogpt-edu; midgpt; frontier-platform |
| **Shared-space retrieval tier evaluation** | Builds one versioned index, queries with quality/fast tiers, and measures recall, latency, storage, quantization, and re-index avoidance | Can route retrieval cost without rebuilding the corpus | Shared space is provider-specific; first-party scores do not predict corpus recall or access-control correctness | CPU + representative corpus/API or private model | [83] | 🔜 planned | frontier-platform |
| **Audited task/environment/verifier triplets** | Versions synthetic tasks, their environments, reference solutions, and checks; re-audits failed trajectories and difficulty | Targets useful training data rather than novelty in the optimizer alone | A flawed verifier teaches reward hacking; synthetic held-out sets can share generator biases | CPU tests + suitable small model | [66, §5.1; 73, §§4, 7] | 🔜 planned; prior task audits/provenance are not a task factory | nanogpt-edu; coder-finetune; frontier-platform |
| **Clarification and approval as explicit states** | Separates ambiguity, help requests, and permission for consequential actions | Makes interactive/unattended workflows easier to control | Waiting agents need escalation/timeouts; automatic approval defeats the boundary | CPU protocol tests | [58, 60, 62, 70] | 🔜 planned | coder-finetune; frontier-platform |
| **Outcome checks independent of narration** | Checks executable results, resource access, and authorization outside agent self-report | Reduces dependence on complete or faithful reasoning traces | Observation can miss failures; enforcement and incident response need separate controls | CPU sandbox + tests | [58, §§8–9; 70] | 🟡 prior verifier/release gates; authorization checks planned | nanogpt-edu; midgpt; coder-finetune; frontier-platform |
| **Reasoning-effort ablations** | Measures shallow/deep/no-thinking settings on the same held-out tasks with complete token/time costs | Finds effort levels with useful marginal gains | Effort labels/scalars are not calibrated across models; some models cannot disable thinking; maximum effort is not automatically optimal | small local model or API | [59, 61, 63; 66, §5.3; 76–81] | 🔜 planned | midgpt; coder-finetune |
| **Scientific artifact validity contracts** | Separates numerical equivalence, formal validity, experimental evidence, and novelty review | Prevents a local check from being called a validated discovery | Domain experts, lab access, or reproducible environments may dominate cost | CPU checks; domain-specific resources | [60] | 🟡 prior proof/review contracts; broader domains track | coder-finetune; frontier-platform |

---

## Tier 2 — scale- or hardware-gated wins

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| **CED + cross-layer sparse-cache reuse** | Separates prefill/decode compute and shares global KV/index states across layers | DeepSeek reports lower prefill active compute and smaller global KV | Requires architecture/training changes, compatible kernels, and quality checks; not a drop-in GPT-2 flag | [65; 66, §2] | track / ideal; reduced-scale research prototype first | midgpt; distgpt; frontier-platform |
| **FP4 cache + bounded replay** | Compresses global state and reconstructs recent sliding-window state instead of persisting everything | Reported global/persistent KV footprints of roughly ¼ / ⅛ of V4-Flash | Separate live-cache, persisted-cache, bandwidth, and replay measurements; test approximation errors and cache-resumption boundaries | [66, §§2–3, 6] | track; text compaction and exact training resume are different mechanisms | distgpt; frontier-platform |
| **Asynchronous multi-environment RL / OPD** | Reduces rollout stragglers; IBM uses separate pools, DeepSeek time-shares devices, and MiMo uses partial rollouts with reported staleness four | Enables higher concurrency for heterogeneous trajectories | Policy/log-prob provenance, length-bias control, staleness limits, backpressure, restart semantics, and scheduling; implementations are not interchangeable | [63; 66, §5.2; 73, §§5–6] | ideal; prior lease queue is not a full asynchronous trainer | distgpt; coder-finetune; frontier-platform |
| **Groupwise agentic grading + mixed-harness RL** | Compares passing trajectories, redistributes advantage, and varies harnesses within training | Can learn quality beyond binary pass/fail and reduce harness overfitting | Grader bias/reward hacking, 16-rollout groups, large token batches, and reliable environments; preserve raw verifier outcomes and held-out harnesses | [73, §§4–7] | track / ideal; start with a small fixed-budget ablation | distgpt; coder-finetune; frontier-platform |
| **MoE router freezing and expert-load telemetry** | Freezes routing during RL while tracking load CV, peak/mean load, cold experts, and micro-batch imbalance | MiMo reports stable loads where trainable routing collapsed | Requires an MoE model, expert-parallel telemetry, restartable runs, and an ablation; one vendor run does not establish a universal rule | [73, §§5.4–5.5] | track / ideal; reduced-scale ablation first | distgpt; frontier-platform |
| **Long-context, high-effort agent serving** | Couples large contexts, iterative tools, and adjustable reasoning budgets | Supports work requiring substantial accumulated state | API spend or inference fleet, KV capacity, latency, silent-stream handling, and untrusted-context isolation; include context surcharges and alias changes | [59, 61, 62, 67, 76, 78–82] | 🟡 prior compaction/budget protocols; end-to-end evaluation planned | midgpt; distgpt; frontier-platform |
| **Scientific GPU-kernel optimization** | Generates kernels and reuses intermediate results for repeated scientific workloads | Anthropic reports 1.4–2.5× H100 inference gains across seven models | Representative inputs, target GPU, numerical-equivalence checks, and kernel expertise; public optimization code still forthcoming in the cited source | [60] | track; no local reproduction | distgpt; frontier-platform |
| **Trusted-access defensive agents** | Combines stronger security capabilities with vetting and constrained execution | Extends defensive work while controlling access | Approved programs, isolated targets, independent approvals, and organizational accountability—not GPU capacity alone | [58, 60, 61, 70, 75, 77, 81, 82] | track / ideal; not an implementation of vendor access controls | frontier-platform |
| **Full-trajectory monitoring plus enforcement** | Combines behavior monitoring, action logs, resource isolation, and blocking checks | Adds defense beyond learned alignment | Monitoring compute, privacy/retention rules, evaluated detectors, incident response, and an external action boundary; no detector guarantees absence of misuse | [58, §§9–10; 70, 75, 77, 81, 82] | ideal; design target, not a shipped security guarantee | frontier-platform |

---

## Tier 3 — research bets (track, don't build yet)

- **Monitorability that scales with capability.** Use Astra's revised definitions:
  verbalized metagaming, actual oversight gaming, and monitor evasion are not
  interchangeable rates. Track independent replications and realistic
  false-positive/false-negative behavior, not only explicitly adversarial tests.
  **Harvest: track · frontier-platform.** [58, change log; §§8–9]
- **Agentic self-improvement with controlled evidence.** Google's development
  account, DeepSeek's synthetic task pipeline, and MiMo's groupwise-grading loop
  motivate better experiments, not a declaration of autonomous recursive
  improvement. Track fixed researcher/compute/grader budgets, environment and
  task audits, held-out tasks/harnesses, failures, and gains across successive
  iterations. **Harvest: track · nanogpt-edu; distgpt; coder-finetune;
  frontier-platform.** [61; 66, §5; 73, §§4–7; 82]
- **Robustness of approximate and extreme long-context state.** DeepSeek flags
  sparse-selection/replay boundary errors, while Argon's announced 1M-token output
  raises unmeasured coherence, recovery, review, and cost questions. Evaluate
  retrieval, eviction/resumption, replay frequency, stream recovery, cumulative
  tool errors, and accepted quality—not only bytes/token, maximum length, or one
  needle test. **Harvest: track · midgpt; distgpt; frontier-platform.** [66, §6; 82]
- **General-purpose autonomous science.** Bindings, maps, kernels, and proofs
  suggest breadth, but selected successes do not reveal the full attempted-task
  denominator or human contribution. Retain domain-specific checks and
  independent review. **Harvest: track · coder-finetune; frontier-platform.** [60]
- **Private enterprise oversight.** Anthropic's Enterprise Frontier Safeguards
  proposes customer-controlled storage and customer-led review. The retrieved
  announcement still says phased rollout starts later in the fall; it is not
  verified as generally shipped by this cutoff. Incident response and privacy
  governance remain separate requirements. **Harvest: track · frontier-platform.** [60, 70]
- **Open weights versus reproducible training.** Granite, V4.1-Flash, MiMo V2.6,
  and MiMo's 9B distill have released weights; Muse Spark's remain future work.
  Pin licenses, revisions, custom code, and artifacts, then separately assess data
  disclosure, compute scale, training recipes, environment/verifier release,
  inference support, and independent evaluation. MiMo discloses more of the RL
  stack than most releases, but a checkpoint plus first-party baselines still does
  not establish full reproduction. **Harvest: track · coder-finetune; distgpt;
  frontier-platform.** [62, 63, 65, 66, 68, 72–74]

---

## Roadmap by project

Carried forward from August, with September's next measurements added. These are
recommendations, not authorization to launch paid runs or implement production
safety systems as part of this documentation update.

| Project | Next harvest | Hardware tier | Notes |
|---------|--------------|---------------|-------|
| **nanogpt-edu** | Run the candidate/provenance harness with an external verifier and repeated seeds; compare binary reward with a tiny groupwise ranking under one fixed budget | minimal | Preserve the verifier-timing task. Count invalid tasks, failed rollouts, grader disagreements, and all generated tokens—not only the best accepted result |
| **midgpt** | Drive candidate search with a model/checker; compare context, compaction, and effort at matched accepted-task budgets and add interrupted-stream recovery tests | minimal→ideal | Preserve model-backed search. Separate textual compaction from architectural KV compression. A maximum context/output ceiling is not the target workload |
| **distgpt** | Attach real generation workers and CPU verifiers to the lease queue; add policy/log-prob versioning, staleness, restart, length-bias, host-memory, and expert-load telemetry | ideal | Choose separate-pool versus time-shared/partial-rollout execution explicitly. Run a reduced MoE router-freeze ablation before claiming MiMo-style stability or asynchronous RL |
| **coder-finetune** | Integrate a jailed Lean checker and held-out theorem set; extend eval provenance and add API migration fixtures for effort, forced tools, thinking continuity, safeguards, substitutes, and streams | minimal→ideal | Preserve exact-theorem and novelty boundaries. Granite 8B and MiMo Distill 9B exceed the stated 7B envelope; use them as references unless sizing is separately approved |
| **frontier-platform** | Add literature/reviewer backends, a corpus-specific Pro/Fast retrieval evaluation, model migration fixtures, approval enforcement, monitor evaluation, tariff provenance, and incident response | minimal→ideal | Keep authorization independent of model confidence and monitoring. Cache, retrieval-index, stream, and production-control implementations require corresponding design-doc updates |

---

## What shipped this month

- **This September 30 documentation refresh:** full-month coverage, 13 additional
  primary sources, an expanded evidence/cost ledger, MiMo method analysis,
  shipped-versus-preview labels, API migration hazards, and updated
  [edition-index](./README.md) / [repository README](../README.md) navigation.
- **No new training, serving, safety, retrieval, or evaluation implementation is
  claimed.** July/August protocol work remains prior work. This refresh involved
  source and document checks plus technical-report reading—not paid model runs,
  weight downloads, model execution, GPU benchmarks, Lean executions, or new
  subproject test-suite results.

---

## Sources

Sources continue the cumulative numbering from August (57). Sources [58–70]
were retrieved or rechecked September 13; sources [71–83] were retrieved or
rechecked September 30, 2026. Publication/release dates and revision dates are
distinguished from retrieval dates; mutable pages are not historical launch
snapshots.

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

71. SpaceXAI, *Introducing Grok 4.7*, **21 September 2026**. Hosted-model
    release, listed token prices, vendor benchmark comparisons, larger-base/longer-
    RL description, safety claims, and fast-variant terms. No weights or detailed
    reproducible training recipe are supplied by the announcement.
    <https://x.ai/news/grok-4-7>
72. Xiaomi MiMo Team, *MiMo-V2.6-Pro-RL* model card and released repository,
    **revision `73875d00b30a89ef8cc353a0b60b0e9f9561952d`**, inspected September
    30. MIT license, 1.02T-total/42B-active architecture, modality/context claims,
    model files, deployment examples, and first-party evaluation table. The model
    files were listed but not downloaded or executed.
    <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/README.md>
73. Xiaomi MiMo Team, *MiMo-V2.6: Scaling Reinforcement Learning Towards
    Self-Improvement*, **September 2026 technical report**, pinned to the Pro
    repository revision in [72]. §§4–7 support mixed-harness GRPO, GRS/GAR,
    router freezing, failure analysis, MOPD2, infrastructure, and released open
    foundations. Reported results and causal interpretations remain first-party.
    <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/73875d00b30a89ef8cc353a0b60b0e9f9561952d/MiMo_V2_6_technical_report.pdf>
74. Xiaomi MiMo Team, *MiMo-V2.6-Distill-Qwen-9B* model card and released
    checkpoint, **revision `2367e865d009c13ac81713a2878291d33ab28177`**,
    inspected September 30. MIT license, Qwen3.5-9B base, 77.4B-token weighted
    SFT mixture, internal/public evaluation labels, and SGLang quickstart. Weight
    files were listed but not downloaded or executed.
    <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/2367e865d009c13ac81713a2878291d33ab28177/README.md>
75. Anthropic, *Introducing Claude Opus 5.5*, **22 September 2026**. Capability,
    price/speed, external-evaluator, safety, effort, and benchmark claims,
    including safeguard fallback/substitute-model footnotes and Anthropic's own
    warning that benchmark gaps can overstate practical differences.
    <https://www.anthropic.com/news/claude-opus-5-5>
76. Anthropic, *Claude Opus 5.5 — Overview*, **live platform documentation,
    September 30 snapshot**. Model ID, 1M/128K limits, prices, release/lifecycle,
    always-on thinking, and documented request/response breaking changes.
    <https://platform.claude.com/docs/en/models/opus-5-5/overview>
77. Anthropic, *Introducing Claude Sonnet 5.5*, **28 September 2026**. First-party
    benchmark, speed, completed-task cost, safeguard, and availability claims,
    with explicit caveats about Opus retaining an advantage on open-ended work.
    <https://www.anthropic.com/claude-sonnet-5-5>
78. Anthropic, *Claude Sonnet 5.5 — Overview*, **live platform documentation,
    September 30 snapshot**. Model IDs, 1M/128K limits, cache/base prices,
    thinking/sampling behavior, lifecycle, and five documented breaking changes.
    <https://platform.claude.com/docs/en/models/sonnet-5-5/overview>
79. OpenAI API documentation, *Using GPT-6*, *GPT-6 Sol*, and *GPT-6 Luna*,
    **live pages, September 30 snapshot**. Family selection, Sol supersession,
    reasoning/tool restrictions, model limits, token/cache prices, surcharges,
    supported endpoints, and exposed snapshots.
    <https://developers.openai.com/api/docs/guides/latest-model.md>
    <https://developers.openai.com/api/docs/models/gpt-6-sol.md>
    <https://developers.openai.com/api/docs/models/gpt-6-luna.md>
80. OpenAI API documentation, *GPT-6.1 Sol*, **live model card, September 30
    snapshot**. Model ID, 1,050,000 context, 922,000 maximum input, 128,000
    maximum output, effort/tool rules, prices, surcharges, endpoints, and the
    sole exposed snapshot.
    <https://developers.openai.com/api/docs/models/gpt-6.1-sol.md>
81. OpenAI, *Addendum to GPT-6 Astra System Card: GPT-6.1 Sol*, **29 September
    2026**. Critical-cyber and High-bio/chemical classifications under OpenAI's
    framework, safety/robustness/alignment evaluations, safeguards, and warning
    that displayed prior-model values can differ from their launch values.
    <https://deploymentsafety.openai.com/gpt-6-1-sol>
82. Google, Koray Kavukcuoglu, *Gemini 4 Argon: our next era of frontier
    intelligence*, **30 September 2026**. Restricted Fairwind/trusted-tester
    rollout, 1M-output claim, introductory/later prices, selected benchmark and
    internal-deployment results, monitoring/sandbox descriptions, and future-
    tense broad availability. This is an announcement, not a general API card.
    <https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/>
83. Cohere, *Introducing Embed 5—A new family of frontier embedding models*,
    **30 September 2026**. General availability, Pro/Fast shared embedding space,
    multimodal 128K input, languages, dimensions, output formats, self-hosting,
    prices, and first-party retrieval evaluations.
    <https://cohere.com/blog/embed-5>

> **Methodology and coverage limits.** This full-month edition preserves the
> September 13 primary-source review, reads the late-month announcements, live
> model/API documentation, revision-pinned model cards, and relevant MiMo report
> sections, and retains AA's independent DeepSeek snapshot. Search and secondary
> release trackers were used for discovery and calendar coverage, not as evidence
> for quoted technical claims. There is still no matched cross-vendor experiment
> or universal winner. Threat cases remain selected provider observations;
> internal deployments and vendor benchmarks remain first-party. Model-weight
> listings were inspected but not downloaded or executed. A technical-report PDF
> was downloaded only for reading. Launch notices, staged availability, pricing,
> aliases, API behavior, evaluator indices, and system cards can change; dates,
> revisions, benchmark/harness versions, effort, safeguards, and substitutions are
> part of the evidence. Missing items mean “not verified in this survey,” not
> “nothing else happened.” Independent replication remains future work.
