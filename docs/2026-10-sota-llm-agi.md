# SOTA Watch — LLM & AGI · 2026-10

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-11-11  ·  **Status:** draft — retrospective full-month edition, October 1–31

> **The frontier split between cheaper specialists and more ambitious agents.**
> Haiku 5.5 and GPT-6.1 Terra lowered the price of repetitive and computer-based
> work. Liquid's LFM3 put hybrid sequence modeling and multi-domain post-training
> into a roughly 2B-parameter family. Mistral Large 4 moved from API preview to
> downloadable weights, while Kimi K3.1 Pro launched hosted access with weights
> promised for November. At the other end of the spectrum, Gemini 4 Carbon and
> Genie 4 previewed longer-horizon reasoning and interactive world simulation.
> The practical question is no longer just which model scores highest, but which
> combination of model, effort, context, tools, and deployment terms completes
> a particular task reliably at an acceptable total cost.

**Scope and evidence:** selected October developments, researched November 11;
not an exhaustive catalog or a reconstructed historical leaderboard. Dated
announcements establish release chronology. Model cards and licenses inspected
in November are labeled as retrospective artifact checks, not October snapshots.
Performance, training, and safety results below are **first-party reports** unless
explicitly stated otherwise. A provider quoting an independent benchmark is not
an independent reproduction by this repository. No weights were downloaded or
executed. Search became unavailable during research; direct provider pages and
public model-repository APIs supplied the cited evidence. Source numbering
continues from [September](./2026-09-sota-llm-agi.md), which ended at 83.

## The frontier this month

- **Haiku 5.5 makes narrow agent work substantially cheaper.** Anthropic's
  October 7 release adds adjustable effort to its small-model line and targets
  compaction, extraction, classification, browser use, and subagents. Standard
  input/output rates are $0.10/$0.50 per million tokens for prompts up to 100K,
  but $0.50/$2.50 above that threshold. Anthropic also halves Sonnet 5.5 cache-read
  pricing to $0.10/M. Its reported roughly 75% lower Haiku running cost is an
  average over its workloads, not a guarantee for any agent. Anthropic still
  recommends larger models for complex coding. [84]
- **GPT-6.1 Terra advances the cost–quality frontier for computer use.** OpenAI's
  October release combines visual interpretation, application interaction, and
  writing at standard rates of $1/M input, $0.05/M cached input, and $5/M output.
  It reports 50.3% on Terminal-Bench 4.0 at maximum effort, versus GPT-5.6 Terra's
  published 21.5%. Comparisons with Astra, Sol, and Opus use particular tasks,
  effort settings, and token counts; they do not establish universal superiority.
  OpenAI also classifies Terra as Critical in cybersecurity under its own
  framework: lower inference cost does not imply lower deployment risk. [85]
- **Mistral Large 4 crosses from promised openness to released weights.** The
  October 6 preview becomes a weight release on October 29. The approximately
  1T-total MoE has 49B active parameters per token, or 52B including embeddings
  and output layers. The released card describes 1M context, text/image input,
  adjustable reasoning, and an embedded speculative draft head. The Mistral
  Ecosystem Open-Weight License requires a separate license for specified
  model-as-a-service commercial use; this is not an unrestricted MIT release.
  Its coding and security results are interesting without implying leadership
  on every general-intelligence aggregate. [86–88]
- **LFM3 brings a concrete small-model architecture and specialization recipe.**
  Liquid's October 27 family uses a 1.7B text backbone with 13 gated convolution
  layers, five 256-token sliding-window-attention layers, and six global-attention
  layers. It reports 22T pretraining tokens, including a final 3.5T mid-training
  stage, followed by SFT, mixed-domain RL, multi-domain on-policy distillation,
  and agentic RL. Six roughly 2B-class derivatives cover text, vision, audio,
  omni, and decision workloads. This is a useful counterpoint to trillion-scale
  MoE: small deployed size can still require enormous training investment. [89, 90]
- **Kimi K3.1 Pro emphasizes the entire RL system, not just an objective.**
  Moonshot's late-October announcement describes joint RL across hundreds of
  downstream domains and improvements to rollout scheduling, inference,
  sandboxes, and environment quality checks. It reports 65.7% on Terminal-Bench
  4.0 in its comparison and makes hosted models available, while explicitly
  promising full weights by November 4. October therefore gets an API launch,
  not a retroactively assigned weight release; the artifact check belongs in
  [November](./2026-11-sota-llm-agi.md). [91]
- **Gemini 4 Carbon and Genie 4 extend the research frontier, with restricted
  access.** Google's October 28 Carbon announcement targets sustained agentic
  and security work, initially for selected defenders and trusted partners.
  Several cyber results are explicitly evaluated without guardrails. October 27's
  Genie 4 combines interactive visual simulation, generated audio, dialogue,
  and language-driven scene changes, also as a trusted-tester research preview.
  Neither announcement establishes general public access, physically accurate
  simulation, or broadly reliable autonomous operation by October 31. [92, 93]
- **Evaluator access becomes part of the safety story.** Anthropic's October 15
  announcement says METR is receiving employee-like access and will publish risk
  assessments. This is a change in the evaluation process, not a published finding
  that the models are safe. Access, independence, publication rights, and the
  actual results all need separate scrutiny. [94]

### Release and evidence ledger

| Development | October status supported by the source | Important boundary |
|-------------|-----------------------------------------|--------------------|
| Haiku 5.5 | Released October 7 on Claude and partner platforms | Prompt-length pricing tiers; small does not mean equally capable at long coding tasks [84] |
| GPT-6.1 Terra | Released through API, ChatGPT Work, and Codex in October | First-party task-cost comparisons; no local trial [85] |
| Mistral Large 4 | October 6 API preview; October 29 weight release | Custom license; later pinned card is a November inspection [86–88] |
| LFM3 | October 27 model-family release with weight links | Custom LFM license; runtime memory exceeds weight storage [89, 90] |
| Kimi K3.1 Pro | Hosted release; weights announced for November | Do not use a repository's creation date as proof of public availability [91] |
| Gemini 4 Carbon | October 28 limited defender/trusted-partner rollout | Public launch occurs in November; unguarded cyber evaluations are not public-endpoint results [92] |
| Genie 4 | October 27 trusted-tester research preview | Interactive examples are not a validated general-purpose physics or RL simulator [93] |

### Methods worth extracting from the announcements

**Hardware-aware hybrids are an empirical design choice.** LFM3's architecture
search covered 14 device/backend combinations. Its mix of short convolutions,
local attention, and global attention reduces the amount of full-context state,
while preserving some global information paths. The right comparison is not
“hybrid always beats Transformer.” Compare quality, prefill/decode latency, and
memory at the same context length, precision, backend, and workload. Liquid's
roughly 1 GB four-bit text-model claim is not the memory footprint of a complete
256K-context application: its own blog separately reports 1.6 GB of KV cache at
128K. Training activations, optimizer state, and multimodal inputs require their
own budgets. [89, 90]

**Specialist distillation and joint RL solve related but different problems.**
LFM3 describes six specialist RL teachers distilled back into a single model;
Kimi describes scaling joint RL across hundreds of domains. Mistral attributes
its post-preview improvements to additional RL and feedback. These are recipes
and first-party before/after observations, not controlled cross-lab ablations.
For a smaller reproduction, keep a fixed supervised baseline, equalize generated
and graded tokens, measure held-out domains, and record regressions as well as
wins. More RL is not a substitute for trustworthy tasks and graders. [87, 89, 91]

**Speculative decoding needs its own acceptance and cost measurements.**
Mistral's released card describes DSpark: three dense MLA draft layers in BF16
that propose a three-token block, with a reported *up to* 1.9× throughput gain.
This is an inference mechanism, not the same intervention as adding a
multi-token-prediction training loss. Record draft acceptance, verification cost,
concurrency, latency distribution, extra memory, and output-distribution checks.
The card's eight-way serving examples are not a guarantee that any eight GPUs
will fit the checkpoint and a million-token workload. [88]

**Measure completed-task economics, not the cheapest token tariff.** Haiku's
long-prompt threshold, Sonnet's cheaper cache reads, Terra's effort-dependent
comparisons, and Carbon's introductory pricing all change the cost calculation.
Log uncached input, cache reads and writes where applicable, all reasoning/output
usage, retries, tool costs, parallel branches, and failed attempts. Compare the
sum against independently accepted outcomes; a shorter unsuccessful trajectory
is not an efficiency win. Carbon's announced $2/$10 introductory rates are not
an October generally available service price. [84, 85, 92]

### What the evidence does not establish

- A single overall winner. Terminal-Bench versions, private coding benchmarks,
  internal tasks, effort settings, and safeguard configurations differ. Preserve
  those labels rather than combine the percentages into an invented ranking.
- Fully reproducible frontier training. Weight release does not provide the
  complete data mixture, compute, environments, filtering, and training history.
- General-purpose AGI or autonomous science from curated demonstrations. Genie
  scenes and multi-step coding examples motivate tests of transfer, recovery,
  causal consistency, and failure rates; they do not answer them. [91–93]
- Guaranteed security from a refusal or attack-resistance percentage. Mistral
  explicitly reports concern about real-attack uplift and further guardrail work;
  privileged tool execution still needs an independent authorization boundary. [87]

## TL;DR — what we harvested

- **✅ Documentation:** a sourced October comparison, release-status ledger,
  hardware-aware technique map, and five updated project roadmaps.
- **🔜 Proposed experiments:** narrow-task routing, accepted-task cost accounting,
  small hybrid ablations, and specialist-versus-joint post-training comparisons.
  These are recommendations, not new implementations or scheduled runs.
- **Track / ideal:** trillion-scale MoE, DSpark, native multimodal agents, world
  simulation, and restricted cyber models. No local capability, speed, or safety
  result is claimed.

## Hardware envelopes per project

These are planning envelopes for meaningful work at each project's own scale,
not measured fit guarantees or permission to launch jobs. Sequence length, batch,
optimizer, precision, and checkpointing must be sized separately. Full Mistral and
Kimi checkpoints exceed `distgpt`'s 70B scope. They are relevant to the
`frontier-platform` blueprint's open-ended 500B+ tier, but need separately sized
trillion-parameter deployments; smaller projects should study reduced analogues.

| Project | Scale | Minimal | Ideal | Unlocks at ideal |
|---------|-------|---------|-------|------------------|
| nanogpt-edu | 10M–100M | CPU smoke tests; one 8–16 GB GPU for useful short-context runs | One 24–48 GB GPU | Repeated-seed, matched-budget sequence-block and reward ablations |
| midgpt | 124M–1.5B | One 16–24 GB GPU, using a smaller model or checkpointing | 4–8 × 48–80 GB GPUs | Longer-context DDP runs and quality/latency/memory sweeps |
| distgpt | 1B–70B | One 24–48 GB GPU for reduced tests; 2–4 GPUs for meaningful sharding tests | Multi-node 8×80 GB-class nodes with fast interconnect and storage | TP/PP/FSDP comparisons, reduced MoE, rollout/verifier pools |
| coder-finetune | 0.5B–7B | One 16–24 GB GPU for bounded-context LoRA/QLoRA | 4–8 × 80 GB GPUs plus isolated CPU evaluators | Full fine-tuning, multi-domain post-training, repeated evaluations |
| frontier-platform | 1B–500B+ blueprint | CPU design/contract checks; small GPU/API sandbox for future integration | Multi-node accelerator cluster, separate serving/verifier pools, storage and observability | Architecture-scale MoE, multimodal systems, independent evaluation and enforced approvals |

## Tier 1 — high-ROI, broadly applicable

Here **🔜 planned** means a recommended next harvest, not a committed delivery;
**track** means no implementation claim. All wins remain hypotheses locally.

| Technique | What it does | Win | Cost / risk | Min HW | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|--------|---------|-----------|
| Narrow-task model routing | Assigns extraction/compaction to a small model and escalates difficult cases | Potentially lower accepted-task cost | Routing errors; larger model may still be cheaper after retries | CPU/API client or small local model | [84, 85] | 🔜 planned | coder-finetune; frontier-platform |
| Effort/cache-aware cost ledger | Separates cache, reasoning, retries, and parallel attempts | Makes price/performance comparisons meaningful | Provider usage fields and tariffs can change | CPU | [84, 85, 92] | 🔜 planned; docs ✅ | midgpt; coder-finetune; frontier-platform |
| Small hybrid sequence-block ablation | Compares local/global attention and convolution under matched budgets | Tests memory and long-context trade-offs | Kernel support; quality can regress | One GPU at reduced scale | [89, 90] | 🔜 planned | nanogpt-edu; midgpt |
| Held-out multi-domain grading | Evaluates mixed-RL or specialist-distilled policies beyond training tasks | Detects narrow optimization and regressions | Grader bias, contamination, and synthetic-task validity | One GPU plus CPU checkers | [87, 89, 91] | 🔜 planned | nanogpt-edu; coder-finetune |
| Artifact and license manifest | Pins weights, tokenizer, parsers, precision, terms, and release evidence | Avoids false reproducibility and deployment assumptions | Metadata inspection is not execution or legal approval | CPU | [88, 90, 91] | ✅ report pins; runtime integration planned | distgpt; coder-finetune; frontier-platform |

## Tier 2 — scale- or hardware-gated wins

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| Speculative draft heads | Proposes blocks for target-model verification | Mistral reports up to 1.9× throughput | Compatible backend, draft memory, target capacity, favorable acceptance/concurrency | [88] | ideal; no reproduction | midgpt; distgpt; frontier-platform |
| Large granular MoE | Activates a fraction of a much larger parameter set | More stored capacity per active computation | Total expert memory, interconnect, dispatch balance; active parameters are not memory size | [86–88] | ideal; reduced analogues first | distgpt; frontier-platform |
| Joint long-horizon RL infrastructure | Coordinates varied tasks, sandboxes, rollout scheduling, and graders | Potentially broader agent competence | Worker pools, versioned policies, trusted environments, large generation budgets | [89, 91] | ideal | distgpt; coder-finetune; frontier-platform |
| NVFP4 serving | Uses the separately released quantized Mistral checkpoint | Smaller weight footprint; possible throughput improvement | Blackwell-targeted implementation and task-level quality checks | [88] | ideal; no local speed claim | distgpt; frontier-platform |
| Trusted-access cyber evaluation | Evaluates advanced models in controlled authorized environments | Better evidence about capabilities and mitigations | Approval, isolation, evaluator access and disclosure process—not hardware alone | [92, 94] | track | frontier-platform |

## Tier 3 — research bets (track, don't build yet)

- **Interactive world models as training environments.** Genie 4's controllable
  audiovisual scenes suggest richer experience generation, but need measurements
  of causal consistency, repeatability, task reward validity, and sim-to-real
  transfer before becoming trusted agent-training data.
  **Harvest: track · frontier-platform.** [93]
- **Stable cross-domain scaling of agentic RL.** Kimi's system-level account and
  Liquid's specialist-distillation recipe motivate controlled comparisons, not a
  claim that either is universally preferable. Measure transfer and forgetting
  at fixed data, generation, and grading budgets.
  **Harvest: track · nanogpt-edu; distgpt; coder-finetune.** [89, 91]
- **Multi-day autonomy with independent safety evidence.** Carbon's internal
  long-running examples and Anthropic's evaluator-access changes are important
  signals. They still need public results, realistic tasks, restart tests, and
  approval controls outside the model.
  **Harvest: track · midgpt; frontier-platform.** [92, 94]

## Roadmap by project

Carried forward from September. Earlier verifier, provenance, Lean, and durable
queue work remains on the roadmap; new announcements do not establish that it
has been integrated. No paid runs or implementation work is authorized by this
report.

| Project | Next harvest | Hardware tier | Notes |
|---------|--------------|---------------|-------|
| **nanogpt-edu** | Keep the external-verifier/repeated-seed experiment; add one small sequence-block or reward ablation at a time | minimal | Use equal training and grading budgets; count failed candidates, not only selected successes |
| **midgpt** | Preserve model-backed candidate search and interrupted-stream recovery; compare context, compaction, and one hybrid baseline | minimal→ideal | Measure accepted quality alongside prefill/decode and memory; textual compaction is not KV compression |
| **distgpt** | Attach real workers and verifiers to the lease queue; add policy-version, staleness, length, and expert-load telemetry before mixed RL | ideal | Separate training, rollout, and grading capacity; use reduced MoE rather than silently importing a 1T checkpoint into the 70B track |
| **coder-finetune** | Retain jailed Lean/held-out theorem integration; evaluate an in-envelope specialist baseline and multi-domain grading | minimal→ideal | LFM3-1.7B fits the parameter envelope, subject to backend/license review; Mistral and Kimi do not |
| **frontier-platform** | Extend migration, tariff, license, retrieval, reviewer, and approval contracts with dated release-status evidence | minimal→ideal | Keep execution authorization independent of model confidence; implementation changes still require matching design-doc updates |

## What shipped this month

- **This retrospective documentation update, written November 11:** the missing
  October edition, primary-source references 84–94, revision-pinned model-card
  and license checks, and index navigation. It is not backdated implementation work.
- **No new model training, serving, RL, routing, safety, or evaluation code.** No
  paid API calls, weight downloads, model execution, GPU benchmarks, or subproject
  test-suite results are claimed. “Planned” and “ideal” rows are recommendations.

## Sources

Sources continue September's cumulative numbering. All were inspected November
11, 2026. Announcements establish their own dates; live pages may have changed
since publication. Pinned cards below were inspected after October and are not
presented as historical October snapshots. Technical-report PDFs linked by model
cards were not independently reviewed for this edition.

84. Anthropic, *Introducing Claude Haiku 5.5*, **October 7, 2026**. Release,
    effort, prompt-length tariffs, Sonnet cache-price change, and first-party
    capability/safety comparisons.
    <https://www.anthropic.com/claude-haiku-5-5>
85. OpenAI, *Introducing GPT-6.1 Terra*, **October 2026 release**. Visual and
    computer-use tasks, effort-dependent comparisons, standard pricing,
    availability, and provider safety classification. The retrieved article body
    did not expose a day-level publication date; none is inferred here.
    <https://openai.com/index/introducing-gpt-6-1-terra/>
86. Mistral AI, *Introducing Mistral Large 4*, **October 6, 2026**. API preview,
    architecture summary, training infrastructure, and a future weight-release
    commitment—not proof of weights on that date.
    <https://mistral.ai/news/mistral-large-4/>
87. Mistral AI, *Mistral Large 4* weight-release announcement, **October 29, 2026**
    per publication metadata; page modified November 5. Weight release,
    additional RL, benchmark claims, and guardrail qualifications. Read
    retrospectively, not as an archived October page.
    <https://mistral.ai/news/mistral-large-4-oss/>
88. Mistral AI, *Mistral-Large-4-1T-A52B* model card and license, **revision
    `af26e7869af601441d5cdf75a1272469ddc3d059`**, inspected November 11.
    Active-parameter counting convention, DSpark, serving examples, NVFP4 sibling,
    and model-as-a-service terms. File listings inspected; weights not downloaded.
    <https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/af26e7869af601441d5cdf75a1272469ddc3d059/README.md>
    <https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/af26e7869af601441d5cdf75a1272469ddc3d059/LICENSE>
89. Liquid AI, *Introducing LFM3: Every Modality for Specialized Intelligence*,
    **October 27, 2026**. Architecture, hardware-aware search, pretraining and
    post-training stages, model-family release, and first-party inference results.
    <https://www.liquid.ai/blog/lfm3>
90. Liquid AI, *LFM3-1.7B* model card and LFM license, **revision
    `1da06bec6a819da10b54c67e5c90972b46d81871`**, inspected November 11.
    Current artifact/configuration and custom-license evidence, including a
    commercial-use revenue threshold; consult the complete terms before deployment.
    <https://huggingface.co/LiquidAI/LFM3-1.7B/blob/1da06bec6a819da10b54c67e5c90972b46d81871/README.md>
    <https://huggingface.co/LiquidAI/LFM3-1.7B/blob/1da06bec6a819da10b54c67e5c90972b46d81871/LICENSE>
91. Moonshot AI, *Kimi K3.1 Pro* announcement, **late October 2026**; retrieved
    metadata reports an October 24 update. Hosted availability, joint-RL systems
    account, first-party benchmarks, and explicit promise of weights by November 4.
    <https://www.kimi.com/en/blog/kimi-k3-1-pro>
92. Google, *Gemini 4 Carbon: our next frontier model*, **October 28, 2026**.
    Limited rollout, introductory pricing, long-horizon examples, and benchmark
    conditions including unguarded cyber evaluations.
    <https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/>
93. Google DeepMind, *Genie 4: A new generation of world simulation*, **October
    27, 2026**. Audiovisual interaction, language-driven control, demonstrations,
    and trusted-tester research-preview access.
    <https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/>
94. Anthropic, *An update on independent evaluators*, **October 15, 2026**, with
    subsequent clarifications. Evaluator access and publication arrangements;
    not an independent evaluator's results report.
    <https://www.anthropic.com/news/an-update-on-independent-evaluators>
