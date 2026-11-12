# SOTA Watch — LLM & AGI · 2026-11

**Editor:** LLM-playground maintainers  ·  **Published:** 2026-11-11  ·  **Status:** draft — month-to-date through November 11

> **Frontier capability is becoming a question of access, verification, and
> orchestration—not just model selection.** Claude Fable/Mythos 5.2 raises the
> ambition of coding and scientific work; Gemini 4 Carbon moves from selected
> defenders to paying developers and broader product access. Kimi K3.1 Pro now
> has inspectable weight artifacts, while Inkling 2 Small exposes a detailed
> training account but keeps its promised open weights pending safety testing.
> GPT-6 Cyber makes the distinction between ordinary API availability and
> verified security access explicit. Across these developments, the recurring
> technical themes are mixed-domain RL, long-lived multi-agent work, formal
> checking, and the need to account for every attempt and every safeguard.

**Cutoff and evidence:** November 1–11 only; this edition does **not** forecast
or summarize the rest of November. It is a selected frontier survey, not a
complete release list or a comparable cross-provider leaderboard. Announcements,
model-card metadata, and license text were inspected directly. Unless explicitly
stated otherwise, benchmarks, scientific outcomes, training details, and safety
rates are **provider-reported**, including independent-evaluator results quoted
by providers. No cited frontier-model result was reproduced here; local toy
experiments are documented separately below. Search became unavailable
partway through research; cited primary pages remained accessible. Source
numbering continues from [October](./2026-10-sota-llm-agi.md), ending at 94.

## The frontier this month

- **Claude Fable 5.2 and Mythos 5.2 target the hardest technical work.**
  Anthropic's November 9 release describes the same underlying model with
  different cyber/biology safeguards and access requirements. Standard pricing
  is $8/M input and $40/M output; optional fast mode is $16/$80. Anthropic reports
  gains in coding, vision, and scientific tasks, but still recommends Opus 5.5
  as the starting point for most everyday work. Its benchmark table distinguishes
  guarded Fable, reduced-safeguard Mythos, effort levels, and task-specific
  scoring. Those qualifications are part of the result, not fine print. [95]
- **Gemini 4 Carbon becomes available beyond the October defender preview.**
  Google's November 10 announcement opens API access to paying developers and
  enterprises and rolls out product access to AI Ultra subscribers. Introductory
  public-model pricing is $2/M input and $10/M output through November 30,
  followed by $4/$20. Google also previews Antigravity's `/swarm` orchestration
  and introduces a jointly developed Multimodal Knowledge Agent benchmark.
  Public Carbon access, the `/swarm` preview, and privileged Fairwind cyber
  access remain different products and permission boundaries. [96]
- **Kimi K3.1 Pro moves from a weight promise to inspectable artifacts.**
  The November 11 repository check lists 96 safetensors shards and a model card
  specifying 2.8T total / 104B active parameters, 1M context, and 69 KDA plus
  24 gated DSA attention layers. The custom license includes conditions for
  large model-as-a-service businesses and additional branding conditions.
  Downloadable does not mean cheap to host, unrestricted, or fully reproducible.
  This verifies availability by our cutoff, not the exact hour the repository
  became public or whether every inference path works. [97]
- **Inkling 2 Small provides a substantive training account, not yet open
  weights.** Thinking Machines' November 5 release describes a 594B-total /
  17B-active image-capable MoE with variable effort, initially on Tinker. It
  reports native NVFP4 pretraining and a unified RL run spanning coding,
  documents, vision, tools, and safety. Apache-2.0 weights are announced for after
  external safety testing; the inspected announcement still presents that as
  future work. “Small” describes the active-compute tier, not a 17B storage or
  full-training footprint. [98]
- **GPT-6 Cyber couples specialized post-training to restricted deployment.**
  OpenAI's November 5 system card describes a cyber-specialized Astra variant
  with reduced refusals on authorized security work. Access is restricted through
  Daybreak Red, with identity, organizational-security, and use restrictions.
  OpenAI rates it Critical in cyber, High in biological/chemical capability,
  and below High in AI self-improvement under its framework. The card also shows
  that greater task persistence can accompany regressions on particular
  restriction-following or reporting tests. Capability gain is not a blanket
  improvement in agent safety. [99]
- **Formal verification and scientific workflows become stronger evidence—but
  only inside their actual scope.** Anthropic reports a 30-hour Fermat
  formalization by roughly 1,000 Fable agents, with supplied statements,
  definitions, prior lemmas, and a single run per compared model. Google reports
  Carbon-assisted proofs of key properties of code underlying Signal's protocol,
  using other models and human researchers too. These are compelling accounts
  of assisted verification, not new proofs of Fermat's theorem, certification of
  all of Signal, or evidence that arbitrary research can run without oversight.
  The underlying proof artifacts were not checked in this repository. [95, 96]

### Access and artifact ledger

| Model / development | Verified status by November 11 | What must not be inferred |
|---------------------|--------------------------------|---------------------------|
| Claude Fable 5.2 | November 9 public-facing release with strong safeguards | Same accessible capabilities as Mythos; every task benefits from the premium model [95] |
| Claude Mythos 5.2 | Same underlying model; access through verification programs | General availability of reduced-safeguard cyber/biology behavior [95] |
| Gemini 4 Carbon | November 10 paid developer/enterprise API access; Ultra product rollout | Every product/account already has access, `/swarm` is fully released, or Fairwind access is included [96] |
| Kimi K3.1 Pro | Public weight-file listing, pinned model card and license inspected | Exact first-public date, local fit, executable compatibility, or unrestricted commercial rights [97] |
| Inkling 2 Small | Tinker inference/fine-tuning announced; open weights still future tense in inspected source | Apache-2.0 downloadable weights at this cutoff [98] |
| GPT-6 Cyber | Restricted Daybreak Red deployment documented | Ordinary API entitlement or permission to test third-party systems [99] |

### The technical signal: heterogeneous RL and deployment-aware training

**Inkling discloses where the work goes.** Its account reports an initial 8.4T-token
pretraining phase over 68 hours on GB300 NVL72 racks using native NVFP4, followed
by 3.8T higher-quality tokens at 512K context, then approximately 320B and 34B
trajectory-rich stages. The 68 hours refer to the first phase, not the complete
training run, and the post does not give a basis for a cross-lab training-cost
ranking. “Base-recovered” comparisons use additional fine-tuning of other labs'
post-trained models; they are not comparisons against those labs' original base
checkpoints. [98]

The reported joint RL run uses 2,048 GPUs over nine days, with 1,075 steps,
over 17 million rollouts, and over 3T tokens. Coding accounts for 38.5% of rollouts
but 65.0% of tokens: task count is not compute share. Different tasks use hidden
unit tests, programmatic checks, reference answers, rubric judges, or pairwise
screenshot reward models. Randomized effort conditioning and a length penalty
train speed/thoroughness trade-offs. The asynchronous system reports p50/p99
rollout durations of eight minutes/four hours and uses a score-centering
StableReinforce objective. These are first-party descriptions, not an available
turnkey reproduction of the full run. [98]

**The useful lesson is to budget by domain, duration, and evaluator—not only
samples.** Mixed long/short workloads create policy staleness and worker
utilization problems. A credible reduced experiment needs policy/log-prob
versioning, per-domain token and wall-time accounting, grader versioning,
independent holdouts, restart handling, and refusal/format-failure accounting.
One aggregate reward can hide degradation in a smaller domain. Inkling's own
forecasting demonstration records an RL branch that became overconfident and
performed worse, after which the agent returned to SFT. Retaining those failed
branches is better evidence than reporting only the selected final checkpoint.
These are editorial experimental recommendations, not claims about local code. [98]

**Kimi illustrates why sparse compute is not small memory.** Its 2.8T total
parameters remain the relevant starting point for weight storage even though
104B are active for a token. The hybrid KDA/gated-DSA architecture also requires
backend support and different state/cache accounting from a dense GPT baseline.
Do not estimate a deployable service from active parameter count alone: include
precision, quantization metadata, all experts, attention state, activations,
communication buffers, concurrency, and host memory. This is an ideal-scale
reference, not a new local target for the 0.5B–7B fine-tuning track. [97]

### Scientific evidence: proof, experiment, and novelty are different checks

Anthropic's Fermat account is **formalization of an established result**. Its
footnote describes supplied theorem statements, definitions, a one-sentence
classical-route sketch, and 47 previously proved lemmas. The plotted comparison
uses roughly 1,000 agents and a single run per model. The reported 30-hour
elapsed time therefore must not be read as single-agent compute, total cost,
independent discovery, or a repeated-trial estimate. A reproduction would pin
the theorem, imported axioms/lemmas, Lean/toolchain versions, proof artifacts,
full attempt tree, and total resource use. This edition inspected the report,
not those proof objects. [95]

Google's Signal example similarly describes a research system, **AlphaProof
Code**, using Carbon alongside other models to prove key security properties.
The claim is scoped to the formalized goals and code, not the entire deployed
messaging system, implementation supply chain, or operating environment.
Specification correctness and threat-model coverage remain separate questions
from whether a proof checker accepts a theorem. [96]

For empirical science, the verification boundary changes again. Anthropic
reports 28 successful capsid models among 34 attempts; Google reports homolog
retrieval improvements across seven low-homolog protein families with broader
benchmarking still underway. These selected results are not clinical validation,
general protein-design performance, or evidence of complete experimental
autonomy. Distinguish model outputs, computational checks, laboratory evidence,
human contributions, and independent review before promoting a scientific claim.
These reports are promising precisely because they can motivate concrete checks,
not because “science” makes every result equally verified. [95, 96]

### Evaluation and safety: preserve the conditions

- **Benchmark revisions and score types matter.** Fable's release table uses
  FrontierCode 2.0, FrontierSWE v2, and an offline partial-credit OSWorld subset.
  AA-Briefcase rubric scores and Elo are different measures. Google's new MKA
  evaluation targets multimodal agent work; it is not a continuation of an older
  general-intelligence time series. Do not compare unlike versions or measures
  as if their numerical gaps were progress. [95, 96]
- **Safeguard and substitute-model identity matter.** Anthropic says certain
  blocked Fable cyber/biology queries receive responses from earlier Opus models.
  Log actual model/substitute behavior, policy mode, and refusal outcomes rather
  than attributing all observed responses to Fable. Mythos and public Fable
  results cannot be freely exchanged. [95]
- **Persistence has trade-offs.** OpenAI's warning-following evaluation reports
  73.00% for Cyber versus 78.38% for Astra where stopping is appropriate, but
  80.43% versus 72.19% where continuing is appropriate. Its final-answer blocker
  test also flags Cyber more often, while explaining that many cases disclosed
  the blocker earlier rather than fabricating success. These are scoped tests,
  not overall deception rates or a reason to ignore stop conditions. [99]
- **Low observed attack rates are not guarantees.** Google reports 0.8%
  per-attempt breach rate in one adaptive cyberattack setup and other results
  under different protocols. Anthropic describes unresolved evaluation-awareness
  concerns. Keep the denominator, attacker access, tools, safeguard setting,
  task distribution, and test budget attached to each rate. Enforce approvals,
  isolation, credential scope, and resource limits outside model output. [95, 96]

### Price and effort ledger

These are announced standard input/output tariffs in USD per million tokens,
not end-to-end costs. Tool fees, retries, caches, request tiers, long-context
conditions, provider-specific billing, and availability must be checked before
use. Missing prices are intentionally not inferred.

| Model / mode | Input $/M | Output $/M | Qualification |
|--------------|----------:|-----------:|---------------|
| Fable / Mythos 5.2, standard | 8 | 40 | Mythos requires verified access [95] |
| Fable / Mythos 5.2, fast | 16 | 80 | Up to 2.5× token-generation speed is a vendor claim, not whole-task speed [95] |
| Public Gemini 4 Carbon, introductory | 2 | 10 | Through November 30; not a quote for privileged cyber access [96] |
| Public Gemini 4 Carbon, announced later tariff | 4 | 20 | Applies after the introductory period; future price, not current billing [96] |

A large model at modest effort can cost less per accepted task than a small model
at maximum effort; the reverse can also hold. Fable's and Carbon's launch
comparisons motivate workload-specific routing experiments, not automatic
upgrades. Account for all orchestrator, worker, critic, verifier, and retry
costs before calling a multi-agent result efficient. [95, 96]

## TL;DR — what we harvested

- **✅ Tested local implementations:** extended attempt accounting, real local
  midGPT generation with strict verification, token-fenced rollout/checker
  workers, and bounded cost/effort routing with deployment-contract checks.
  These are working-tree reference implementations, not production services.
- **✅ Bounded CPU experiments:** a causal convolution/local/global hybrid
  ablation and a tiny classifier baseline/SFT/LoRA comparison. The latter retains
  a LoRA regression; neither is an LFM3 or frontier-model reproduction.
- **🟡 Lean adapter:** hardened local-container invocation and failure tests are
  implemented, but live Lean execution remains unverified because Docker access
  and a trusted local image were unavailable. Actual pretrained-LLM specialization
  also remains a separately provisioned experiment.
- **Initial-harvest evidence:** [implementation, commands, results, and limits](./2026-11-harvest.md)
  records **1,162 passing tests and 20 skips** across the five independent suites.
  This initial CPU-only harvest used no paid APIs or model downloads.
- **Subsequent repair evidence:** [repair status and acceptance](./2026-11-repair-pass.md)
  records **1,465 CPU tests passed, 33 skipped**, plus **six tiny two-GPU FSDP
  checks passed**. It hardens checkpoints, evaluation, training, provenance,
  packaging, and safe defaults across all five projects—not model quality.
- **Track / ideal:** native NVFP4 training, trillion-scale deployment,
  thousand-agent proof work, and privileged cyber workflows remain research or
  production targets—not completed harvests.

## Hardware envelopes per project

Planning envelopes, not measured fit guarantees. Minimum hardware is for a
meaningful reduced job; ideal hardware permits the project's broader technique
set. Full Kimi and Inkling deployments are references beyond the smaller
projects' scopes, not substitutes for sizing their own models.

| Project | Scale | Minimal | Ideal | Unlocks at ideal |
|---------|-------|---------|-------|------------------|
| nanogpt-edu | 10M–100M | CPU checks; one 8–16 GB GPU for short-context experiments | One 24–48 GB GPU | Repeated-seed reward/effort ablations and independent candidate verification |
| midgpt | 124M–1.5B | One 16–24 GB GPU with appropriately sized model/context | 4–8 × 48–80 GB GPUs | Longer-context training, restart tests, controlled policy comparisons |
| distgpt | 1B–70B | One 24–48 GB GPU for small checks; 2–4 GPUs for sharding tests | Multi-node high-memory GPUs with fast interconnect; Blackwell when testing native NVFP4 | Heterogeneous rollouts, policy staleness, reduced MoE and low-precision ablations |
| coder-finetune | 0.5B–7B | One 16–24 GB GPU for LoRA/QLoRA plus isolated CPU checkers | 4–8 × 80 GB GPUs, independent evaluation workers | Full fine-tuning and bounded multi-domain post-training with held-out verification |
| frontier-platform | 1B–500B+ blueprint | CPU protocol/design checks; bounded integration sandbox | Multi-node training/serving pools, isolated verifiers, audit storage and enforced permissions | Ideal-scale mixed RL, scientific review workflows, governed multi-agent deployment |

## Tier 1 — high-ROI, broadly applicable

“✅ tested reference” means implemented and locally tested, not a released
production integration. “🟡 partial” retains the external validation boundary.
See the [harvest evidence](./2026-11-harvest.md) for exact scope and commands.

| Technique | What it does | Win | Cost / risk | Min HW | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|--------|---------|-----------|
| Domain-aware rollout accounting | Records task, tokens, duration, failures, policy and grader version | Makes mixed-RL cost and regressions visible | Aggregate reward and sample counts hide expensive domains | CPU metadata plus reduced GPU runs | [98] | ✅ tested local ledgers/workers; production mixed RL remains open | nanogpt-edu; distgpt; coder-finetune |
| Effort-aware routing with escalation | Uses premium reasoning only where accepted quality improves | Potentially better cost/quality trade-off | Router errors and multi-agent overhead | CPU/API or in-envelope local models | [95, 96] | 🟡 bounded frontier router and midGPT search implemented; real routing gains unmeasured | midgpt; coder-finetune; frontier-platform |
| Exact-theorem and attempt provenance | Pins statements, imported lemmas, checker, artifacts, and full search history | Makes proof claims auditable | Checker acceptance does not prove novelty or specification correctness | CPU checker; model budget separate | [95, 96] | 🟡 provenance and jailed adapter tested; live Lean blocked | nanogpt-edu; coder-finetune; frontier-platform |
| Safeguard-aware evaluation fixtures | Records policy mode, substitute models, refusals, warnings, and stop conditions | Prevents misleading capability attribution | Provider behavior changes; prompt-only enforcement is insufficient | CPU fixtures and authorized endpoints | [95, 99] | 🟡 frontier routing fixtures tested; provider migration checks remain open | coder-finetune; frontier-platform |
| Calibration and failed-branch retention | Keeps unsuccessful RL/SFT branches and independent validation outcomes | Detects overoptimization and winner-selection bias | Extra evaluation cost; domain-specific metrics | CPU for toy validation; GPU for larger models | [98] | ✅ tested accounting and toy calibration; pretrained-LLM validation remains open | nanogpt-edu; midgpt; coder-finetune |

## Tier 2 — scale- or hardware-gated wins

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| Native NVFP4 pretraining | Trains with Blackwell-era low-precision arithmetic | Potential training throughput/memory gains | Hardware, kernels, stability controls, and BF16 quality baselines; full recipe not supplied in the announcement | [98] | ideal; no reproduction | distgpt; frontier-platform |
| Asynchronous mixed-horizon RL | Trains from rollouts with very different completion times | Better utilization and broader task mix | Policy staleness, correct log-probs, reward normalization, restartable queues | [98] | ideal; reduced integration first | distgpt; coder-finetune; frontier-platform |
| Trillion-scale hybrid MoE | Combines sparse expert compute with hybrid attention | High capacity and long-context capability | Total weight/state memory, expert communication, supported kernels | [97] | track / ideal | distgpt; frontier-platform |
| Large multi-agent verification | Uses parallel workers and explicit checker/reviewer roles | May shorten elapsed time on decomposable work | Shared budgets, dependency tracking, isolated workspaces, exact artifact validation | [95, 96] | ideal; small controlled trials first | coder-finetune; frontier-platform |
| Restricted security-model workflows | Couples stronger capabilities to verified access and runtime controls | More capable authorized defensive work | Program approval, organization controls, isolated targets and independent enforcement | [95, 96, 99] | track | frontier-platform |

## Tier 3 — research bets (track, don't build yet)

- **Generalizable autonomous research.** Compare successful and failed tasks,
  total budgets, human assistance, and independent confirmation. Formal proof,
  a simulation result, and a laboratory discovery require different acceptance
  tests. **Harvest: track · nanogpt-edu; coder-finetune; frontier-platform.** [95, 96, 98]
- **Scalable supervision of multi-agent systems.** A thousand-agent result does
  not prove safe delegation or favorable total cost. Measure conflict recovery,
  correlated errors, independent review, stop propagation, and budget adherence.
  **Harvest: track · distgpt; frontier-platform.** [95, 96]
- **Robust safety evaluation under evaluation awareness.** Anthropic reports
  that Fable/Mythos voice evaluation suspicions more often than recent models;
  realistic unseen workflows and independent evaluator access remain important.
  **Harvest: track · frontier-platform.** [95]
- **Open training versus open artifacts.** Kimi's released weights and Inkling's
  promised weights represent different evidence states; neither alone releases
  the complete data, grader, environment, and compute stack. Track actual artifact
  publication rather than promoting promises into shipped status.
  **Harvest: track · distgpt; coder-finetune; frontier-platform.** [97, 98]

## Roadmap by project

October's roadmap carries forward, updated after the November 11
[bounded harvest](./2026-11-harvest.md). The following are **remaining**
experiments and integrations, not production capabilities already demonstrated.

| Project | Next harvest | Hardware tier | Notes |
|---------|--------------|---------------|-------|
| **nanogpt-edu** | Extend the repeated-seed hybrid comparison to a realistic local corpus and context-length sweep; connect external generator/grader usage to provenance v2 | minimal→ideal | Synthetic CPU evidence exists; sparse-kernel memory savings, general language gains, and external-verifier integration remain unproven |
| **midgpt** | Evaluate a trained local checkpoint on a held-out task set, then compare context/compaction/effort and interrupted-stream recovery | minimal→ideal | Real tiny-model inference and strict verification work; the smoke accepted zero tasks and is not quality evidence |
| **distgpt** | Attach a neural generator and learner to the tested local workers; add a transactional concurrent backend | ideal | Token fencing, policy/log-prob versions, staleness, costs, and restart tests exist; synchronous callbacks are not an async RL trainer |
| **coder-finetune** | Provision a trusted Lean image and validate true/false theorems; run the specialization protocol on an approved local pretrained LM | minimal→ideal | Container contract tests and tiny classifier calibration/regression results exist; live Lean and LFM3 fine-tuning remain unverified |
| **frontier-platform** | Add authenticated provider adapters, enforced cancellation, unknown-usage reconciliation, and durable reservations; evaluate routing on real holdouts | minimal→ideal | Cost-aware routing and deployment fixtures are tested references; production approval enforcement and reviewer/retrieval backends remain external |

## What shipped this month

- **November 11 working-tree harvest:** tested attempt accounting, real local
  GPT candidate generation, executable strict verification, acquisition-fenced
  rollout workers, a constrained Lean container adapter, and bounded routing
  with explicit tariffs and authorization checks. See [the implementation and
  evidence note](./2026-11-harvest.md); no release or Git commit is claimed.
- **Small CPU experiments:** three-seed hybrid sequence-model training and
  baseline/SFT/rank-one-LoRA classifier comparisons, with held-out results,
  regression/calibration checks, and failed-attempt accounting. These are
  educational measurements, not pretrained frontier-model benchmarks.
- **Initial validation:** 1,162 tests passed and 20 skipped across independent
  project suites. That CPU-only harvest did not validate live Lean or use paid
  inference, model/dataset downloads, or GPU training.
- **Repair follow-up:** [all ten audit findings mapped to current status](./2026-11-repair-pass.md),
  with **1,465 CPU tests passed, 33 skipped**, six tiny two-GPU FSDP checks,
  installed-wheel checks, and real offline tiny HF trainer steps. Live containment,
  large-scale training, and pretrained-model quality remain separate gates.
- **The rest of November is intentionally uncovered.** Later releases, the
  promised Inkling weights, and the announced Carbon tariff transition require
  fresh evidence before changing status.

## Sources

Sources continue October's cumulative numbering. All were inspected November
11, 2026. Dates are provider publication dates except the explicitly dated
repository inspection. External reports and papers linked from announcements
were not independently reproduced or proof-checked. No claim is made that search
coverage was exhaustive.

95. Anthropic, *Introducing Claude Fable 5.2 and Claude Mythos 5.2*, **November 9,
    2026**. Availability/safeguards, standard and fast-mode prices, benchmark
    conditions, proof/physics/hardware/protein examples, supplied-lemma footnote,
    substitute-model behavior, and safety limitations. The linked full system
    card and proof artifacts were not independently evaluated here.
    <https://www.anthropic.com/claude-fable-and-mythos-5-2>
96. Google, *Gemini 4 Carbon is now available*, **November 10, 2026**. Paid
    API/enterprise access, Ultra rollout, `/swarm` preview, MKA introduction,
    scientific examples, safety-test conditions, and introductory-price expiry.
    <https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-carbon-now-available/>
97. Moonshot AI, *Kimi-K3.1-Pro* model card, file listing, and license, **revision
    `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`**, inspected November 11.
    Architecture and 96 listed safetensors shards; weights not downloaded or
    executed. Model-as-a-service terms apply when aggregate licensee/affiliate
    revenue exceeds $20M over any consecutive 12 months; separate branding
    conditions and exceptions also apply. Read the full license, not this summary.
    <https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/README.md>
    <https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/LICENSE>
98. Thinking Machines Lab, *Introducing Inkling 2 Small*, **November 5, 2026**.
    Tinker access and future weight-release plan; pretraining stages, native
    NVFP4, joint-RL scale/objective/task mix, grading and effort, autoresearch
    examples including a failed RL branch, and model limitations.
    <https://thinkingmachines.ai/news/inkling-2-small/>
99. OpenAI, *GPT-6 Cyber System Card*, **November 5, 2026**. Specialized
    post-training, restricted Daybreak Red access, provider-framework risk
    classifications, safeguards, and scoped alignment/agent-safety evaluations.
    The published HTML card was read; no cyber tasks were executed.
    <https://deploymentsafety.openai.com/gpt-6-cyber>
