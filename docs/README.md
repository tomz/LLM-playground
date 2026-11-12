# SOTA Watch — LLM & AGI

> A monthly digest of **what's new in LLM & AGI at the state of the art** — the
> frontier models, papers, benchmarks, and techniques moving large language model
> training, fine-tuning, inference, and the broader road to AGI. Written for
> practitioners. A *secondary* lens asks what we can actually *harvest* into the
> [`LLM-playground`](../README.md) projects — but **the SOTA is the subject; the
> harvest is the footnote.**

Each edition is a self-contained, dated Markdown file in this folder. Editions
follow a fixed structure (see [`TEMPLATE.md`](./TEMPLATE.md)) so a reader can
diff month-over-month and so new findings slot into a predictable place.

## Repository engineering reviews

The [**November 2026 SOTA-readiness assessment**](./repository-assessment-2026-11.md)
records the pre-repair five-project working tree, including the recent harvest.
It preserves test results, ten targeted counterexamples, verified correctness
and reproducibility gaps, pinned upstream reference stacks, and project-specific
roadmaps with acceptance gates and minimal/ideal hardware envelopes.

The subsequent [**repair and enhancement pass**](./2026-11-repair-pass.md) maps
all ten findings to implemented fixes, tested support limits, and remaining
external gates. Acceptance: **1,465 CPU tests passed, 33 skipped**, plus **six
tiny two-GPU FSDP checks passed**. See its
[machine-readable evidence](./research/sota-repair-evidence-2026-11.json) and the
[independent test profiles](./testing-and-reproducibility.md). These are
correctness and integration results, not frontier-model quality measurements.

The [**post-repair SOTA gap analysis**](./sota-gap-analysis-2026-11.md) ranks
the next improvements (P0–P2) with acceptance gates for each project.

The [**June 2026 repository architecture assessment**](./repository-assessment-2026-06.md)
is the earlier code- and test-backed baseline. November explicitly revisits its
major findings rather than assuming they are still open or already fixed.
These engineering reviews are separate from the monthly SOTA Watch editions.

## Machine-specific model research

The [**September 8 local open-weight model study**](./research/local-open-weight-models-2026-09-08.md)
checks this workstation's 2× RTX 5060 Ti 16GB GPUs, memory, PCIe topology, and
installed Ollama models. It recommends general, coding, single-GPU, vision, and
writing candidates using primary model cards and actual quantized artifact sizes.
Memory-fit estimates are distinguished from benchmarks; no model loads, weight
downloads, or service changes were performed. This is a hardware-specific research
note, not a monthly SOTA edition.

## Editions

| Edition | Theme | Headline (frontier first) |
|---------|-------|---------------------------|
| [2026-11](./2026-11-sota-llm-agi.md) **draft · MTD through Nov 11** | Frontier access, heterogeneous RL, and verifiable scientific work | **Frontier:** Claude Fable/Mythos 5.2, broader Gemini 4 Carbon access, inspectable Kimi K3.1 Pro weights, Inkling 2 Small's training account and pending weights, and restricted GPT-6 Cyber. Formalization results retain their premises, parallelism, and verification limits. **Harvest (secondary):** [local accounting, verification, rollout workers, routing, and CPU ablations](./2026-11-harvest.md), plus [validated repairs](./2026-11-repair-pass.md); 1,465 CPU tests and six tiny two-GPU checks pass. Live Lean and pretrained-LLM specialization remain unverified; no frontier-model reproduction. |
| [2026-10](./2026-10-sota-llm-agi.md) **draft · retrospective full month** | Cheaper specialists, open-weight agents, and world-model previews | **Frontier:** Haiku 5.5 and GPT-6.1 Terra lower task costs; Mistral Large 4 releases weights; LFM3 brings small hybrid models; Kimi K3.1 Pro launches hosted access; Carbon and Genie 4 remain restricted previews at the October cutoff. **Harvest (secondary):** source-backed technique map, license/artifact checks, and five roadmaps; written November 11 without backdating implementation work. |
| [2026-09](./2026-09-sota-llm-agi.md) **draft · full month through Sep 30** | Longer-horizon agents, cheaper state, and harder integration contracts | **Frontier:** the first-half Astra/Fable/Gemini/Muse/DeepSeek wave is joined by MiMo V2.6's open agentic-RL stack, GPT-6 Sol/Luna/6.1, Claude Opus/Sonnet 5.5, Grok 4.7, restricted-preview Gemini 4 Argon, and Embed 5. Shipped APIs, open weights, and previews remain distinct. **Harvest (secondary):** cost, migration, retrieval, safety, and mixed-RL guidance plus all five roadmaps updated; no new implementation or local model benchmark claimed. |
| [2026-08](./2026-08-sota-llm-agi.md) **draft · MTD through Aug 2** | AI-generated mathematics with formal certificates | **Frontier:** OpenAI reports ten Astra-generated results on longstanding open problems across mathematics and theoretical CS, followed by human/model manuscript preparation and Lean formalization. **Harvest (secondary):** full attempt/research-artifact provenance, all-branch cost accounting, structured proof/checker contracts, and independent-review release gates shipped; real Lean, literature, and reviewer backends remain external. |
| [2026-07](./2026-07-sota-llm-agi.md) | Persistent, efficient, auditable agent systems | **Frontier:** GPT-5.6 advances useful work per token through programmatic tools and parallel-agent `ultra`; retained reasoning + compaction triples ARC-AGI-3 score at 6× fewer output tokens; a SWE-Bench Pro audit estimates ~30% broken tasks; GPT-Red and scientific-computing field evidence put self-play safety, verification, and stewardship on the critical path. **Harvest (secondary):** compaction, benchmark QA, durable rollout queues, programmatic tools, multi-agent budgets, and held-out red-team gates shipped as tested protocols; a 2-GPU A/B cut 350M prefill latency **57.0%** at 75% fewer context tokens. |
| [2026-06](./2026-06-sota-llm-agi.md) | The open-weight reasoning wave | **Frontier:** open-weight reasoning models closed on the closed labs — **Kimi K2 Thinking** (1 T MoE, native INT4-via-QAT, 200–300 tool calls), **DeepSeek-V3.2** (production sparse attention), **Qwen3-Next** / **Nemotron Nano 2** (hybrid linear-attention + ultra-sparse MoE), **gpt-oss** (MXFP4 from release), **Olmo 3** (fully-open model-flow), **nanochat**; **MAI-Thinking-1** from-scratch "hill-climbing" reasoning; method papers **LoRA Without Regret · DeepConf · RLPR · GSPO**; and a maturing **automated-AI-research** direction. **Harvest (secondary):** two on-hardware A/Bs — llamafied beats GPT-2 **16.8 % ppl**, FSDP2-over-PCIe flipped 0.69×→**1.28×** — plus MAI's 10 components, the four method papers behind in-repo tests, and a minimal AI-Scientist-style research harness (val_bpb 2.95→2.75) |
| [2026-05](./2026-05-sota-llm-agi.md) | Cheaper/faster training on commodity GPUs | **Frontier:** Muon at scale (Moonlight/MuonClip/Megatron), Multi-Token Prediction, the GRPO-successor family (DAPO/Dr.GRPO/GSPO) + SimPO/KTO, agentic RL & self-play, NSA/DSA sparse attention, FineWeb-Edu/DCLM data scaling. **Harvest:** Liger Kernel, DoRA/rsLoRA/NEFTune, Muon + MTP across the core repos |

## Cadence & conventions

- **SOTA-first, harvest-second.** This is a digest of *what's new at the frontier
  of LLM & AGI*, not a changelog of this repo. **Lead every edition and every
  entry with the external development** — the model, paper, benchmark, or
  technique and why it matters to the field. *Then*, as a secondary annotation,
  note what (if anything) we harvested into the projects. If an edition reads
  like release notes for `LLM-playground`, it has drifted — rebalance it. A
  rough budget: **the frontier should be the majority of every edition**; our
  in-repo runs are evidence and illustration, not the headline.
- **Published monthly**, named `YYYY-MM-sota-llm-agi.md`.
- Every claim carries a **source** and, where possible, a **reproducible
  reference** (repo, paper arXiv id, or our own run).
- Each technique is tagged with a **tier** (1 = high-ROI/broad → 3 = research
  bet). **Harvest status** (`shipped` / `planned` / `ideal`) is a *secondary*
  tag — it records our engagement with a frontier item, it is never the reason
  an item is in the digest. An item with `harvest: none` can still be the most
  important entry of the month.
- **Hardware is a sizing note, never a blocker.** Content is not constrained by
  any one workstation's GPUs. For each project we assume both a **minimal** box
  (runs a meaningful job) and an **ideal** box (unlocks the full technique set),
  and recommend what is correct at that scale. Datacenter-only techniques
  (FP8/NVFP4, FlashAttention-3, MoE, multi-node parallelism) are first-class,
  flagged with the scale/arch at which they pay off.
- To start next month's edition, copy [`TEMPLATE.md`](./TEMPLATE.md) to
  `YYYY-MM-sota-llm-agi.md`, carry forward the **Roadmap by project** table, and
  fill in what changed.

## How to add a finding mid-month

Editions are living documents until the month closes. To add a technique:

1. **Lead with the frontier development**, not our repo. Drop it in the correct
   **tier** table of the current edition and describe *what it is in the field
   and why it matters* — the model/paper/benchmark, who shipped it, what it
   changes.
2. Fill the row: *technique · what it does · win · cost/gate · min HW · source ·
   **then** harvest status · which project(s) it touches (if any)*. Harvest is
   the last column for a reason.
3. If we implemented it, link the commit and update the harvest status to
   `shipped` — but keep the entry's *framing* on the external development; the
   harvest is a one-line annotation, not the body.
4. If it only pays off at a larger scale/arch, mark it `ideal` and add it to the
   **Roadmap by project** table with the hardware tier — sized, not blocked.
5. **Frontier items with no harvest are still in scope.** If a paper or model
   matters to the field but we have no plan to build it, it still belongs in the
   digest (tag it `harvest: none` / `track`). Do not omit SOTA just because it
   isn't actionable for us.
