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

## Repository engineering review

The [**June 2026 repository architecture assessment**](./repository-assessment-2026-06.md)
is a separate code- and test-backed review of the five projects: architecture,
implementation maturity, distributed correctness risks, safety, packaging,
reproducibility, and a three-phase improvement roadmap. It is not a SOTA Watch
edition.

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
| [2026-09](./2026-09-sota-llm-agi.md) **draft · MTD through Sep 7** | More capable agents: cost, control, and evidence | **Frontier:** GPT-6 Astra, Claude Fable/Mythos 5.1, Gemini 3.8 Flash/Cyber, and Muse Spark 1.3 advance long-running agent work; Astra's reduced monitorability, task-cost trade-offs, and safeguard-aware evaluation qualify the gains. Includes Granite 4.2 as a dated August carry-in. **Harvest (secondary):** five-project roadmap carried forward; no new implementation or benchmark claimed. |
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
