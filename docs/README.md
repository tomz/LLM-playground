# SOTA Watch — LLM & AGI

> A monthly digest of state-of-the-art techniques in large language model
> training, fine-tuning, inference, and the broader road to AGI — written for
> practitioners, scoped to what we can actually *harvest* into the
> [`LLM-playground`](../README.md) projects.

Each edition is a self-contained, dated Markdown file in this folder. Editions
follow a fixed structure (see [`TEMPLATE.md`](./TEMPLATE.md)) so a reader can
diff month-over-month and so new findings slot into a predictable place.

## Editions

| Edition | Theme | Headline harvest |
|---------|-------|------------------|
| [2026-05](./2026-05-sota-llm-agi.md) | Cheaper/faster training on commodity GPUs | Muon (now scale-proven: Moonlight/MuonClip/Megatron), Multi-Token Prediction, Liger Kernel, DoRA/rsLoRA/NEFTune, the GRPO-successor family (DAPO/Dr.GRPO/GSPO) + SimPO/KTO, agentic RL & self-play, NSA/DSA sparse attention, FineWeb-Edu/DCLM data scaling |

## Cadence & conventions

- **Published monthly**, named `YYYY-MM-sota-llm-agi.md`.
- Every claim carries a **source** and, where possible, a **reproducible
  reference** (repo, paper arXiv id, or our own run).
- Each technique is tagged with a **tier** (1 = high-ROI/broad → 3 = research
  bet) and a **harvest status** against our projects
  (`shipped` / `planned` / `ideal`).
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

1. Drop it in the correct **tier** table of the current edition.
2. Fill the row: *technique · what it does · win · cost/gate · min HW · source ·
   harvest status · which project(s) it touches*.
3. If you implemented it, link the commit and update the harvest status to
   `shipped`.
4. If it only pays off at a larger scale/arch, mark it `ideal` and add it to the
   **Roadmap by project** table with the hardware tier — sized, not blocked.
