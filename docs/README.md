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
| [2026-05](./2026-05-sota-llm-agi.md) | Cheaper/faster training on commodity GPUs | Muon optimizer, Multi-Token Prediction, Liger Kernel, DoRA/rsLoRA/NEFTune, FineWeb-Edu/DCLM data scaling |

## Cadence & conventions

- **Published monthly**, named `YYYY-MM-sota-llm-agi.md`.
- Every claim carries a **source** and, where possible, a **reproducible
  reference** (repo, paper arXiv id, or our own run).
- Each technique is tagged with a **tier** (1 = high-ROI/low-risk → 3 =
  research bet) and a **harvest status** against our projects
  (`shipped` / `planned` / `deferred` / `skip`).
- Hardware reality is stated up front: most of this repo runs on consumer
  Ampere/Blackwell cards (RTX 3050 8 GB, RTX 5060 Ti 16 GB), so techniques
  that only pay off on H100/B200 are explicitly flagged and deferred.
- To start next month's edition, copy [`TEMPLATE.md`](./TEMPLATE.md) to
  `YYYY-MM-sota-llm-agi.md`, carry forward the "Watchlist / deferred" table,
  and fill in what changed.

## How to add a finding mid-month

Editions are living documents until the month closes. To add a technique:

1. Drop it in the correct **tier** table of the current edition.
2. Fill the row: *technique · what it does · win · cost/risk · source · harvest
   status · which project(s) it touches*.
3. If you implemented it, link the commit and update the harvest status to
   `shipped`.
4. If it's only viable on hardware we don't have, move it to **Watchlist /
   deferred** with the gating condition.
