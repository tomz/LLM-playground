<!--
  SOTA Watch edition template.
  Copy to docs/YYYY-MM-sota-llm-agi.md, then fill in. Keep section order stable
  so editions diff cleanly month-over-month. Delete these HTML comments.

  PURPOSE (read first): this is a digest of WHAT'S NEW AT THE FRONTIER of LLM &
  AGI — models, papers, benchmarks, techniques. It is NOT a changelog of this
  repo. Lead every section and every entry with the external development and why
  it matters to the field; only THEN note what (if any) we harvested. If the
  edition reads like release notes for LLM-playground, it has drifted. Budget:
  the frontier is the majority of the edition; our in-repo runs are evidence,
  not the headline. Frontier items we can't build are still in scope (tag them
  harvest: none / track) — never omit SOTA just because it isn't actionable.

  Framing rule: do NOT constrain content to any one machine's GPUs. For each
  project, assume both a MINIMAL and an IDEAL hardware envelope are available
  and recommend what is correct at that scale. "Harvest status" tracks whether
  we've implemented it in-repo; it is a SECONDARY tag and hardware is a sizing
  note, never a blocker.
-->
# SOTA Watch — LLM & AGI · YYYY-MM

**Editor:** <name/handle>  ·  **Published:** YYYY-MM-DD  ·  **Status:** draft | published

> One-paragraph theme for the month. **Lead with the frontier:** what moved at
> the state of the art since last edition (models, papers, benchmarks)? What's
> the through-line? Who is this edition most useful for? Mention our own harvest
> only after the field-level story is set.

## The frontier this month

<!-- SOTA-FIRST. 4–8 bullets on what's new in LLM & AGI at large, independent of
     whether we can build it: frontier model releases, landmark papers, benchmark
     movement, capability/safety milestones. This is the heart of the edition.
     Each bullet: what happened · why it matters to the field · source. -->

- …

## TL;DR — what we harvested

<!-- SECONDARY. 3–6 bullets on what (if anything) we pulled into the repo this
     month. Keep it short; it is the footnote to "The frontier this month". -->

- …

## Hardware envelopes per project

<!-- For each project, state the MINIMAL box that runs a meaningful job and the
     IDEAL box that unlocks the full technique set. These are aspirational
     targets, not a description of any one workstation. -->

| Project | Scale | Minimal | Ideal | Unlocks at ideal |
|---------|-------|---------|-------|------------------|
| nanogpt-edu | 10M–100M | … | … | … |
| midgpt | 124M–1.5B | … | … | … |
| distgpt | 1B–70B | … | … | … |
| coder-finetune | 0.5B–7B | … | … | … |
| frontier-platform | 1B–500B+ | … | … | … |

## Tier 1 — high-ROI, broadly applicable

| Technique | What it does | Win | Cost / risk | Min HW | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|--------|---------|-----------|
| … | … | … | … | … | … | shipped/planned | … |

## Tier 2 — scale- or hardware-gated wins

<!-- Techniques that need a specific arch generation or model scale to pay off.
     State the gate as a sizing fact (e.g. "FP8: Hopper+; pays off >1B"), not as
     "we can't run it." -->

| Technique | What it does | Win | Gate (scale / arch) | Source | Harvest | Project(s) |
|-----------|--------------|-----|---------------------|--------|---------|-----------|
| … | … | … | … | … | planned/ideal | … |

## Tier 3 — research bets (track, don't build yet)

- …

## Roadmap by project

<!-- Carry forward each month. What's the next correct technique to add to each
     project, and at what hardware tier. This replaces the old "deferred"
     table — everything is on a roadmap, sized by hardware, not blocked by it. -->

| Project | Next harvest | Hardware tier | Notes |
|---------|--------------|---------------|-------|
| … | … | minimal/ideal | … |

## What shipped this month

- …

## Sources

1. …
