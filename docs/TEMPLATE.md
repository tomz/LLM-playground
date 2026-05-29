<!--
  SOTA Watch edition template.
  Copy to docs/YYYY-MM-sota-llm-agi.md, then fill in. Keep section order stable
  so editions diff cleanly month-over-month. Delete these HTML comments.
-->
# SOTA Watch — LLM & AGI · YYYY-MM

**Editor:** <name/handle>  ·  **Published:** YYYY-MM-DD  ·  **Status:** draft | published

> One-paragraph theme for the month. What's the through-line? What changed
> since last edition? Who is this edition most useful for?

## TL;DR — this month's harvest

<!-- 3–6 bullets. Lead with what we shipped into the repo this month. -->

- …

## Hardware reality check

<!-- State the target hardware so readers know which techniques are live vs
     deferred. Update if our fleet changes. -->

| Card | Arch | VRAM | bf16 | FP8 | Notes |
|------|------|-----:|:----:|:---:|-------|
| RTX 3050 | Ampere sm_86 | 8 GB | ✓ | ✗ | smoke / small LoRA |
| RTX 5060 Ti | Blackwell sm_120 | 16 GB | ✓ | (HW yes, SW immature) | main single-GPU workhorse |

## Tier 1 — high-ROI, low-risk (recommend doing)

<!-- Techniques worth adopting now. One subsection each, or a table. -->

| Technique | What it does | Win | Cost / risk | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|---------|-----------|
| … | … | … | … | … | shipped/planned | … |

## Tier 2 — meaningful but heavier lifts

| Technique | What it does | Win | Cost / risk | Source | Harvest | Project(s) |
|-----------|--------------|-----|-------------|--------|---------|-----------|
| … | … | … | … | … | planned/deferred | … |

## Tier 3 — research bets (track, don't build yet)

- …

## Watchlist / deferred (with gating condition)

<!-- Carry this table forward every month. The "unblocks when" column is the
     whole point: it turns "someday" into a trigger. -->

| Technique | Why deferred | Unblocks when |
|-----------|--------------|---------------|
| … | … | … |

## What shipped this month

<!-- Concrete changelog with commit links, so the digest doubles as a record. -->

- …

## Sources

<!-- Numbered, with arXiv ids / repo links / our own run logs. Flag any claim
     that relied on unverified or rate-limited search. -->

1. …
