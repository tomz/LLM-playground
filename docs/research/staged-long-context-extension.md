# Engineering Note: Staged Long-Context Extension

**Status:** design note (track) · **Tier:** Tier-3 → planned · **Min HW:** design
only; the recipe itself is *ideal*-tier (multi-node) · **Source:** MAI-Thinking-1
deep dive, [§4 / Appendix B](./mai-thinking-1-deep-dive.md#4-long-context-extension-appendix-b--a-clean-reusable-recipe)
([src 37](../2026-06-sota-llm-agi.md#sources)); long-context eval machinery in
[`platform/eval/long_context.py`](../../frontier-platform/platform/eval/long_context.py).

> **What this note is.** The standalone engineering note the June SOTA edition
> flagged as "still to write" — the cheap, concrete recipe to adopt *the moment a
> run trains past a short `block_size`*. It is a **design note, not code**: the
> bodies live at *ideal* (multi-node) scale, but the *recipe* and its
> implications for our config/eval surface are recordable now, and the
> measurement machinery (the long-context eval adapters) already shipped.

---

## TL;DR

Don't pay for long-context attention throughout pre-training. **Train short,
extend at the end:**

1. **Pre-train at 16K** — the bulk of the tokens, at the cheapest sequence length.
2. **Mid-train at 64K** — a medium stage to start stretching positions.
3. **A short, cheap 256K extension** — ~**140B tokens** (a few % of the budget).

The headline empirical facts from the source:

- A progressive **32K→256K** checkpoint **matches** a full **128K** run (trained
  at length on ~1T tokens) on **code NLL** — i.e. the cheap staged path buys the
  same long-context quality as paying for length the whole way.
- **Adaptation is remarkably fast**: most of the gain lands in the **first 1–10 %**
  of extension steps. The model is **recalibrating positional / attention
  behavior**, not learning new capability.
- It **extends to 1M+ tokens** at modest additional cost.

The "fast adaptation" finding is the load-bearing one: if long-context ability is
mostly a *recalibration* of RoPE frequencies + attention temperature rather than
new knowledge, then a short tail phase is *sufficient*, and the expensive
full-length run is *wasteful*.

---

## Why it works (the mechanism)

Quadratic attention makes a token at sequence length `L` cost `O(L²)`. Training
the *whole* run at 256K is therefore ~256× the per-token attention cost of 16K,
spent on every one of trillions of tokens — almost all of which don't need it.

What actually has to change to handle longer contexts is narrow:

- **RoPE frequency coverage.** Positions beyond the pre-train length were never
  seen, so the high-wavelength rotations are out of distribution. Extension
  (optionally with RoPE-base rescaling / NTK-by-parts / YaRN) teaches them.
- **Attention entropy / temperature.** Softmax over many more keys dilutes
  attention; the model recalibrates its logit scale (this is exactly the kind of
  thing **QK-norm**, which we already ship across the core repos, stabilizes).
- **Long-range induction patterns.** A modest amount of genuinely long data
  (retrieval, repo-level code, multi-doc) at the tail.

None of these require re-learning the model's knowledge — hence the 1–10 %-of-steps
adaptation curve. Contrast with the parts that *don't* transfer cheaply (and so
the source dropped): long-context **data mixes that didn't help**, and MRCR-style
objectives that **overfit**.

---

## The staged recipe (reference shape)

| Stage | Seq len | Token budget (illustrative) | Purpose |
|-------|--------:|----------------------------:|---------|
| Pre-train | 16K | bulk (≫ 90 %) | cheap; learn the language/knowledge |
| Mid-train | 64K | medium | begin stretching positions |
| Extension | 256K | ~140B (a few %) | recalibrate positions + attention; add long data |

Knobs that matter at each boundary:

- **Position scaling.** Bump `rope_base` (or apply NTK-by-parts / YaRN) at each
  length increase so the rotations cover the new range. (Tier-3 "long-context
  position schemes" — YaRN / NTK-by-parts / NoPE — are the menu here.)
- **Attention backend.** The extension phase is exactly where **FlexAttention**
  (shipped this month in `midgpt`/`nanogpt-edu`) and document/block masking pay
  off — packed long documents with a block-causal mask. This note **pairs with
  the FlexAttention work**: extension is its first real use case.
- **Data mix.** Tail in genuinely-long samples (repo-level code, multi-doc
  retrieval, books) — but treat the mix as an *ablation target*, since the source
  explicitly found some long mixes unhelpful.
- **LR.** A small constant or short cosine tail (the source ran cosine peak
  `2e-5` → constant `1e-6` globally); the extension is a *recalibration*, not a
  fresh training phase, so it wants a gentle LR.

---

## How we'd measure it (machinery already shipped)

The evaluation side of this recipe is **already in the repo** — the MAI Tier-2
harvest landed the long-context eval adapters this note's quality claims are
stated against:

- [`eval/long_context.py`](../../frontier-platform/platform/eval/long_context.py)
  — **`CodeNLLAdapter`** (position-bucketed code NLL — the exact metric the
  "32K→256K matches 128K" claim is measured on), **`RetrievalNLLAdapter`**
  (needle-in-haystack by LM loss), **`LongContextQAAdapter`**
  (answer-accuracy-by-depth), plus `make_needle_record` and `bucketize`.
- These plug into `Evaluator.run_long_context`, so a staged-extension run is
  **gradeable today**: bucket NLL by position to confirm the tail phase actually
  flattened the loss-vs-depth curve (the signature of successful extension), and
  watch accuracy-by-depth to catch a "lost-in-the-middle" regression.

So the asymmetry is deliberate: **we can already score long-context quality; what
this note scopes is the cheap *training schedule* that produces it.**

---

## When this becomes a build (triggers)

Per the SOTA roadmap this is "the cheap, concrete recipe to adopt the moment we
train past `block_size 1024`." Concretely, promote from note → build when:

1. **A run wants a context longer than its pre-train `block_size`.** That's the
   trigger; until then, single-length training is correct and this is overhead.
2. **FlexAttention is exercised on a packed-doc / long-ctx config** (the
   `nanogpt-edu` roadmap item) — that's the minimal-tier proving ground for the
   masking the extension stage relies on.
3. **A multi-node run** (`distgpt` / `frontier-platform` ideal tier) reaches the
   token budget where paying for full-length attention throughout is the
   measurable waste this recipe removes.

### Project landing spots (when triggered)

- **`distgpt`** — the natural home for the *training* schedule: a staged
  sequence-length curriculum (`16K → 64K → 256K`) with per-stage `rope_base` /
  position-scaling, since it already owns the multi-stage, resumable,
  reshardable training loop. The DCP checkpoint boundary is exactly where a
  stage transition (length + position-scaling change) is cleanest.
- **`frontier-platform`** — a `docs/04-pretraining.md` curriculum section + a
  `TrainingConfig` shape for staged lengths, scored by the long-context eval
  adapters above. Interfaces only, per the platform's design-doc-first rule.
- **`midgpt` / `nanogpt-edu`** — minimal-tier *demonstration*: take a
  short-`block_size` checkpoint and run a tiny extension phase with a rescaled
  `rope_base` + FlexAttention block mask, then show the position-bucketed NLL
  curve flatten. A legible, single-GPU illustration of the mechanism (not the
  scale).

---

## Scope rule / caveats

- **Design note, not a body.** Nothing here is a `NotImplementedError` stub to
  fill — it's a recipe + the config/eval surface it would touch. No
  `frontier-platform/` code changes accompany this note (the doctrine's
  "no platform change without a design-doc update" cuts both ways: this *is* the
  doc, and it intentionally ships alone).
- **Source is a company report, not peer-reviewed** — the "32K→256K matches 128K"
  and "1–10 % of steps" figures are the originators' (Appendix B), carried at
  their confidence. They become an in-repo number only when a staged run is
  actually executed and scored with our adapters.
- **The hard parts are ideal-tier.** 256K × multi-node attention, the long-data
  curation, and the FP8/parallelism to make the tail phase affordable are
  out-of-scope sizing facts — first-class as *targets*, not blockers.

---

## Sources

- MAI-Thinking-1 deep dive, [§4 / Appendix B](./mai-thinking-1-deep-dive.md#4-long-context-extension-appendix-b--a-clean-reusable-recipe)
  (staged 16K→64K→256K, 140B-token tail; 32K→256K matches 128K on code NLL;
  1–10 %-of-steps adaptation).
- June SOTA edition, [Tier 3 — "Staged long-context extension"](../2026-06-sota-llm-agi.md#tier-3--research-bets-track-dont-build-yet)
  and the `frontier-platform` roadmap row.
- Long-context eval machinery already in-repo:
  [`platform/eval/long_context.py`](../../frontier-platform/platform/eval/long_context.py)
  (MAI Tier-2 harvest item #7).
