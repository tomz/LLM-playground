# 16 — Multimodality

> **Status: design stub.** Gap #2 in `14-gap-analysis-vs-frontier.md`. The
> blueprint is text+code only across data, tokenizer, model, and eval. The
> 2025–2026 flagships (GPT-5.x, Claude Opus 4.x, Gemini 3.x) are **natively
> multimodal**; a text-only model is by definition not a flagship. This is
> effectively a second platform — this doc scopes the minimum viable version.

---

## Scope tiers

| Tier | Modalities | Difficulty | Notes |
|------|------------|------------|-------|
| MM-1 | text + image **understanding** | high | the table-stakes flagship capability |
| MM-2 | + image **generation** | higher | separate decoder / diffusion head or unified tokens |
| MM-3 | + audio (in/out) | higher | speech in, TTS out; real-time tiers |
| MM-4 | + video understanding | highest | Gemini's differentiator; temporal modeling + huge context |

This doc specifies **MM-1** (vision understanding) as the entry point; later
tiers get their own docs.

## Architecture options

1. **Adapter / late-fusion (LLaVA-style):** frozen-ish vision encoder (ViT/SigLIP)
   → projector (MLP or perceiver-resampler) → image tokens prepended to the LLM
   sequence. Cheapest; bolt-on; quality ceiling is lower.
2. **Early-fusion / native tokens:** images become tokens in the same stream the
   transformer trains on from (or early in) pretraining. Higher ceiling, much
   more expensive, touches the tokenizer contract. This is the frontier default.

Recommended path: ship MM-1 as adapter-style for a fast baseline, then move
vision tokens earlier in training for the frontier version.

## What it touches (this is the point)

Multimodality is not one module; it perturbs the whole pipeline:

- **Data (`01`):** interleaved image-text corpora (web docs with images,
  captions, OCR, charts, documents); per-modality filtering, dedup (perceptual
  hashing for near-dup images), and decontamination vs multimodal evals.
- **Tokenizer (`02`):** the contract gains image patch tokens / VQ codes (and
  audio frames / video tokens later). New special tokens: `<|image|>`,
  `<|image_end|>`, patch placeholders. Freezing the tokenizer now must account
  for this.
- **Model (`03`):** vision encoder + projector; position handling for 2D image
  grids; context-length pressure (images are token-expensive).
- **Pretraining (`04`):** modality-balanced mixes and a modality curriculum
  (text-heavy → introduce image-text → up-weight hard multimodal reasoning).
- **Eval (`08`):** MMMU, MathVista, ChartQA, DocVQA, RealWorldQA; (video: 
  for MM-4). Today's suite has zero multimodal coverage.
- **Serving (`10`):** image preprocessing, variable-resolution tiling, much
  larger and more variable prompt token counts → KV-cache and batching changes.
- **Safety (`09`):** image-based jailbreaks, CSAM/abuse classifiers, visual PII.

## Minimum viable MM-1 build

1. Integrate a pretrained vision encoder (SigLIP/ViT) behind
   `platform/model/vision.py` with a projector.
2. Extend the tokenizer contract with image placeholder tokens.
3. Add an interleaved image-text data path: `platform/data/multimodal.py`
   (acquire, perceptual-dedup, caption/OCR pairing, shard with image refs).
4. Two-stage train: (a) projector-only alignment on caption data, (b) joint
   instruction tuning on multimodal SFT.
5. Add an MMMU/ChartQA/DocVQA slice to `08-evaluation.md`.

## Cost / simulator implications (`12`, `13`)

- Vision tokens inflate effective sequence length → higher FLOPs/example; the
  `6·N·D` model needs a modality-weighted token count.
- Vision-encoder forward + image preprocessing add non-trivial dataloader/CPU
  and serving cost not currently modeled.

## References

- LLaVA / SigLIP / Qwen-VL / InternVL lines for adapter-style vision-language.
- Gemini & GPT-4o/5 system cards for native-multimodal direction (capabilities,
  not recipes — exact training is not public).
- See `14-gap-analysis-vs-frontier.md` §4.
