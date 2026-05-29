# nanogpt-edu — Educational GPT trainer

~500 lines of PyTorch. Trains a 10M–100M parameter GPT on TinyShakespeare on a single GPU or CPU. Inspired by Karpathy's nanoGPT, restructured for readability.

## What you get

- `model.py` — decoder-only transformer (RoPE, RMSNorm, SwiGLU, GQA optional). One file, ~180 lines.
- `data.py` — character-level dataset + binary-shard dataset.
- `train.py` — training loop with cosine LR, grad clipping, AMP, eval, checkpoints.
- `sample.py` — generate text from a checkpoint.
- `prepare_shakespeare.py` — download + tokenize TinyShakespeare.

## Quickstart

```bash
pip install torch numpy requests
python prepare_shakespeare.py            # ~1 MB download
python train.py --config configs/tiny.py # ~10M params, ~5 min on 1 GPU, ~1h on CPU
python sample.py --ckpt out/ckpt.pt --prompt "ROMEO:"
```

## Configs

| Config        | Params | Layers | d_model | Train time (1× RTX 4090) |
|---------------|-------:|-------:|--------:|--------------------------:|
| `tiny.py`     | ~10M   | 6      | 384     | ~5 min                    |
| `small.py`    | ~30M   | 8      | 512     | ~20 min                   |
| `medium.py`   | ~110M  | 12     | 768     | ~2 h                      |

## Actual training results (1× RTX 3050, 8 GB)

Char-level Tiny Shakespeare (1 MB). Training stdout logs are committed
under `out/<run>/train.log`; PNG/SVG plots are written by
`python tools/plot_nanogpt.py out/<run>/train.log`.

| Run         | Params  | Iters done   | ms/it (3050) | Wall    | Final train | **Best val** | Final val | Overfit Δ |
|-------------|--------:|-------------:|-------------:|--------:|------------:|-------------:|----------:|----------:|
| `smoke`     |  0.86 M | 275 / 300    |  14.0        |  ~5 s   | 1.90        | **1.99**     | 1.99      |  ~0       |
| `tiny`      | 10.65 M | 4,990 / 5,000|  203.4       | ~17 min | 0.07        | **1.53**     | 4.34      | **2.81**  |
| `tiny_clean`| 10.65 M | 1,500 / 1,500|  203.0       | ~5 min  | 0.53        | **1.48**     | 1.85      |   0.36    |
| `small`     | 25.73 M | 9,600 / 15,000 (killed @ 64 %) |  798.4 | ~2 h 15 | 0.04        | **1.87**     | 5.21      |   3.35    |

`tiny_clean` is the same architecture as `tiny`, run for 1,500 iterations
with `dropout=0.1` instead of `0.0`. It's the **regularized** counterpart
that produces the textbook U-shaped val curve (descent → minimum → mild
ascent) — and notably, it lands on a *slightly better* best-val (1.48 vs
1.53) than the un-regularized `tiny` did, while overfitting **8× less**.

Things the plots make obvious:

1. **Step time scales roughly with `d_model²·block_size`** on the 3050:
   14 → 203 → 798 ms (d_model 128 → 384 → 512, block 128 → 256 → 512).
2. **Tiny Shakespeare overfits hard on any non-trivial model.** Both
   `tiny` and `small` hit their best val between iter ~500–800 (right
   after warmup), then val rises monotonically while train collapses
   toward zero. **The useful checkpoint is *not* the last one.**
3. **Bigger model is worse on val** at this dataset scale (best val 1.87
   for `small` vs 1.53 for `tiny`) — classic overcapacity for a 1 MB
   corpus. To make the larger config actually shine, scale tokens with
   params (i.e. use a larger dataset) or add dropout/early stopping.
4. **Cosine schedule looks textbook** in panel 2 — short warmup, smooth
   decay to `min_lr`.

### Per-run plots

Each PNG is 3-panel: train+val loss / cosine LR / step time.

| smoke | tiny | tiny_clean | small |
|---|---|---|---|
| ![smoke](out/smoke/loss.png) | ![tiny](out/tiny/loss.png) | ![tiny_clean](out/tiny_clean/loss.png) | ![small](out/small/loss.png) |

### Cross-run comparison

Linear-x and log-x val-loss overlays for all three "real" runs (`tiny`,
`tiny_clean`, `small`):

![compare](out/compare.png)

### Overfit vs regularized — same architecture, two outcomes

`tiny` (no dropout, 5,000 iters) vs `tiny_clean` (dropout 0.1, 1,500 iters).
Both are 10.65 M params on the same 1 MB of Shakespeare. The *only*
intervention is dropout + early stopping; you can read the textbook
overfit picture directly off the val curves:

![tiny vs tiny_clean](out/compare_overfit.png)

### Reproducing

```bash
mkdir -p out/{smoke,tiny,tiny_clean,small}
python train.py --config configs/smoke.py      2>&1 | tee out/smoke/train.log
python train.py --config configs/tiny.py       2>&1 | tee out/tiny/train.log
python train.py --config configs/tiny_clean.py 2>&1 | tee out/tiny_clean/train.log
python train.py --config configs/small.py      2>&1 | tee out/small/train.log

# Plot one run, all + a comparison overlay, or the overfit-vs-regularized panel:
python tools/plot_nanogpt.py out/tiny/train.log
python tools/plot_nanogpt.py out/{smoke,tiny,tiny_clean,small}/train.log \
    --compare out/compare.png \
    --hardware "RTX 3050 bf16" --dataset "Tiny Shakespeare 1.1 MB"
python tools/plot_nanogpt.py out/{tiny,tiny_clean}/train.log \
    --compare out/compare_overfit.png
```

`tools/plot_nanogpt.py` uses matplotlib for PNGs if available and always
writes a zero-dependency SVG fallback.

## Going faster — the modded-nanogpt speedrun knobs

The base configs are deliberately vanilla. Several techniques from Keller
Jordan's [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)
(which trains GPT-2 to the llm.c target ~15× faster) are now available as
opt-in config flags — small, readable, and pedagogically interesting:

| Knob | Config key | What it does |
|------|-----------|--------------|
| **Muon optimizer** | `optimizer='muon'` (+`muon_lr`) | Orthogonalizes the SGD-momentum update for 2D hidden weights via a 5-step Newton-Schulz iteration; embeddings/lm_head/norms stay on AdamW. ~1.35× sample-efficiency on FineWeb. See `muon.py`. |
| **QK-Norm** | `qk_norm=True` | Per-head RMSNorm on Q and K before RoPE — stabilizes attention logits, lets you push LR higher. |
| **Zero-init projections** | `zero_init_proj=True` | Zero-inits the residual-write matrices (attn `o_proj`, ffn `w2`) so each block starts as identity — stable high-LR warmup (muP-like). |
| **Untied embeddings** | `tie_embeddings=False` | Gives `lm_head` its own weight; helps loss once you have the tokens to support the extra params. |

A ready-made head-to-head config is `configs/tiny_muon.py` — same 10.65M
architecture as `tiny_clean.py`, same iters/budget, all four knobs on:

```bash
python train.py --config configs/tiny_clean.py  # AdamW baseline
python train.py --config configs/tiny_muon.py   # Muon + speedrun knobs
python tools/plot_nanogpt.py out/{tiny_clean,tiny_muon}/train.log --compare out/muon_vs_adamw.png
```

> On 1 MB of TinyShakespeare the **data** is the bottleneck, not the optimizer,
> so the gap is modest. To see Muon shine, scale the tokens (next section).

## Scaling the data — FineWeb-Edu (fixes the overfit story)

The README results above show every non-trivial model overfitting 1 MB of
Shakespeare, with the *bigger* model doing *worse* on val. That's a data-scale
artifact, not a model flaw (Chinchilla: scale tokens with params). Swap in a
slice of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(GPT-2 BPE tokenized, same shard format) and the inversion disappears:

```bash
pip install tiktoken datasets
python prepare_fineweb.py --tokens 100_000_000 --out-dir data_fineweb
python train.py --config configs/small_fineweb.py   # Muon + real data
python sample.py --ckpt out/small_fineweb/ckpt_best.pt --prompt "The mitochondria"
```

`prepare_fineweb.py` streams the dataset and writes `train.bin`/`val.bin`/
`meta.pkl` exactly like `prepare_shakespeare.py`; `sample.py` auto-detects the
BPE tokenizer from `meta["tokenizer"]`.

## What this teaches

1. How attention, RoPE, RMSNorm, SwiGLU, and the residual stream fit together.
2. The full training loop: data → forward → loss → backward → step → log → checkpoint.
3. Mixed precision (`torch.amp`) and gradient clipping.
4. Cosine LR schedule with warmup.
5. Sampling with temperature + top-k.

## What it deliberately omits

FSDP, MoE, RLHF — see the `mid-scale-trainer` and `distributed-trainer`
projects. FlashAttention is used implicitly via PyTorch SDPA. Many of the
modded-nanogpt speedrun's heavier tricks (FlexAttention long-short windows,
FP8 matmul, U-net skip connections, value embeddings) are also omitted to keep
the core legible — the four cheapest/most-instructive ones (Muon, QK-norm,
zero-init, untied head) are available as opt-in flags; see above. FP8 in
particular needs Hopper+ and is useless on consumer Ampere/Blackwell cards.
