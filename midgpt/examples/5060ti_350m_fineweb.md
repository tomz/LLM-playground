# 5060 Ti example — 350M GPT-2 trained from scratch on FineWeb-Edu in 2.5 hours

A real, reproducible pretraining run on a **single RTX 5060 Ti (16 GB,
Blackwell, sm_120, native bf16)**. Trains a 354 M-parameter GPT-2 from
random init on a 1 B-token slice of `HuggingFaceFW/fineweb-edu` (the
`sample-10BT` config), eats 131 M tokens, and lands at **val ppl 58.2**
(loss 4.06) on held-out FineWeb-Edu.

Companion to [`out/smoke_124m/`](../out/smoke_124m/) (the 200-iter MPS
smoke run on Apple Silicon). Same `train.py`, same `model.py`, same
plotter — just scaled up to a real GPU and a real corpus.

## TL;DR

```bash
cd midgpt
.venv/bin/pip install -r requirements.txt        # one-time

# 1. Tokenize a 1 B-token slice of FineWeb-Edu (~9 min, ~2 GB on disk)
.venv/bin/python prepare.py --dataset fineweb-edu \
    --streaming --max-tokens 1000000000 --val-tokens 2000000

# 2. Train (2 h 27 min on a 5060 Ti)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
    --config configs/gpt2_350m_fweb_5060ti.yaml

# 3. Generate
.venv/bin/python sample.py --ckpt out/gpt2_350m_fweb_5060ti/ckpt_best.pt \
    --prompt 'The history of the telescope' --max-new-tokens 120
```

## Headline numbers

| Metric                     | Value |
|----------------------------|------:|
| GPU                        | RTX 5060 Ti 16 GB (sm_120) |
| Model                      | GPT-2 354 M (24 L × 1024 d × 16 H, tied embeddings) |
| Tokenizer                  | tiktoken `gpt2` (50 257 → padded to 50 304) |
| Dataset                    | FineWeb-Edu `sample-10BT`, 1 B-token slice |
| Sequence length            | 1024 |
| Effective batch            | 4 × grad_accum 8 = 32 sequences = **32 768 tokens / step** |
| Steps                      | 4 000 |
| **Wall-clock**             | **2 h 27 min** (median 2 204 ms / step) |
| **Throughput**             | **14.9 k tokens / second**, sustained 99 % GPU util |
| **Peak VRAM**              | **11.9 GB allocated / 12.8 GB reserved** |
| Tokens trained             | **131 M** (~0.37× Chinchilla for 354 M) |
| Train loss                 | 11.00 → **3.97** |
| **Best val loss**          | **4.064** (perplexity **58.2**) at iter 3 800 |
| Checkpoint size            | 4.0 GB (bf16 weights + AdamW state + RNG) |

## Training curve

![training curves](../out/gpt2_350m_fweb_5060ti/loss.png)

Three panels: loss (train EMA + per-step + val), cosine LR schedule, and
step time. Notes:

- **Loss panel**: textbook descent. Fast drop in the first ~400 iters as
  the model learns the vocab + frequent bigrams (11.0 → 6.0), then a long
  slow climb-down as it actually starts modelling text (6.0 → 4.0). Val
  tracks train within 0.05 the entire run — no overfitting, the model
  has plenty of capacity left for more tokens.
- **LR panel**: cosine decay from 3e-4 to 3e-5 over all 4 000 iters, with
  a 200-iter linear warmup. Min-LR is reached at the very end.
- **Step-time panel**: dead flat at 2 204 ms. The only spikes are the
  iters right after eval / checkpoint saves (CUDA cache churn). No drift,
  no thermal throttling on a 2.5 h sustained 99 % util run.

The `best val 4.064 (ppl 58.2)` star is at iter 3 800 (95 % of the way
through). That val curve still has slope — running to 10 000 iters
(another ~3.5 h) would land in the 3.5–3.7 range based on the trend.

## Sample completions

`sample.py --ckpt out/gpt2_350m_fweb_5060ti/ckpt_best.pt --temperature 0.7 --top-k 50`

> **The history of the telescope** \
> In the late 13th century, the first and second stage of the telescope
> was completed by a small telescope in the late 14th century. The
> telescope was designed to cover a large portion of the telescope's
> shape and reflect light at the angle of the light. The telescope was
> designed to reflect a small part of the telescope's shape and reflect
> light at the angle of the light. […]

> **Photosynthesis is** \
> an important component of plant cell metabolism. It is important for
> the action of plants. The cell's cell activity is responsible for the
> formation of the micro-organisms that are involved in the formation of
> the microorganisms. […]

> **When water is heated to 100 degrees Celsius,** \
> this means the water is not moving to the atmosphere. At this level,
> the water is not a solid, but it's not a solid. But if you need
> drinking water, you need to take a very deep breath and you should
> use enough water. […]

> **A neural network is a** \
> microcontroller that is used to connect the host computer to a
> computer. The computer is a microcontroller that can be used to
> connect the host computer to a computer. […]

This is **exactly** what a 354 M model trained on 131 M tokens (≈0.37×
Chinchilla) is supposed to sound like: fluent English, vaguely on-topic,
locally coherent within ~20 tokens, repetition loops on long generations,
factually nonsense ("13th-century telescope", "microcontroller neural
network"). The plumbing is correct — the model is just *undertrained*.

For comparison, real GPT-2-345M (the OpenAI release, ~50 B training
tokens, ~380× more compute) reaches val ppl ~26 on WebText and produces
factually-grounded continuations. To get there you'd run this same recipe
with `max_iters: 1_500_000` (~1.7 yr on this card), or rent 8× H100 for
a weekend.

## Why these numbers (calibration)

The recipe is the result of four calibration runs against the 5060 Ti.
All four use the same 354 M model + block_size 1024; the only thing that
changes is the micro_batch / grad_accum / grad_checkpoint mix:

| micro_batch | grad_accum | grad_ckpt | tok/s | peak VRAM |
|---:|---:|:-:|---:|---:|
| 4  | 8 | ✓ | 12.5 k | (n/a — not measured before VRAM hook landed) |
| 6  | 6 | ✓ | 12.5 k | 10.9 GB |
| 8  | 4 | ✓ | 12.5 k | 12.5 GB |
| 2  | 16 | ✗ | 14.2 k | 9.4 GB |
| **4** | **8** | **✗** | **14.9 k** | **12.8 GB** ← shipped |

Two findings:

1. **With grad-ckpt on, throughput is recompute-bound at 12.5 k tok/s** no
   matter how you slice the batch — you're just paying the ~30 %
   recompute penalty on every forward. The 8-GB-class cards have to
   accept that cost. The 5060 Ti doesn't.
2. **Turning grad-ckpt off and using a moderate micro_batch (4)** is the
   sweet spot: +19 % throughput at a peak-VRAM cost the 16 GB card can
   easily absorb. Pushing further to micro_batch=8 / no-ckpt OOMs at
   ~15.3 GB (close but no headroom for the gradient unscale).

## What this teaches vs. doesn't

**Teaches:**
- The full single-GPU pretraining loop end-to-end: tiktoken tokenization
  to uint16 shards, mmap dataloader, bf16 autocast, cosine + warmup,
  gradient accumulation, periodic val + ckpt, JSONL logging, plotter.
- That a 354 M-param transformer trained on 131 M tokens learns English
  surface form (grammar, plausible word co-occurrence) but not facts or
  long-range coherence — the classic undertrained shape.
- 5060 Ti throughput on a real production-size model: 14.9 k tok/s
  sustained, 99 % util, no thermal throttling over 2.5 h.

**Doesn't teach:**
- Multi-GPU. Single device only. For DDP across 1–8 GPUs see the same
  `train.py` invoked via `torchrun --standalone --nproc_per_node N`.
- FSDP / TP / PP. That's the `distgpt` project's job.
- A useful model. 0.37× Chinchilla is a smoke test for the *training
  curve*, not for downstream quality. Run 30× longer (or use a smaller
  model) for HellaSwag / LAMBADA numbers worth quoting.

## Files

```
configs/gpt2_350m_fweb_5060ti.yaml     # the recipe
out/gpt2_350m_fweb_5060ti/
├── ckpt.pt                            # final iter weights + optim + RNG (4 GB)
├── ckpt_best.pt                       # iter 3 800, best val 4.064 (4 GB)
├── log.jsonl                          # one record per log_interval (10) iter
├── loss.png                           # the 3-panel plot above
└── (intermediate checkpoint-*.pt at iter 500/1000/.../3500)
out/gpt2_350m_fweb_5060ti_train.log    # raw train.py stdout
```

## Reproducing exactly

```bash
cd midgpt
.venv/bin/pip install -r requirements.txt
.venv/bin/python prepare.py --dataset fineweb-edu \
    --streaming --max-tokens 1000000000 --val-tokens 2000000
rm -rf out/gpt2_350m_fweb_5060ti
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python train.py --config configs/gpt2_350m_fweb_5060ti.yaml \
    2>&1 | tee out/gpt2_350m_fweb_5060ti_train.log
.venv/bin/python tools/plot_midgpt.py out/gpt2_350m_fweb_5060ti_train.log \
    --out-dir out/gpt2_350m_fweb_5060ti \
    --hardware "RTX 5060 Ti 16 GB (Blackwell, bf16, no grad-ckpt)" \
    --dataset "FineWeb-Edu sample-10BT, 1B-token slice (131M tokens trained, 2.2 s/it)"
```

Drop-in alternatives to try:

- `optim.max_iters: 10000` → ~6.2 h, ~328 M tokens, expect val ppl ~40.
- `model.n_layer: 12, model.d_model: 768` → drop back to 124 M, train 4×
  more iters in the same wall-clock for a better-converged small model.
- `dataset: openwebtext` + the same `--streaming --max-tokens` flags →
  swap FineWeb-Edu for Web text. Expect slightly worse val ppl since
  OpenWebText is messier than FineWeb-Edu.
