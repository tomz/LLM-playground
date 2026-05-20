# midgpt — Mid-scale single-node GPT-2 trainer

GPT-2 scale (124M–1.5B). Real BPE tokenizer (`tiktoken`), WikiText-103 or OpenWebText, mixed precision, gradient checkpointing, gradient accumulation, cosine LR, DDP across 1–8 GPUs on a single node.

## What's different from `nanogpt-edu`

| Feature                       | nanogpt-edu | midgpt |
|-------------------------------|:----------:|:------:|
| Tokenizer                     | char-level | tiktoken (GPT-2 BPE, 50257) |
| Dataset                       | TinyShakespeare | WikiText-103 / OpenWebText |
| Mixed precision               | autocast   | autocast + GradScaler |
| Gradient checkpointing        | —          | ✅ (per-block) |
| Gradient accumulation         | basic      | DDP-aware (no_sync) |
| Multi-GPU                     | —          | torchrun + DDP |
| Eval                          | held-out loss | loss + perplexity + HellaSwag (zero-shot) |
| FlashAttention                | SDPA picks it | SDPA picks it; checkpoint-friendly |
| Logging                       | print      | print + JSON lines + optional W&B |
| Resume                        | last ckpt  | last ckpt with full RNG/optim/loader state |

## Quickstart

```bash
pip install -r requirements.txt

# 1. Tokenize a dataset (writes shards to data/<name>/)
python prepare.py --dataset wikitext103             # ~500 MB tokens, ~10 min
# or
python prepare.py --dataset openwebtext --num-proc 16  # ~9B tokens, hours

# 2. Train (single GPU)
python train.py --config configs/gpt2_124m.yaml

# 3. Train (8 GPUs on one node)
torchrun --standalone --nproc_per_node 8 train.py --config configs/gpt2_350m.yaml

# 4. Evaluate
python eval.py --ckpt out/gpt2_124m/ckpt.pt --tasks ppl,hellaswag

# 5. Generate
python sample.py --ckpt out/gpt2_124m/ckpt.pt --prompt "Once upon a time"
```

## Configs

| File                  | Params | Layers | d_model | Tokens (target) | 1× H100 | 8× H100 |
|-----------------------|-------:|-------:|--------:|----------------:|--------:|--------:|
| `gpt2_124m.yaml`      | 124M   | 12     | 768     | 10B             | ~30 h   | ~4 h    |
| `gpt2_350m.yaml`      | 350M   | 24     | 1024    | 20B             | ~5 d    | ~16 h   |
| `gpt2_774m.yaml`      | 774M   | 36     | 1280    | 40B             | —       | ~4 d    |
| `gpt2_1558m.yaml`     | 1.5B   | 48     | 1600    | 60B             | OOM     | ~10 d   |

("~" = napkin estimate at ~50% MFU, BF16, ZeRO-1 via DDP+grad-accum.)

## Layout

```
midgpt/
├── model.py          # GPT-2 architecture (LayerNorm, learned posn, full attention)
├── data.py           # tiktoken loader, packed sequences, mmap shards
├── prepare.py        # download + tokenize WikiText / OpenWebText
├── train.py          # DDP loop, AMP, grad-ckpt, grad-accum, resume
├── eval.py           # ppl + HellaSwag zero-shot harness
├── sample.py         # generation
├── utils/            # logging, schedule, ckpt manager
├── configs/          # YAML per model size
└── tests/
```

## Apple Silicon (MPS) support

`train.py` auto-detects MPS (Metal Performance Shaders) on macOS and runs in
native bf16 with `GradScaler` disabled (Metal doesn't need it for bf16). Tested
on an Apple M1 Pro (24 GB unified, macOS 26.3.1, PyTorch 2.12) — the 124M
config fits at micro-batch 4, sequence length 1024 with grad checkpointing.

A 200-iter smoke run on WikiText-103:

![midgpt MPS smoke run](out/smoke_124m/loss.png)

|              |              |
|--------------|--------------|
| device       | MPS bf16     |
| iters        | 200          |
| wall-clock   | ~1.5 min     |
| step time    | ~485 ms/it (median) |
| throughput   | ~2.1k tok/s  |
| train loss   | 10.96 → 6.99 |
| best val ppl | 1031         |

> ⚠️ Caveat — running a heavier 500-iter config (micro_batch=4, grad_accum=16,
> block_size=512 → 32× more compute per iter than the smoke run) triggered
> what looked like an MPS Metal-driver stall after ~30 iters: the process
> stayed alive but stopped advancing (`STAT=U`, ~8% CPU, 25 s of CPU time
> across 25 min wall-clock). The smoke config above completes cleanly — if
> you hit the same hang, reduce `micro_batch`/`grad_accum` and/or
> `block_size`, or set `grad_checkpoint: false`.

## What's still omitted (see `distributed-trainer`)

FSDP / tensor parallelism / pipeline parallelism / multi-node orchestration / RLHF.
