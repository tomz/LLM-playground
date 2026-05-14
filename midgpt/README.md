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

## What's still omitted (see `distributed-trainer`)

FSDP / tensor parallelism / pipeline parallelism / multi-node orchestration / RLHF.
