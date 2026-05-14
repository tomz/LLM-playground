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

## What this teaches

1. How attention, RoPE, RMSNorm, SwiGLU, and the residual stream fit together.
2. The full training loop: data → forward → loss → backward → step → log → checkpoint.
3. Mixed precision (`torch.amp`) and gradient clipping.
4. Cosine LR schedule with warmup.
5. Sampling with temperature + top-k.

## What it deliberately omits

FlashAttention, FSDP, gradient accumulation tricks, MoE, RLHF — see the `mid-scale-trainer` and `distributed-trainer` projects.
