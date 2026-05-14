# 03 — Model Architecture

## Default: decoder-only transformer

- **Positional**: RoPE with configurable base (10k → 500k+ for long context).
- **Norm**: RMSNorm, pre-norm.
- **Activation**: SwiGLU, FFN dim ≈ 8/3 × hidden, rounded to multiple of 256.
- **Attention**: Grouped-Query Attention (GQA, 8 KV heads typical) for inference efficiency.
- **Bias**: none on linear layers.
- **Tied embeddings**: optional; usually untied at frontier scale.

## Reference shapes

| Name  | Layers | Hidden | Heads | KV heads | FFN  | Params | Tokens |
|-------|-------:|-------:|------:|---------:|-----:|-------:|-------:|
| 1B    | 24     | 2048   | 16    | 8        | 5632 | ~1.2B  | 1T     |
| 7B    | 32     | 4096   | 32    | 8        | 11008| ~6.7B  | 2T     |
| 70B   | 80     | 8192   | 64    | 8        | 28672| ~70B   | 5T     |
| 400B  | 126    | 16384  | 128   | 16       | 53248| ~400B  | 15T    |

## Optional MoE

Mixtral-style sparse FFN: 8 experts, top-2 routing, capacity factor 1.25, load-balancing aux loss (coef 0.01), router z-loss (coef 0.001). Tradeoff: 4× active params for ~1× compute, but training instability and serving complexity rise sharply.

## Implementation discipline

- All matmuls go through a single `linear()` wrapper so we can swap in FP8/Transformer Engine globally.
- All attention goes through one `attention()` wrapper (FlashAttention-2/3, with fallback).
- `init_weights()` uses scaled init (μP or GPT-NeoX-style) — critical for stable scaling experiments.
