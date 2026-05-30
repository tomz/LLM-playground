# 03 — Model Architecture

## Default: decoder-only transformer

- **Positional**: RoPE with configurable base (10k → 500k+ for long context).
- **Norm**: RMSNorm, pre-norm.
- **Activation**: SwiGLU, FFN dim ≈ 8/3 × hidden, rounded to multiple of 256.
- **Attention**: Grouped-Query Attention (GQA, 8 KV heads typical) for inference efficiency.
- **Bias**: none on linear layers.
- **Tied embeddings**: optional; usually untied at frontier scale.

## Sparsity: MoE is the frontier default (2025)

Every leading 2025 model is a **large sparse MoE** (DeepSeek-V3 671B-total /
37B-active, Llama-4, Qwen3). Dense is now the exception, used for small tiers and
ablations. The model code (`platform/model/transformer.py`, `MoEFFN`) implements
the frontier recipe — enable it with `moe_num_experts > 1`:

- **Fine-grained experts** (`moe_expert_d_ffn` smaller than `d_ffn`): many narrow
  experts instead of a few wide ones, for finer specialization.
- **Shared expert(s)** (`moe_shared_experts`, DeepSeek-V3 uses 1): always-on
  FFN(s) that capture common knowledge so routed experts can specialize.
- **Aux-loss-free load balancing** (`moe_balance="aux_free"`, the default): a
  per-expert routing **bias** is nudged toward under-loaded experts each training
  step, equalizing load *without* the quality-degrading auxiliary loss. Only the
  router z-loss remains in the objective. Set `moe_balance="aux_loss"` for the
  legacy Switch-style load-balance loss.
- **Active vs. total params**: `ModelConfig.active_param_count()` gives the
  per-token cost (`top_k` routed + shared experts fire); `param_count()` gives
  the full knowledge capacity. The cost/scaling model (`12`, `13`) prices runs by
  *active* params — that is the lever that buys 600B-param quality at 37B-param
  training/inference cost.

## Reference shapes

| Name  | Layers | Hidden | Heads | KV heads | FFN  | Experts (top-k/shared) | Total | Active | Tokens |
|-------|-------:|-------:|------:|---------:|-----:|:----------------------:|------:|-------:|-------:|
| 1B    | 24     | 2048   | 16    | 8        | 5632 | dense                  | ~1.2B | ~1.2B  | 1T     |
| 7B    | 32     | 4096   | 32    | 8        | 11008| dense                  | ~6.7B | ~6.7B  | 2T     |
| 70B   | 80     | 8192   | 64    | 8        | 28672| dense                  | ~70B  | ~70B   | 5T     |
| 400B  | 126    | 16384  | 128   | 16       | 53248| dense                  | ~400B | ~400B  | 15T    |
| **MoE-1T** | 80 | 8192 | 64 | 8 | 2048 (exp) | 128 / top-8 / 1 shared | ~1T | ~37B | 15T |

The dense tiers remain useful baselines; **MoE-1T** is the frontier-shaped tier
(fine-grained + shared, aux-free) and is what the simulator prices for a frontier
run. See `configs/model_moe_1t.yaml`.

## Legacy: coarse Mixtral-style MoE

The 2023 recipe — 8 wide experts, top-2, capacity factor 1.25, load-balancing aux
loss (coef 0.01), router z-loss (coef 0.001) — is reproducible by setting
`moe_num_experts=8, moe_top_k=2, moe_balance="aux_loss"` with no fine-grained or
shared experts. Prefer the aux-free fine-grained recipe above for new runs.

## Implementation discipline

- All matmuls go through a single `linear()` wrapper so we can swap in FP8/Transformer Engine globally.
- All attention goes through one `attention()` wrapper (FlashAttention-2/3, with fallback).
- `init_weights()` uses scaled init (μP or GPT-NeoX-style) — critical for stable scaling experiments.
