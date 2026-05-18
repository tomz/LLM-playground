# 03 — MoE vs dense at matched active params

Trains two small transformers on the example 01 Shakespeare shards with
**matched active parameter counts** and compares them.

- **Dense**: `d_model=192, d_ffn=512, n_layer=6` (~3.7 M total = active)
- **MoE**: `d_model=128, d_ffn=384, n_layer=6, num_experts=4, top_k=2`
  (~3.0 M total; per-token compute uses top-2 experts so the FFN matmul cost is comparable to the dense FFN)

This is the textbook MoE setup from the Switch Transformer paper, scaled
down to fit in 1 GiB of GPU memory and finish in ~5 min on a 3050.

## What we measure

- Loss curve every 50 steps for both models
- Tokens/sec averaged over the last 500 steps
- For MoE only: `last_aux_loss` trajectory (z-loss + load-balance) and final per-expert token counts (should be roughly uniform if routing is healthy)

## Run

```bash
bash run.sh    # requires examples/01 shards + tokenizer
```

Produces `out/dense.json`, `out/moe.json`, `result.md`.
