# 04 — Max throughput benchmark (RTX 3050)

Push the RTX 3050 (8 GB, sm_86) to its practical limit and report what
fraction of peak compute we actually achieve.

## What it does

1. Loads (or builds) a BPE tokenizer + token shards from TinyShakespeare
   (corpus repeated ×30 → ~8 M tokens so the run is not data-bound).
2. Builds a **~125 M-parameter** decoder transformer (16 layers, d_model 512,
   16 heads GQA→4, ffn 2048, seq 1024) with **selective activation
   checkpointing**, in **fp16**.
3. **Auto-tunes the batch size** by OOM-probing in descending order
   `[16, 12, 8, 6, 4, 3, 2, 1]`.
4. Spawns `nvidia-smi -l 1` in the background and trains for 2000 steps.
5. Reports tokens/sec, **MFU%** (vs 9.05 TFLOPS fp16 theoretical on the 3050),
   peak memory, and mean / P50 / P95 GPU utilisation.
6. Generates a 200-token Shakespeare sample.

## Run

```bash
bash run.sh
```

Wall time: ~15–20 min on the 3050. Output report: `result.md`.
