# 04 — Max throughput benchmark

Push the GPU to its practical limit and report what fraction of peak compute
we actually achieve. Device-agnostic: the theoretical fp16 peak used for the
MFU denominator is derived at runtime from the live device (see
`theoretical_fp16_tflops()` in `run.py`), so the report is correct on whatever
card it runs on. Currently targeted at the **RTX 5060 Ti (16 GB, sm_120)**.

## What it does

1. Loads (or builds) a BPE tokenizer + token shards from TinyShakespeare
   (corpus repeated ×30 → ~8 M tokens so the run is not data-bound).
2. Builds a **~500 M-parameter** decoder transformer (24 layers, d_model 1024,
   16 heads GQA→4, ffn 4096, seq 1024) with **selective activation
   checkpointing**, compute in **bf16** autocast (fp32 master weights).
3. **Auto-tunes the batch size** by OOM-probing in descending order
   `[48, 40, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]` — probes high first so
   larger-VRAM cards saturate.
4. Spawns `nvidia-smi -lms 500` in the background and trains for 1500 steps.
5. Reports tokens/sec, **MFU%** (vs the device-derived fp16 peak),
   peak memory, and mean / P50 / P95 GPU utilisation.
6. Generates a 200-token Shakespeare sample.

## Run

```bash
bash run.sh
```

Output report: `result.md`. Raw nvidia-smi log: `out/nvsmi.log`.
