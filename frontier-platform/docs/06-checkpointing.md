# 06 — Checkpointing

## Requirements

- **Asynchronous**: training must not block on disk I/O. Use a background thread that snapshots GPU tensors to pinned host memory, then streams to object store.
- **Sharded**: each rank writes only its shard; no all-gather. Use PyTorch Distributed Checkpoint (DCP) or Megatron's distributed format.
- **Resumable**: dataloader RNG state, optimizer state, LR scheduler step, gradient scaler state — all must round-trip.
- **Reshardable**: a checkpoint saved at TP=8/PP=16 must load at TP=4/PP=32 for inference or continued training on a different cluster.
- **Versioned**: `s3://ckpts/<run>/<step>/` with manifest JSON listing every shard, its sha256, and the model config used.

## Retention

- Last 10 steps → fast-tier storage (NVMe-backed S3 or Lustre).
- Every 10000 steps → cold storage forever.
- Every "milestone" (end of phase A/B/C, post-anneal) → triple-replicated, immutable, with eval report attached.

## Recovery SLA

A 70B run on 512 H100s wastes ~$2k/min when stalled. Target: detect failure in <60s, restart from last ckpt in <10 min.
