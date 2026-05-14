# 05 — Distributed Training

## Parallelism axes

| Axis            | What it splits        | When to use                          |
|-----------------|-----------------------|--------------------------------------|
| Data (DP)       | batch                 | always                               |
| ZeRO / FSDP     | optimizer state, grads, params shards | always (pairs with DP)   |
| Tensor (TP)     | matmul cols/rows      | model > 1 GPU's HBM (typically ≥7B)  |
| Pipeline (PP)   | layers across stages  | model > 1 node                       |
| Sequence (SP)   | sequence dim          | long context (>16k)                  |
| Expert (EP)     | MoE experts           | MoE only                             |
| Context (CP)    | seq dim within attn   | very long context (>128k)            |

## Topology examples

- **7B / 64 H100s**: DP=8 × FSDP-shard=8. No TP, no PP.
- **70B / 512 H100s**: TP=8 (intra-node NVLink) × PP=8 × DP=8. ZeRO-1 across DP.
- **400B / 4096 H100s**: TP=8 × PP=16 × DP=32. Selective activation recomputation. Interleaved 1F1B schedule.

## Comms

- Intra-node: NVLink/NVSwitch (900 GB/s).
- Inter-node: 8× 400 Gb/s InfiniBand NDR per host, rail-optimized fat-tree.
- All-reduce via NCCL with SHARP when available.

## Backend

We wrap one of: Megatron-Core, NVIDIA NeMo, DeepSpeed, or PyTorch native (FSDP2 + TP via DTensor). The `platform/training/parallel.py` interface lets us swap.

## MFU target

- Dense 7B on H100: ≥55% MFU.
- Dense 70B on H100: ≥45% MFU.
- MoE: ≥30% MFU (routing overhead).

Anything below these means a perf bug; halt the run and profile.
