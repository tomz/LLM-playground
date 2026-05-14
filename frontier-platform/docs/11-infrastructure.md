# 11 — Infrastructure

## Cluster

- **GPUs**: H100/H200 SXM5 (80–141 GB HBM), 8 per node. Blackwell (B200) where available.
- **Network**: 8× ConnectX-7 NDR 400 Gb/s per node, rail-optimized fat-tree, non-blocking within superpod (≤4096 GPUs), 2:1 oversubscription across superpods.
- **Storage**:
  - Hot: VAST or WEKA parallel filesystem, 1 TB/s aggregate read for dataloading.
  - Warm: S3-compatible object store for shards & checkpoints.
  - Cold: Glacier/Coldline for archival.
- **Power**: ~10 kW per H100 node; a 4096-GPU pod ≈ 5 MW. Plan PUE, water, and grid contracts.

## Scheduler

- Slurm or Kubernetes + Volcano/Kueue. Gang scheduling required.
- Topology-aware placement (rack/leaf-switch/superpod labels).
- Preemption policy: pretraining > eval > research > batch.
- Per-job budget caps and quota with chargeback.

## Observability

- Per-rank metrics over OTLP → Prometheus + Grafana + Loki.
- DCGM for GPU health (XID errors, ECC, thermal). One bad GPU in 4096 = whole job stalls.
- Auto-quarantine flaky nodes; nightly burn-in tests.
- W&B or MLflow for experiment tracking; immutable run IDs.

## Reliability math

MTBF of an H100 ~ 10 years. On 4096 GPUs, expect ~1 GPU failure every ~21 hours. Training MUST tolerate node failure with checkpoint-restart in <10 min. Hot-spare nodes: 2–5% of fleet.
