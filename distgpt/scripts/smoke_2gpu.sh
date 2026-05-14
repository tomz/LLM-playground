#!/bin/bash
# 2-GPU FSDP smoke run on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."

# Generate dummy token shards if missing
if [ ! -d data/tiny ]; then
    .venv/bin/python -m distgpt.data.prepare_dummy --out data/tiny --tokens 5000000
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NCCL_P2P_DISABLE=1   # heterogeneous GPUs / no NVLink
export NCCL_DEBUG=WARN

.venv/bin/torchrun --standalone --nproc_per_node 2 \
    -m distgpt.cli train --config configs/smoke.yaml --data data/tiny
