#!/bin/bash
# Genuine 2-GPU FSDP2 run on two physical RTX 5060 Ti cards (or any two
# CUDA devices without NVLink). This is the launcher referenced by
# configs/gpt_416m_fweb_2gpu.yaml and examples/5060ti_416m_fineweb.md.
#
# Unlike scripts/run_5060ti_2rank.sh (two ranks colocated on ONE GPU, which
# needs MPS/gloo and is ~50x slower), this drives two SEPARATE visible
# devices, so FSDP2 fires real cross-GPU all-gather (forward) /
# reduce-scatter (backward) collectives.
#
# Topology caveat: the two 5060 Ti's are PCIe-connected (PHB host bridge),
# P2P = CNS (not supported), no NVLink. NCCL must route collectives through
# host shared memory, hence NCCL_P2P_DISABLE=1. With reshard_after_forward
# false (set in the config) + last-micro-step grad sync (in trainer.py), this
# lands at ~1.28x aggregate throughput vs a single 5060 Ti -- see the
# "Going multi-GPU" section of examples/5060ti_416m_fineweb.md.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/gpt_416m_fweb_2gpu.yaml}

# Two physical GPUs. Override to pick a different pair, e.g.
#   CUDA_VISIBLE_DEVICES=2,3 scripts/run_5060ti_2gpu.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# No NVLink / P2P on this consumer pair: NCCL routes through host memory.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# expandable_segments keeps the allocator from fragmenting under the
# unsharded param copy that reshard_after_forward=false keeps resident.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Reuse midgpt's tokenized FineWeb-Edu shards if this project has none of
# its own (same tiktoken gpt2 tokenizer). Harmless if the link already exists.
if [ ! -e data/fineweb-edu ] && [ -d ../midgpt/data/fineweb-edu ]; then
    ln -s ../midgpt/data/fineweb-edu data/fineweb-edu
fi

exec .venv/bin/torchrun --standalone --nproc_per_node 2 \
    -m distgpt.cli train --config "$CONFIG" --data data/fineweb-edu
