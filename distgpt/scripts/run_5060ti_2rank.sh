#!/bin/bash
# 2-rank FSDP run on a single RTX 5060 Ti (or any one CUDA device).
#
# Both ranks share the same GPU; NCCL uses CUDA IPC on the loopback path.
# This exercises the FSDP/DCP wiring without needing two physical GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/gpt_400m_fweb_5060ti.yaml}

# Both ranks pin to the same physical device.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Both ranks pin to physical device 0 (same GPU). distgpt's dist init looks
# at this env var to override the default LOCAL_RANK -> device_id mapping.
export DISTGPT_COLOCATE_RANKS=1
# NCCL refuses to init when two ranks share one physical device. Two ways
# around it:
#  (1) Run an NVIDIA MPS daemon (no sudo needed) so multiple processes
#      share the GPU as one CUDA context. This script auto-starts it.
#  (2) Force backend=gloo via DISTGPT_BACKEND=gloo. Slower (~50x for FSDP
#      all-gather) but no MPS dependency.
if [ "${DISTGPT_BACKEND:-nccl}" = "nccl" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
    export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-log}
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if ! pgrep -f nvidia-cuda-mps-control >/dev/null; then
        nvidia-cuda-mps-control -d
        sleep 1
    fi
fi
# Disable P2P + IB just to be clean on consumer hardware (no NVLink, no IB HCA).
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# expandable_segments helps two ranks share one GPU's memory pool without
# fragmenting the allocator.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Each rank will torch.cuda.set_device(local_rank). With CUDA_VISIBLE_DEVICES=0
# and nproc_per_node=2, both ranks see the same physical GPU but local_rank
# differs (0, 1). We override that below so both go to device 0.
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node 2 \
    -m distgpt.cli train --config "$CONFIG"
