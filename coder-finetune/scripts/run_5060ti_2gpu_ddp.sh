#!/usr/bin/env bash
# Genuine 2-GPU DDP run for coder-finetune (item 4b — the GPU escalation of
# tests/test_dist_launch.py). Spawns two processes via `accelerate launch
# --multi_gpu`, one full 0.5B replica per RTX 5060 Ti, and runs a real LoRA SFT:
# forward / backward / optimizer step with the gradient all-reduce firing across
# both cards every step. TRL's Trainer (HuggingFace accelerate) owns the process
# group; cf_dist only *reads* the WORLD_SIZE/RANK/LOCAL_RANK it publishes.
#
# Prereq: the base model must be fetched once (CPU/network only):
#   .venv/bin/python -c "from huggingface_hub import snapshot_download as s; \
#       s('Qwen/Qwen2.5-Coder-0.5B')"
#
# Usage:
#   scripts/run_5060ti_2gpu_ddp.sh                         # default config
#   scripts/run_5060ti_2gpu_ddp.sh configs/other.yaml      # any 2-GPU config
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-configs/lora_2gpu_5060ti.yaml}

# Both cards visible; one rank pins to each via accelerate/LOCAL_RANK.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# The consumer 5060 Ti pair has no NVLink (PHB host bridge, P2P=CNS); NCCL must
# route the DDP all-reduce through host shared memory over PCIe.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# expandable_segments keeps the two replicas from fragmenting the allocator.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# --mixed_precision bf16 matches the config's dtype: bfloat16 (SFTConfig also
# sets bf16=True; aligning the launcher avoids a precision-mismatch warning).
exec .venv/bin/accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
    train.py --config "$CONFIG"
