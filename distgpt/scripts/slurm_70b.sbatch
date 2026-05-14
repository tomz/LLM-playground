#!/bin/bash
#SBATCH --job-name=distgpt-70b
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Network env (NCCL + IB)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES

srun --label torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m distgpt.cli train \
    --config configs/70b.yaml
