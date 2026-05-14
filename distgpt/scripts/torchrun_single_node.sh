#!/bin/bash
# Single-node, 8-GPU FSDP training of the 1B config.
set -euo pipefail
CONFIG=${1:-configs/1b.yaml}
DATA=${2:-data/tiny}
torchrun --standalone --nproc_per_node 8 -m distgpt.cli train --config "$CONFIG" --data "$DATA"
