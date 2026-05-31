#!/usr/bin/env bash
# Post-reboot helper: re-point examples 01-04 run.sh at a chosen GPU by UUID.
#
# After switching P100 -> RTX 5060 Ti (which requires a reboot for the card to
# enumerate), the CUDA_VISIBLE_DEVICES UUIDs pinned in each run.sh must be
# re-verified. This script finds the GPU whose name matches a pattern and
# rewrites the four example scripts in place.
#
# Usage:
#   tools/set_example_gpu.sh                 # auto: match '5060'
#   tools/set_example_gpu.sh "5060 Ti"       # match a custom name substring
#   tools/set_example_gpu.sh --uuid GPU-xxxx # set an explicit UUID
#
# Always prints the before/after and asks nothing destructive of running jobs.
set -euo pipefail

EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../examples" && pwd)"
SCRIPTS=(
  "$EX_DIR/01_pretrain_shakespeare/run.sh"
  "$EX_DIR/02_align_chain/run.sh"
  "$EX_DIR/03_moe_vs_dense/run.sh"
  "$EX_DIR/04_max_throughput/run.sh"
)

echo "== visible GPUs =="
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv

UUID=""
if [[ "${1:-}" == "--uuid" ]]; then
  UUID="${2:?--uuid requires a value}"
else
  PATTERN="${1:-5060}"
  # Pick the first GPU whose name contains PATTERN (case-insensitive).
  UUID="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader \
          | awk -F', ' -v p="$PATTERN" 'tolower($1) ~ tolower(p) {print $2; exit}')"
  if [[ -z "$UUID" ]]; then
    echo "ERROR: no GPU name matched '$PATTERN'. Use --uuid to set explicitly." >&2
    exit 1
  fi
  echo "matched '$PATTERN' -> $UUID"
fi

echo
echo "== patching run.sh files -> $UUID =="
for s in "${SCRIPTS[@]}"; do
  if [[ ! -f "$s" ]]; then echo "skip (missing): $s"; continue; fi
  # Match the whole CUDA_VISIBLE_DEVICES assignment value, whether it's the
  # portable default (`"${CUDA_VISIBLE_DEVICES:-0}"`) or a previously-pinned
  # UUID/index, and replace it with the chosen UUID.
  old="$(grep -oE 'CUDA_VISIBLE_DEVICES=[^[:space:]]+' "$s" | head -1 || true)"
  sed -i -E "s|export CUDA_VISIBLE_DEVICES=.*|export CUDA_VISIBLE_DEVICES=$UUID|" "$s"
  new="$(grep -oE 'CUDA_VISIBLE_DEVICES=[^[:space:]]+' "$s" | head -1 || true)"
  printf '  %-48s %s -> %s\n' "$(basename "$(dirname "$s")")/run.sh" "${old:-<none>}" "${new:-<none>}"
done
echo
echo "Done. Verify with: grep -n CUDA_VISIBLE_DEVICES $EX_DIR/0*/run.sh"
