#!/usr/bin/env bash
# Run all example pipelines in dependency order and capture logs.
#
# Examples 02 and 03 depend on 01's checkpoint/shards, so order matters.
# Each example's run.sh pins CUDA_VISIBLE_DEVICES and uses the frontier-platform
# venv. Re-verify the GPU UUID first (see tools/set_example_gpu.sh).
#
# Usage:
#   examples/run_all.sh                 # run 01,02,03,04
#   examples/run_all.sh 01 04           # run only the named examples
#   SKIP_GPU_CHECK=1 examples/run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

ALL=(01_pretrain_shakespeare 02_align_chain 03_moe_vs_dense 04_max_throughput)

# Resolve which examples to run (match by numeric prefix or full name).
if [ "$#" -gt 0 ]; then
    SELECTED=()
    for arg in "$@"; do
        for d in "${ALL[@]}"; do
            if [[ "$d" == "$arg"* || "$d" == "$arg" ]]; then SELECTED+=("$d"); fi
        done
    done
else
    SELECTED=("${ALL[@]}")
fi

if [ "${SKIP_GPU_CHECK:-0}" != "1" ]; then
    echo "== GPUs visible =="
    nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv || true
    PINNED="$(grep -hoE 'CUDA_VISIBLE_DEVICES=GPU-[0-9a-fA-F-]+' 01_pretrain_shakespeare/run.sh | cut -d= -f2 || true)"
    echo "== run.sh is pinned to: ${PINNED:-<none>} =="
    if ! nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -q "${PINNED:-__nomatch__}"; then
        echo "ERROR: pinned UUID not found among visible GPUs." >&2
        echo "Re-pin first:  tools/set_example_gpu.sh 5060   (or --uuid GPU-...)" >&2
        echo "Or bypass this check with SKIP_GPU_CHECK=1." >&2
        exit 1
    fi
fi

ts="$(date +%Y%m%d_%H%M%S)"
echo
echo "== running: ${SELECTED[*]} =="
for d in "${SELECTED[@]}"; do
    log="$d/run_${ts}.log"
    echo
    echo ">>> $d  (log: $log)"
    t0=$(date +%s)
    if bash "$d/run.sh" 2>&1 | tee "$log"; then
        echo "<<< $d OK  ($(( $(date +%s) - t0 ))s)"
    else
        echo "<<< $d FAILED — stopping (downstream examples depend on it)." >&2
        exit 1
    fi
done

echo
echo "== all done. result.md files updated: =="
for d in "${SELECTED[@]}"; do
    [ -f "$d/result.md" ] && echo "  $d/result.md"
done
echo "Review diffs with:  git -C .. diff -- examples/*/result.md"
