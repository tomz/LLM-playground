#!/usr/bin/env bash
# Track B1 orchestrator: sequential 2-GPU A/B of GPT-2 (Arm A) vs llamafied
# (Arm B) at 350M on FineWeb-Edu. Arm A is assumed ALREADY RUNNING (launched
# separately); this script waits for it to finish, launches Arm B on the same
# two cards, waits for that, then renders the comparison plot.
#
# Idempotent-ish: it keys off the "done ->" completion marker that train.py
# prints, so re-running after Arm A finishes will skip straight to Arm B.
#
# Usage:  cd midgpt && nohup bash tools/run_llamafied_AB.sh > out/AB_orchestrator.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."   # -> midgpt/

# Arm B (llamafied: SwiGLU 3-matmul MLP + QK-norm) needs ~0.75 GiB more
# activation memory than Arm A (GPT-2 GELU). Arm A peaked at 13.35 GiB on the
# 16 GB card; Arm B OOM'd on the first backward with 1.26 GiB stranded as
# "reserved but unallocated" (fragmentation). expandable_segments uses one
# growable virtual address range to reclaim that stranded memory. This is a
# pure allocator change: zero effect on the math / loss / tokens-seen, so the
# iso-token, iso-param A/B stays clean. No-op for Arm A (already done, skipped).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=.venv/bin/python
TORCHRUN=.venv/bin/torchrun

A_CFG=configs/gpt2_350m_fweb_5060ti_2gpu.yaml
B_CFG=configs/gpt2_350m_llamafied_fweb_5060ti_2gpu.yaml
A_LOG=out/gpt2_350m_fweb_5060ti_2gpu_train.log
B_LOG=out/gpt2_350m_llamafied_fweb_5060ti_2gpu_train.log
A_OUT=out/gpt2_350m_fweb_5060ti_2gpu
B_OUT=out/gpt2_350m_llamafied_fweb_5060ti_2gpu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_done() {  # $1=log file, $2=human label
    local f=$1 label=$2
    log "waiting for $label to finish (marker 'done ->' in $f) ..."
    while true; do
        if grep -q "^done ->" "$f" 2>/dev/null; then
            log "$label DONE: $(grep '^done ->' "$f" | tail -1)"
            return 0
        fi
        # bail out if no torchrun is running AND the marker never appeared
        if ! pgrep -f "train.py --config" >/dev/null 2>&1; then
            sleep 5  # grace: maybe between processes
            if ! pgrep -f "train.py --config" >/dev/null 2>&1 \
               && ! grep -q "^done ->" "$f" 2>/dev/null; then
                log "ERROR: no training process running and '$label' never hit 'done ->'. Aborting."
                tail -15 "$f"
                return 1
            fi
        fi
        sleep 20
    done
}

# --- Arm A (already running) ---------------------------------------------
wait_done "$A_LOG" "Arm A (GPT-2)" || exit 1
test -f "$A_OUT/ckpt_best.pt" || { log "ERROR: Arm A best ckpt missing"; exit 1; }
log "Arm A best val: $(grep -o '"eval_val": [0-9.]*' "$A_OUT/log.jsonl" | sort -t: -k2 -g | head -1)"

# --- Arm B ----------------------------------------------------------------
log "launching Arm B (llamafied) on 2 GPUs ..."
$TORCHRUN --standalone --nproc_per_node 2 train.py --config "$B_CFG" > "$B_LOG" 2>&1
log "Arm B torchrun returned $?"
wait_done "$B_LOG" "Arm B (llamafied)" || exit 1
test -f "$B_OUT/ckpt_best.pt" || { log "ERROR: Arm B best ckpt missing"; exit 1; }
log "Arm B best val: $(grep -o '"eval_val": [0-9.]*' "$B_OUT/log.jsonl" | sort -t: -k2 -g | head -1)"

# --- Comparison plot ------------------------------------------------------
log "rendering comparison plot ..."
$PY tools/plot_midgpt_compare.py \
    --run  "$B_OUT/log.jsonl" "llamafied (RoPE+RMSNorm+SwiGLU+QKnorm)" \
    --base "$A_OUT/log.jsonl" "GPT-2 (learned-pos+LN+GELU)" \
    --out  "$B_OUT/compare_llamafied.png" \
    --hardware "2x RTX 5060 Ti 16 GB (Blackwell sm_120, bf16, DDP)" \
    --dataset  "FineWeb-Edu (1B-token slice), iso-param 354.6M vs 353.5M, iso-token 32768/step" \
    --title    "midgpt . llamafied vs GPT-2 (350M, iso-param, iso-token, 2-GPU DDP)"
log "plot -> $B_OUT/compare_llamafied.png"
log "ALL DONE. Both arms trained + plotted."
