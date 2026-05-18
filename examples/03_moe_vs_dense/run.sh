#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=GPU-bb18b9d3-2ce9-0696-26a6-fffaee036dbb
export PYTHONPATH=/home/support/dev-macrohard/LLM-playground/frontier-platform${PYTHONPATH:+:$PYTHONPATH}
PY=/home/support/dev-macrohard/LLM-playground/frontier-platform/.venv/bin/python

if [ ! -x "$PY" ]; then
    echo "ERROR: frontier-platform venv not found at $PY" >&2
    exit 1
fi

mkdir -p out
exec "$PY" run.py "$@"
