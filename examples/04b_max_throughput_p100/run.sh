#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Force the Tesla P100 (sm_60). NB: needs a torch build with sm_60 kernels
# (we use a dedicated .venv-p100 with torch 2.4.1+cu121).
export CUDA_VISIBLE_DEVICES=GPU-8d50f734-1f9e-0c0d-07b4-ef8b15cb7254
export PYTHONPATH=/home/support/dev-macrohard/LLM-playground/frontier-platform${PYTHONPATH:+:$PYTHONPATH}
PY=/home/support/dev-macrohard/LLM-playground/frontier-platform/.venv-p100/bin/python

if [ ! -x "$PY" ]; then
    echo "ERROR: P100 venv not found at $PY" >&2
    echo "Create it with: python3.11 -m venv $PY ... + torch==2.4.1+cu121" >&2
    exit 1
fi

"$PY" -c "import tokenizers" 2>/dev/null || "$PY" -m pip install tokenizers --quiet
"$PY" -c "import requests" 2>/dev/null || "$PY" -m pip install requests --quiet

mkdir -p out
exec "$PY" run.py "$@"
