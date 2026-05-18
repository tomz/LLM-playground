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

"$PY" -c "import tokenizers" 2>/dev/null || "$PY" -m pip install tokenizers --quiet
"$PY" -c "import requests" 2>/dev/null || "$PY" -m pip install requests --quiet

mkdir -p out
exec "$PY" run.py "$@"
