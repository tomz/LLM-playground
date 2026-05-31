#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Force the Tesla P100 (sm_60). NB: needs a torch build with sm_60 kernels
# (we use a dedicated .venv-p100 with torch 2.4.1+cu121).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/frontier-platform"${PYTHONPATH:+:$PYTHONPATH}
PY="$REPO_ROOT/frontier-platform/.venv-p100/bin/python"

if [ ! -x "$PY" ]; then
    echo "ERROR: P100 venv not found at $PY" >&2
    echo "Create it with: python3.11 -m venv $PY ... + torch==2.4.1+cu121" >&2
    exit 1
fi

"$PY" -c "import tokenizers" 2>/dev/null || "$PY" -m pip install tokenizers --quiet
"$PY" -c "import requests" 2>/dev/null || "$PY" -m pip install requests --quiet

mkdir -p out
exec "$PY" run.py "$@"
