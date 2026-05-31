#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/frontier-platform"${PYTHONPATH:+:$PYTHONPATH}
PY="$REPO_ROOT/frontier-platform/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "ERROR: frontier-platform venv not found at $PY" >&2
    exit 1
fi

"$PY" -c "import tokenizers" 2>/dev/null || "$PY" -m pip install tokenizers --quiet
"$PY" -c "import requests" 2>/dev/null || "$PY" -m pip install requests --quiet

mkdir -p out
exec "$PY" run.py "$@"
