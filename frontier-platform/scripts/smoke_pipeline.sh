#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
exec .venv/bin/python scripts/smoke_pipeline.py
