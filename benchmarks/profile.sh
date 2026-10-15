#!/usr/bin/env bash
set -euo pipefail
# msprof may resolve a Python symlink outside the venv. Keep a real shell entry.
: "${XDSL_NPU_PYTHON:?Set XDSL_NPU_PYTHON to the workspace venv Python}"
benchmark_root="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$benchmark_root${PYTHONPATH:+:$PYTHONPATH}"
# First argument is a module, e.g. benchmarks.gemm.run; remaining args pass through.
exec "$XDSL_NPU_PYTHON" -m "$@"
