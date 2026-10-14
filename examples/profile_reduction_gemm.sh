#!/usr/bin/env bash
set -euo pipefail
: "${XDSL_NPU_PYTHON:?Set the workspace venv Python path}"
exec "$XDSL_NPU_PYTHON" "$(dirname "$0")/run_reduction_gemm.py" "$@"
