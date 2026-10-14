#!/usr/bin/env bash
set -euo pipefail
: "${XDSL_NPU_PYTHON:?Set XDSL_NPU_PYTHON to the workspace venv Python}"
exec "$XDSL_NPU_PYTHON" "$(dirname "$0")/run_attention.py" "$@"
