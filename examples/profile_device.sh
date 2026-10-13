#!/usr/bin/env bash
set -euo pipefail
# msprof canonicalizes a Python symlink and can lose its venv prefix. Keep
# the selected interpreter path intact by invoking it from this real script.
: "${XDSL_NPU_PYTHON:?Set XDSL_NPU_PYTHON to the workspace venv Python}"
exec "$XDSL_NPU_PYTHON" "$(dirname "$0")/run_device.py" "$@"
