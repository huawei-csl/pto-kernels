#!/usr/bin/env bash
# One-command on-device benchmark for fast_hadamard_a5 on an Ascend 950 (A5).
# Requires a real A5 device, torch + torch_npu, and bisheng (CANN toolkit).
source /usr/local/Ascend/cann-9.0.0/set_env.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
export ASCEND_HOME_PATH
cd "${SCRIPT_DIR}"
exec python3 benchmark.py "$@"
