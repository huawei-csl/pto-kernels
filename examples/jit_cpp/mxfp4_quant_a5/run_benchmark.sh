#!/usr/bin/env bash
# One-command on-device benchmark for mxfp4_quant_a5 on an Ascend 950 (A5).
# Requires a real A5 device, torch + torch_npu, and bisheng (CANN toolkit).
#
# Source your own toolkit first to pick it: the TQuant arm needs PTO 9.1.0, and
# torch_npu dispatches into libopapi_nn.so from whichever one is active, so the
# vendor arm moves with it. Falls back to 9.0.0, where the TQuant arm is skipped.
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" && -z "${ASCEND_HOME_PATH:-}" ]]; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
export ASCEND_HOME_PATH
cd "${SCRIPT_DIR}"
exec python3 benchmark.py "$@"
