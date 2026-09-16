#!/usr/bin/env bash
# One-command on-device benchmark for mxfp4_matmul_a5 on an Ascend 950 (A5).
# Requires a real A5 device, torch + torch_npu, and bisheng (CANN toolkit).
#
# Source your own toolkit first to pick it: the vendor arm is torch_npu
# dispatching into libopapi from whichever toolkit is active, so that arm moves
# with it, and the MXFP4 matmul needs 9.1.0 or newer.
if [[ -z "${ASCEND_TOOLKIT_HOME:-}" && -z "${ASCEND_HOME_PATH:-}" ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ASCEND_HOME_PATH:=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
export ASCEND_HOME_PATH
cd "${SCRIPT_DIR}"
exec python3 benchmark.py "$@"
