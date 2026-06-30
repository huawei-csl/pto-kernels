#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-msprof}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_WITH_TIMEOUT="${REFERENCE_DIR}/run_with_timeout.sh"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
SOC_VERSION="${MSPROF_SOC_VERSION:-Ascend910B2}"
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
cd "${SCRIPT_DIR}"

case "${MODE}" in
  msprof)
    export PTO_SIMULATOR=1
    SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
    export LD_LIBRARY_PATH="${SIM_LIB}:${LD_LIBRARY_PATH:-}"
    MSPROF_TIMEOUT="${MSPROF_TIMEOUT:-120}" msprof op simulator \
      --soc-version="${SOC_VERSION}" \
      --timeout="${MSPROF_TIMEOUT}" \
      --output="${SCRIPT_DIR}/outputs/msprof_static" \
      python3 "${SCRIPT_DIR}/run_kernel_ctypes.py" "$@"
    ;;
  direct)
    "${RUN_WITH_TIMEOUT}" python3 "${SCRIPT_DIR}/run_kernel_ctypes.py" "$@"
    ;;
  *)
    echo "usage: $0 {msprof|direct} [run_kernel_ctypes.py args]" >&2
    exit 2
    ;;
esac
