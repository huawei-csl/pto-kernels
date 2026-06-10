#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-msprof}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFERENCE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_WITH_TIMEOUT="${REFERENCE_DIR}/run_with_timeout.sh"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
cd "${SCRIPT_DIR}"

case "${MODE}" in
  msprof)
    export PTO_SIMULATOR=1
    SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
    export LD_LIBRARY_PATH="${SIM_LIB}:${LD_LIBRARY_PATH:-}"
    MSPROF_TIMEOUT="${MSPROF_TIMEOUT:-60}" msprof op simulator \
      --soc-version=Ascend950PR_9599 \
      --timeout="${MSPROF_TIMEOUT:-60}" \
      --output="${SCRIPT_DIR}/outputs/msprof_static" \
      python3 "${SCRIPT_DIR}/run_kernel_ctypes.py" "$@"
    ;;
  linked)
    export PTO_SIMULATOR=1
    sim_lib="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
    export LD_LIBRARY_PATH="${sim_lib}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${LD_LIBRARY_PATH:-}"
    exe="$(bash "${SCRIPT_DIR}/compile.sh" acl_add | tail -n 1)"
    "${RUN_WITH_TIMEOUT}" "${exe}"
    ;;
  cannsim)
    export PTO_SIMULATOR=1
    mkdir -p "${SCRIPT_DIR}/outputs/cannsim_static"
    USER_OPTS="--kernel add --output-json outputs/static_add_cannsim.json $*"
    set +e
    cannsim record -s "${CANNSIM_SOC:-Ascend950}" \
      -o "${SCRIPT_DIR}/outputs/cannsim_static" \
      "${SCRIPT_DIR}/run_sim_entry.sh" \
      -u "${USER_OPTS}"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      if [[ -f "${SCRIPT_DIR}/outputs/static_add_cannsim.json" ]]; then
        python3 - "${SCRIPT_DIR}/outputs/static_add_cannsim.json" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
sys.exit(0 if data.get("result") == "PASS" else 1)
PY
        echo "cannsim exited ${rc} after writing PASS JSON; treating as success"
        exit 0
      fi
      exit "${rc}"
    fi
    ;;
  direct)
    "${RUN_WITH_TIMEOUT}" python3 "${SCRIPT_DIR}/run_kernel_ctypes.py" "$@"
    ;;
  *)
    echo "usage: $0 {msprof|cannsim|linked|direct} [run_kernel_ctypes.py args]" >&2
    exit 2
    ;;
esac
