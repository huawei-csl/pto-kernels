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
    mkdir -p "${SCRIPT_DIR}/outputs/msprof_add"
    RESULT_JSON="${SCRIPT_DIR}/outputs/msprof_add/result.json"
    rm -f "${RESULT_JSON}"
    MSPROF_TIMEOUT="${MSPROF_TIMEOUT:-60}" msprof op simulator \
      --soc-version=Ascend950PR_9599 \
      --timeout="${MSPROF_TIMEOUT:-60}" \
      --output="${SCRIPT_DIR}/outputs/msprof_add" \
      python3 "${SCRIPT_DIR}/run_kernel_ctypes.py" --n 128 --output-json "${RESULT_JSON}" "$@"
    python3 - "${RESULT_JSON}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
sys.exit(0 if data.get("result") == "PASS" else 1)
PY
    ;;
  cannsim)
    export PTO_SIMULATOR=1
    mkdir -p "${SCRIPT_DIR}/outputs/cannsim_add"
    USER_OPTS="--n 128 $*"
    set +e
    cannsim record -s "${CANNSIM_SOC:-Ascend950}" \
      -o "${SCRIPT_DIR}/outputs/cannsim_add" \
      "${SCRIPT_DIR}/run_sim_entry.sh" \
      -u "${USER_OPTS}"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 && "${USER_OPTS}" == *"--output-json"* ]]; then
      out_json="$(python3 - "${USER_OPTS}" <<'PY'
import shlex, sys
args = shlex.split(sys.argv[1])
for i, arg in enumerate(args):
    if arg == "--output-json" and i + 1 < len(args):
        print(args[i + 1])
        break
PY
)"
      if [[ -n "${out_json}" && -f "${out_json}" ]]; then
        python3 - "${out_json}" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
sys.exit(0 if data.get("result") == "PASS" else 1)
PY
        echo "cannsim exited ${rc} after writing PASS JSON; treating as success"
        exit 0
      fi
    fi
    exit "${rc}"
    ;;
  direct)
    "${RUN_WITH_TIMEOUT}" python3 "${SCRIPT_DIR}/run_kernel_ctypes.py" "$@"
    ;;
  *)
    echo "usage: $0 {msprof|cannsim|direct} [run_kernel_ctypes.py args]" >&2
    exit 2
    ;;
esac
