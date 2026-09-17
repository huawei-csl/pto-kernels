#!/usr/bin/env bash
# Run A5 pure-vector kernels under CANN Simulator (cannsim record, Ascend950).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A5_DIR="${SCRIPT_DIR}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/bin/setenv.bash"

SOC="${CANNSIM_SOC:-Ascend950}"
OUTPUT_DIR="${A5_DIR}/outputs/cannsim_vec"
mkdir -p "${OUTPUT_DIR}"
ulimit -n 65535

cd "${A5_DIR}"

USER_OPTS="${*}"
if [[ -z "${USER_OPTS}" ]]; then
  USER_OPTS="--kernel silu --mode bench --num-elements 128 --label smoke --skip-correctness"
fi

echo "==> cannsim record (${SOC})"
echo "    ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "    user options: ${USER_OPTS}"

cannsim record \
  -s "${SOC}" \
  -o "${OUTPUT_DIR}" \
  "${A5_DIR}/run_cannsim_entry.sh" \
  -u "${USER_OPTS}"
CANNSIM_RC=$?

# cannsim may segfault during teardown after Python wrote JSON; accept if output exists.
if [[ "${CANNSIM_RC}" -ne 0 ]] && [[ "${USER_OPTS}" == *"--output-json"* ]]; then
  OUT_JSON="$(python3 - "${USER_OPTS}" <<'PY'
import shlex, sys
args = shlex.split(sys.argv[1])
for i, a in enumerate(args):
    if a == "--output-json" and i + 1 < len(args):
        print(args[i + 1])
        break
PY
)"
  if [[ -n "${OUT_JSON}" && -f "${OUT_JSON}" ]]; then
    if python3 - "${OUT_JSON}" <<'PY'
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
rows = data.get("results", [])
sys.exit(0 if rows and rows[0].get("sim_wall_s") is not None else 1)
PY
    then
      echo "==> cannsim exited ${CANNSIM_RC} but ${OUT_JSON} is valid; treating as success"
      exit 0
    fi
  fi
fi
exit "${CANNSIM_RC}"
