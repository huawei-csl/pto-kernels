#!/usr/bin/env bash
# Run A5 pure-vector kernels under msprof op simulator (Ascend950PR_9599).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A5_DIR="${SCRIPT_DIR}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/bin/setenv.bash"

SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export LD_LIBRARY_PATH="${SIM_LIB}:${LD_LIBRARY_PATH:-}"
ulimit -n 65535

TIMEOUT="${MSPROF_TIMEOUT:-30}"
OUTPUT_DIR="${A5_DIR}/outputs/msprof_vec"
mkdir -p "${OUTPUT_DIR}"

cd "${A5_DIR}"

echo "==> msprof op simulator (Ascend950PR_9599)"
echo "    ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "    timeout=${TIMEOUT} min"

msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --timeout="${TIMEOUT}" \
  --output="${OUTPUT_DIR}" \
  python3 "${A5_DIR}/vec_sim.py" "$@"
