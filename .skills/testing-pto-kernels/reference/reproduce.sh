#!/usr/bin/env bash
# Sequential smoke pass for the PTO kernel testing reference tree.
# Exits non-zero on the first unexpected failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
NPU_DEVICE="${NPU_DEVICE:-npu:0}"
RUN_WITH_TIMEOUT="${SCRIPT_DIR}/run_with_timeout.sh"

source "${ASCEND_HOME_PATH}/bin/setenv.bash"

SOC="$(python3 - <<'PY'
import acl
print(acl.get_soc_name())
PY
)"

echo "=== PTO reference smoke (SOC=${SOC}, NPU_DEVICE=${NPU_DEVICE}) ==="

run() {
  echo
  echo ">>> $*"
  "$@"
}

# --- A2A3 real device (910B hosts only) ---
if [[ "${SOC}" == Ascend910B* ]]; then
  run bash -c "cd '${SCRIPT_DIR}/dynamic_multi_core/a2a3' && NPU_DEVICE='${NPU_DEVICE}' '${RUN_WITH_TIMEOUT}' python3 run_kernel_ctypes.py --kernel all --n 4096 --m 128"
  run bash -c "cd '${SCRIPT_DIR}/dynamic_multi_core/a2a3' && NPU_DEVICE='${NPU_DEVICE}' '${RUN_WITH_TIMEOUT}' python3 run_mix_ctypes.py --kernel all --rounds 1"
  run bash -c "cd '${SCRIPT_DIR}/static_single_core/a2a3' && NPU_DEVICE='${NPU_DEVICE}' '${RUN_WITH_TIMEOUT}' python3 run_kernel_ctypes.py --kernel add"
else
  echo "Skipping A2A3 real-device tests (SOC=${SOC} is not Ascend910B*)"
fi

# --- A2A3 CA model ---
run bash -c "cd '${SCRIPT_DIR}/static_single_core/a2a3' && MSPROF_SOC_VERSION=Ascend910B2 MSPROF_TIMEOUT=120 ./run_sim.sh msprof --kernel add"
run bash -c "cd '${SCRIPT_DIR}/dynamic_multi_core/a2a3' && MSPROF_SOC_VERSION=Ascend910B2 MSPROF_TIMEOUT=120 ./run_sim.sh msprof --kernel add --n 4096"

# --- A5 CA model (always use simulator paths on 910B hosts) ---
run bash -c "cd '${SCRIPT_DIR}/dynamic_multi_core/a5' && MSPROF_TIMEOUT=60 ./run_sim.sh msprof --n 128 --block-dim 8"
run bash -c "cd '${SCRIPT_DIR}/static_single_core/a5' && MSPROF_TIMEOUT=60 ./run_sim.sh msprof --kernel add"
run bash -c "cd '${SCRIPT_DIR}/static_single_core/a5' && ./run_sim.sh linked"

if [[ "${SOC}" == Ascend910B* ]]; then
  echo
  echo "NOTE: A5 ./run_sim.sh direct is intentionally skipped on Ascend910B hosts."
  echo "      A5 kernels must use msprof, cannsim, or -lruntime_camodel here."
fi

echo
echo "=== PASS: reference smoke completed ==="
