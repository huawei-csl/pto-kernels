#!/usr/bin/env bash
# Shared cannsim record + report helper.
# Usage: run_sim.sh <python_script> [output_dir]
#
# Derives the kernel root from the test script path:
#   <kernel>/test/foo.py  → KERNEL_ROOT=<kernel>
#   <kernel>/cce/test/foo.py → KERNEL_ROOT=<kernel> (parent of cce/)
set -euo pipefail

SHARED="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: run_sim.sh <python_script> [output_dir]
USAGE
  exit 1
}

SCRIPT="${1:-}"
[ -z "${SCRIPT}" ] && usage
SCRIPT="$(realpath "${SCRIPT}")"
[ ! -f "${SCRIPT}" ] && { echo "ERROR: ${SCRIPT} not found"; usage; }
shift || true

TEST_DIR="$(cd "$(dirname "${SCRIPT}")" && pwd)"
# test/ → kernel; cce/test/ → kernel (parent of cce)
if [ "$(basename "$(dirname "${TEST_DIR}")")" = "cce" ]; then
  KERNEL_ROOT="$(cd "${TEST_DIR}/../.." && pwd)"
else
  KERNEL_ROOT="$(cd "${TEST_DIR}/.." && pwd)"
fi

OUT_DIR="${1:-${KERNEL_ROOT}/sim_outputs/$(basename "$SCRIPT" .py)}"
mkdir -p "${OUT_DIR}"

# Ensure a real run_sim_entry.sh exists under kernel scripts/ for asset collection.
ENTRY="${KERNEL_ROOT}/scripts/run_sim_entry.sh"
if [ ! -f "${ENTRY}" ]; then
  echo "ERROR: missing ${ENTRY} (required real file for cannsim asset collection)" >&2
  exit 1
fi

echo "============================================"
echo "cannsim record: ${SCRIPT}"
echo "output: ${OUT_DIR}"
echo "============================================"

set +e
cannsim record \
  "${ENTRY}" \
  -s Ascend950 \
  --gen-report \
  -o "${OUT_DIR}" \
  -u "${SCRIPT}"
rc=$?
set -e

echo ""
echo "cannsim exit code: ${rc}"

CANNSIM_RUN="$(find "${OUT_DIR}" -maxdepth 1 -type d -name 'cannsim_*' 2>/dev/null | sort | tail -1 || true)"
if [ -n "${CANNSIM_RUN}" ]; then
  if [ -d "${KERNEL_ROOT}/log_ca" ] && [ ! -d "${CANNSIM_RUN}/log_ca" ]; then
    cp -a "${KERNEL_ROOT}/log_ca" "${CANNSIM_RUN}/log_ca"
  fi
  if [ -f "${KERNEL_ROOT}/instr.bin" ]; then
    dest="${CANNSIM_RUN}/instr.bin"
    if [ ! -f "${dest}" ] || [ -L "${dest}" ] || [ "${dest}" -ef "${KERNEL_ROOT}/instr.bin" ]; then
      tmp="${CANNSIM_RUN}/.instr.bin.tmp"
      cp -f "${KERNEL_ROOT}/instr.bin" "${tmp}"
      mv -f "${tmp}" "${dest}"
    fi
  fi
  if [ -f "${CANNSIM_RUN}/instr.bin" ]; then
    mkdir -p "${CANNSIM_RUN}/report"
    set +e
    cannsim report -e "${CANNSIM_RUN}" -o "${CANNSIM_RUN}/report" -n 0
    set -e
  fi
fi

echo ""
echo "--- VF cycle metrics (RVEC / instr_log) ---"
python3 "${SHARED}/cannsim_metrics.py" "${OUT_DIR}" || true
exit 0
