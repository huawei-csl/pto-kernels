#!/usr/bin/env bash
# Shared camodel asset setup + python entry for cannsim.
# Caller must set:
#   KERNEL_ROOT  — kernel directory (cwd for the test)
#   SCRIPT_DIR   — real scripts/ dir under that kernel (cannsim collects here)
set -euo pipefail

: "${KERNEL_ROOT:?KERNEL_ROOT must be set}"
: "${SCRIPT_DIR:?SCRIPT_DIR must be set}"

mkdir -p "${KERNEL_ROOT}/log_ca"
rm -f "${KERNEL_ROOT}/instr.bin"
find "${KERNEL_ROOT}/log_ca" -mindepth 1 -delete 2>/dev/null \
  || rm -rf "${KERNEL_ROOT}/log_ca"/* 2>/dev/null || true

for asset in log_ca instr.bin; do
  target="${KERNEL_ROOT}/${asset}"
  link="${SCRIPT_DIR}/${asset}"
  if [ -e "${link}" ] && [ ! -L "${link}" ]; then
    if [ -d "${link}" ]; then
      rmdir "${link}" 2>/dev/null || rm -rf "${link}"
    else
      rm -f "${link}"
    fi
  fi
  ln -sfn "${target}" "${link}"
done

cd "${KERNEL_ROOT}"
exec python3 "$@"
