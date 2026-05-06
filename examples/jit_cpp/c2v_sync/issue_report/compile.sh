#!/usr/bin/env bash
# Compile repro kernels under issue_report/.
# Requires:
#   - ASCEND_TOOLKIT_HOME
#   - PTO_LIB_PATH

set -euo pipefail

if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
  echo "ERROR: ASCEND_TOOLKIT_HOME is not set" >&2
  exit 1
fi

if [[ -z "${PTO_LIB_PATH:-}" ]]; then
  echo "ERROR: PTO_LIB_PATH is not set" >&2
  exit 1
fi

COMMON_FLAGS=(
  -fPIC -shared -xcce -O2 -std=c++17
  --npu-arch=dav-2201
  -DMEMORY_BASE
  -I"${PTO_LIB_PATH}/include"
  -I"${ASCEND_TOOLKIT_HOME}/include"
  -I"${ASCEND_TOOLKIT_HOME}/pkg_inc"
  -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime"
  -I"${ASCEND_TOOLKIT_HOME}/pkg_inc/profiling"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=== Compiling TSYNC repro ==="
bisheng "${COMMON_FLAGS[@]}" ./repro_tsync_issue.cpp -o ./repro_tsync_issue_lib.so

echo "=== Compiling TPipe repro (native Producer::record path) ==="
bisheng "${COMMON_FLAGS[@]}" -DREPRO_USE_NATIVE_TPIPE_RECORD=1 \
  ./repro_tpipe_issue.cpp -o ./repro_tpipe_native_lib.so

echo "=== Compiling TPipe repro (workaround PIPE_MTE3 path) ==="
bisheng "${COMMON_FLAGS[@]}" -DREPRO_USE_NATIVE_TPIPE_RECORD=0 \
  ./repro_tpipe_issue.cpp -o ./repro_tpipe_workaround_lib.so

echo "Done."
