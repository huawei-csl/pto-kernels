#!/usr/bin/env bash
set -euo pipefail

KERNEL="${1:-add}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
PTO_LIB_PATH="${PTO_LIB_PATH:-${ASCEND_HOME_PATH}}"
BISHENG="${ASCEND_HOME_PATH}/bin/bisheng"
if [[ ! -x "${BISHENG}" ]]; then
  BISHENG="$(command -v bisheng)"
fi

case "${KERNEL}" in
  add)
    SRC="${SCRIPT_DIR}/add.cpp"
    OUT="${BUILD_DIR}/libadd.so"
    ARCH_FLAGS=(--cce-aicore-arch=dav-c220-vec)
    ;;
  matmul|simple_matmul)
    SRC="${SCRIPT_DIR}/matmul.cpp"
    OUT="${BUILD_DIR}/libmatmul.so"
    ARCH_FLAGS=(--cce-aicore-arch=dav-c220-cube)
    ;;
  matmul_add|mix_c2v)
    SRC="${SCRIPT_DIR}/matmul_add.cpp"
    OUT="${BUILD_DIR}/libmatmul_add_c2v.so"
    ARCH_FLAGS=(--cce-aicore-arch=dav-c220)
    ;;
  add_matmul|mix_v2c)
    SRC="${SCRIPT_DIR}/add_matmul.cpp"
    OUT="${BUILD_DIR}/libadd_matmul_v2c.so"
    ARCH_FLAGS=(--cce-aicore-arch=dav-c220)
    ;;
  *)
    echo "usage: $0 {add|matmul|matmul_add|add_matmul}" >&2
    exit 2
    ;;
esac

"${BISHENG}" \
  -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 \
  "${ARCH_FLAGS[@]}" \
  -I"${PTO_LIB_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/pkg_inc" \
  -I"${ASCEND_HOME_PATH}/pkg_inc/runtime" \
  -I"${ASCEND_HOME_PATH}/pkg_inc/profiling" \
  -I"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/kernel/inc" \
  "${SRC}" -o "${OUT}"

echo "${OUT}"
