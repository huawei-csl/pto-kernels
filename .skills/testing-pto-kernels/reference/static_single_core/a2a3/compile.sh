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
    OUT="${BUILD_DIR}/libstatic_add_a2a3.so"
    ARCH_FLAGS=(--cce-aicore-arch=dav-c220-vec)
    ;;
  matmul)
    SRC="${SCRIPT_DIR}/matmul.cpp"
    OUT="${BUILD_DIR}/libstatic_matmul_a2a3.so"
    ARCH_FLAGS=(--cce-aicore-arch=dav-c220-cube)
    ;;
  matmul_add)
    SRC="${SCRIPT_DIR}/matmul_add.cpp"
    OUT="${BUILD_DIR}/libstatic_matmul_add_a2a3.so"
    ARCH_FLAGS=(--cce-aicore-arch=dav-c220)
    ;;
  acl_add)
    SRC="${SCRIPT_DIR}/add.cpp"
    OUT="${BUILD_DIR}/static_add_acl"
    ARCH_FLAGS=(--cce-aicore-arch=dav-c220-vec)
    ;;
  *)
    echo "usage: $0 {add|matmul|matmul_add}" >&2
    exit 2
    ;;
esac

if [[ "${KERNEL}" == "acl_add" ]]; then
  so_path="${BUILD_DIR}/libstatic_add_a2a3.so"
  "${BISHENG}" -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 \
    "${ARCH_FLAGS[@]}" \
    -I"${PTO_LIB_PATH}/include" \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/kernel/inc" \
    "${SRC}" -o "${so_path}"
  "${BISHENG}" -std=gnu++17 -O2 -Wno-macro-redefined -Wno-ignored-attributes \
    -xc++ -include stdint.h -include stddef.h \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/kernel/inc" \
    "${SCRIPT_DIR}/main.cpp" -o "${OUT}" \
    -L"${ASCEND_HOME_PATH}/lib64" -lascendcl -ldl -lstdc++ -lpthread \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64"
  echo "${OUT}"
  exit 0
fi

"${BISHENG}" -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 \
  "${ARCH_FLAGS[@]}" \
  -I"${PTO_LIB_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/pkg_inc" \
  -I"${ASCEND_HOME_PATH}/pkg_inc/runtime" \
  -I"${ASCEND_HOME_PATH}/pkg_inc/profiling" \
  -I"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/kernel/inc" \
  "${SRC}" -o "${OUT}"

echo "${OUT}"
