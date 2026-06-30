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
    OUT="${BUILD_DIR}/libadd_a5.so"
    ARCH="dav-c310-vec"
    ;;
  matmul)
    SRC="${SCRIPT_DIR}/matmul.cpp"
    OUT="${BUILD_DIR}/libmatmul_a5.so"
    ARCH="dav-c310-cube"
    ;;
  mix)
    OUT="${BUILD_DIR}/libmix_a5.so"
    ARCH="dav-c310"
    ;;
  *)
    echo "usage: $0 {add|matmul|mix}" >&2
    exit 2
    ;;
esac

COMMON_FLAGS=(
  -I"${PTO_LIB_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/kernel/inc" \
  -I"${SCRIPT_DIR}" \
  -I"${ASCEND_HOME_PATH}/pkg_inc" \
  -I"${ASCEND_HOME_PATH}/pkg_inc/profiling" \
  -I"${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
  -std=gnu++17 -O2 -Wno-macro-redefined -Wno-ignored-attributes \
  -fPIC -xcce -Xhost-start -Xhost-end \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  --cce-aicore-arch="${ARCH}" -DREGISTER_BASE
)

if [[ "${KERNEL}" == "add" || "${KERNEL}" == "matmul" ]]; then
  "${BISHENG}" "${COMMON_FLAGS[@]}" -c "${SRC}" -o "${BUILD_DIR}/${KERNEL}.o"
  "${BISHENG}" -fPIC -shared --cce-fatobj-link \
    -Wl,-soname,"$(basename "${OUT}")" \
    "${BUILD_DIR}/${KERNEL}.o" -o "${OUT}"
else
  kernel_objs=()
  for src in matmul_add.cpp add_matmul.cpp; do
    obj="${BUILD_DIR}/${src%.cpp}.o"
    "${BISHENG}" "${COMMON_FLAGS[@]}" -c "${SCRIPT_DIR}/${src}" -o "${obj}"
    kernel_objs+=("${obj}")
  done
  kernel_so="${BUILD_DIR}/libmix_a5_kernels.so"
  "${BISHENG}" -fPIC -shared --cce-fatobj-link \
    -Wl,-soname,"$(basename "${kernel_so}")" \
    "${kernel_objs[@]}" -o "${kernel_so}"
  "${BISHENG}" \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/kernel/inc" \
    -I"${SCRIPT_DIR}" \
    -std=gnu++17 -O2 -Wno-macro-redefined -Wno-ignored-attributes \
    -xc++ -include stdint.h -include stddef.h -fPIC \
    -c "${SCRIPT_DIR}/launch_api.cpp" -o "${BUILD_DIR}/launch_api.o"
  "${BISHENG}" "${BUILD_DIR}/launch_api.o" -shared -o "${OUT}" \
    -L"${BUILD_DIR}" -lmix_a5_kernels -lstdc++ -Wl,-rpath,"${BUILD_DIR}"
fi

echo "${OUT}"
