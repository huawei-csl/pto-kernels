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
    OUT="${BUILD_DIR}/libstatic_add_a5.so"
    ARCH="dav-c310-vec"
    ;;
  matmul)
    SRC="${SCRIPT_DIR}/matmul.cpp"
    OUT="${BUILD_DIR}/libstatic_matmul_a5.so"
    ARCH="dav-c310-cube"
    ;;
  matmul_add)
    SRC="${SCRIPT_DIR}/matmul_add.cpp"
    OUT="${BUILD_DIR}/libstatic_matmul_add_a5.so"
    ARCH="dav-c310"
    ;;
  acl_add)
    SRC="${SCRIPT_DIR}/add.cpp"
    OUT="${BUILD_DIR}/static_add_acl"
    ARCH="dav-c310-vec"
    ;;
  *)
    echo "usage: $0 {add|matmul|matmul_add|acl_add}" >&2
    exit 2
    ;;
esac

COMMON_KERNEL_FLAGS=(
  -I"${PTO_LIB_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/kernel/inc" \
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

if [[ "${KERNEL}" == "acl_add" ]]; then
  "${BISHENG}" "${COMMON_KERNEL_FLAGS[@]}" -c "${SRC}" -o "${BUILD_DIR}/acl_add_kernel.o"
  kernel_so="${BUILD_DIR}/libstatic_add_a5.so"
  "${BISHENG}" -fPIC -shared --cce-fatobj-link \
    -Wl,-soname,"$(basename "${kernel_so}")" \
    "${BUILD_DIR}/acl_add_kernel.o" -o "${kernel_so}"
  "${BISHENG}" \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}/kernel/inc" \
    -std=gnu++17 -O2 -Wno-macro-redefined -Wno-ignored-attributes \
    -xc++ -include stdint.h -include stddef.h \
    -c "${SCRIPT_DIR}/main.cpp" -o "${BUILD_DIR}/main.o"
  sim_lib="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
  "${BISHENG}" "${BUILD_DIR}/main.o" -o "${OUT}" \
    -L"${BUILD_DIR}" -lstatic_add_a5 \
    -L"${ASCEND_HOME_PATH}/lib64" -L"${sim_lib}" \
    -lruntime_camodel -lstdc++ -lascendcl -lm -lc_sec -ldl -lpthread \
    -Wl,-rpath,"${BUILD_DIR}:${ASCEND_HOME_PATH}/lib64:${sim_lib}"
else
  "${BISHENG}" "${COMMON_KERNEL_FLAGS[@]}" -c "${SRC}" -o "${BUILD_DIR}/${KERNEL}.o"
  "${BISHENG}" -fPIC -shared --cce-fatobj-link \
    -Wl,-soname,"$(basename "${OUT}")" \
    "${BUILD_DIR}/${KERNEL}.o" -o "${OUT}"
fi

echo "${OUT}"
