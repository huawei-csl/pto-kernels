#!/usr/bin/env bash
# Compile both reproducer kernels.
# Requires: ASCEND_TOOLKIT_HOME and PTO_LIB_PATH to be set.

set -euo pipefail

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

echo "=== Compiling repro_builtin (TSync_Custom — buggy) ==="
bisheng "${COMMON_FLAGS[@]}" repro_builtin.cpp -o repro_builtin_lib.so

echo "=== Compiling repro_mytync (MyTSync — correct) ==="
bisheng "${COMMON_FLAGS[@]}" repro_mytync.cpp  -o repro_mytync_lib.so

echo "Done."
