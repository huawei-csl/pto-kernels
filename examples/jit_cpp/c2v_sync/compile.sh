#!/usr/bin/env bash
# Compile both pto-isa C2V sync kernel versions.
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

echo "=== Compiling TSYNC version ==="
bisheng "${COMMON_FLAGS[@]}" ./sync_c2v_tsync.cpp -o ./sync_c2v_tsync_lib.so

echo "=== Compiling TPUSH/TPOP version ==="
bisheng "${COMMON_FLAGS[@]}" ./sync_c2v_tpushpop.cpp -o ./sync_c2v_tpushpop_lib.so

echo "Done."
