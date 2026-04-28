#!/usr/bin/env bash
# After copying fresh dumps from ``tilelang_codegen/kernels/chunk_gated_delta_rule_varlen_H{32,48}.cpp``:
#   - Replace ``#include \"tl_templates/pto/common.h\"`` + duplicate pto include with ``#include \"common.h\"``.
#   - Replace ``tl::ascend_pto::`` with ``chunk_gdn_pto::``.
#   - Replace ``TSELS(g_exp_ub_pad, g_mask_ub_pad, g_exp_ub_pad, -CUDART_INF_F);`` with
#     ``pto::TSELS(g_exp_ub_pad, g_mask_ub_pad, g_exp_ub_pad, tmp_ub, -CUDART_INF_F);`` (pto-isa API).
set -euo pipefail
export PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
cd "$(dirname "$0")"
python3 - <<'PY'
from pto_static_common import compile_pto_kernel

compile_pto_kernel(
    "chunk_gated_delta_rule_varlen_H32_kernel.cpp",
    "chunk_gated_delta_rule_varlen_H32_static.so",
)
compile_pto_kernel(
    "chunk_gated_delta_rule_varlen_H48_kernel.cpp",
    "chunk_gated_delta_rule_varlen_H48_static.so",
)
print("compiled chunk_gated_delta_rule_varlen_H{32,48}_static.so")
PY
