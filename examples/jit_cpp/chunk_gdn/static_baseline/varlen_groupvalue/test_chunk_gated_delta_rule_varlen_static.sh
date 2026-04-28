#!/usr/bin/env bash
# Compile and run static PTO varlen chunk_gated_delta_rule kernels (bisheng + ctypes).
# Prefer latest PTO headers from the pto-isa tree used by TileLang dumps:
#   export PTO_LIB_PATH=/sources/pto-isa
set -euo pipefail
export PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
cd "$(dirname "$0")"
./compile_varlen_kernels.sh
python3 run_chunk_gated_delta_rule_varlen_static.py --profile H48
python3 run_chunk_gated_delta_rule_varlen_static.py --profile H32
