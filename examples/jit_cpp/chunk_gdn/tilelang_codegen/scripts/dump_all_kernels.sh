#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../kernels"
for py in \
  opt_gdn_chunk_cumsum.py \
  opt_gdn_chunk_h.py \
  opt_gdn_chunk_o.py \
  opt_gdn_chunk_scaled_dot_kkt.py \
  opt_gdn_wy_fast.py
do
  echo "Running ${py} ..."
  python3 "${py}"
done
echo "All kernels dumped."
