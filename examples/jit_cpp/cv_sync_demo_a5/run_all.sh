#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate torch_npu_dev
fi

# shellcheck source=/dev/null
source /usr/local/Ascend/cann-9.0.0/set_env.sh

export NPU_DEVICE="${NPU_DEVICE:-npu:0}"
python3 run_stream.py
python3 run_matmul.py

