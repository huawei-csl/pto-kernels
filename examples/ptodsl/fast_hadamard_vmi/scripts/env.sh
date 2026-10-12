#!/usr/bin/env bash
# Shared CANN + PTOAS environment for vmi_demo kernel scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"
PTOAS_ROOT="${PTOAS_ROOT:-${REPO_ROOT}/third_party/ptoas}"

# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
# shellcheck disable=SC1091
source "${PTOAS_ROOT}/scripts/ptoas_env.sh"

export ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
ARCH=$(uname -m)-linux
CANN="${ASCEND_HOME_PATH}"
export LD_LIBRARY_PATH="${CANN}/${ARCH}/simulator/dav_3510/camodel:${CANN}/${ARCH}/simulator/dav_3510/lib:${CANN}/${ARCH}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# Prefer built ptoas under the submodule; allow override for local builds.
if [ -d "${PTOAS_ROOT}/build/tools/ptoas" ]; then
  export PATH="${PTOAS_ROOT}/build/tools/ptoas:${PATH}"
fi
export PTOAS_HOST_TARGET_CPU="${PTOAS_HOST_TARGET_CPU:-znver3}"
unset CCC_OVERRIDE_OPTIONS || true
