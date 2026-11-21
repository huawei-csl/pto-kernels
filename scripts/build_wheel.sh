#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
#
# Builds the pto-kernels wheel through scikit-build-core, with the same CANN
# environment and the same SOC_VERSION default as scripts/build.sh.
#
# Usage: scripts/build_wheel.sh [-v|--soc-version <SoC>]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

SHORT=v:,
LONG=soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done
# The flag above takes precedence over the environment. SOC_VERSION and
# ASCEND_CANN_PACKAGE_PATH are read back by the [tool.scikit-build] defines in
# pyproject.toml (the base addressing mode is derived from SOC_VERSION by
# CMakeLists.txt); ascend_env.sh also sources the CANN setenv.bash so that
# bisheng is on PATH for the kernel compilation.
export SOC_VERSION

# shellcheck source=./ascend_env.sh
source "${SCRIPT_DIR}/ascend_env.sh"

if ! command -v bisheng >/dev/null 2>&1; then
    echo "[ERROR] bisheng not found on PATH after sourcing ${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash"
    exit 1
fi

# See https://docs.pytorch.org/cppdocs/installing.html
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Unix Makefiles}"

set -e
pip wheel -v "${PARENT_DIR}" --extra-index-url https://download.pytorch.org/whl/cpu "$@"
