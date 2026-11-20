# shellcheck shell=bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
#
# Shared build environment for the CMake build (scripts/build.sh) and the wheel
# build (scripts/build_wheel.sh). Source it, do not execute it.
#
# It resolves the CANN installation, sources its setenv.bash (which puts the
# bisheng/ccec compilers on PATH) and exports the build knobs:
#
#   SOC_VERSION              target SoC          (default Ascend950)
#   BASE_MODE                derived from SOC_VERSION (REGISTER on Ascend950,
#                            MEMORY on Ascend910B/910_93/310P)
#   ASCEND_CANN_PACKAGE_PATH resolved CANN root
#
# SOC_VERSION is the only input knob; BASE_MODE is not read from the
# environment.

# Target SoC. The only input knob; overridable via the SOC_VERSION environment
# variable.
export SOC_VERSION="${SOC_VERSION:-Ascend950}"

# Base addressing mode, derived from the SoC: Ascend950 kernels use the
# register-base addressing path, earlier SoCs the memory-base one.
_SOC_LOWER="${SOC_VERSION,,}"
case "$_SOC_LOWER" in
ascend950*)
    BASE_MODE=REGISTER
    ;;
ascend910b* | ascend910_93* | ascend310p*)
    BASE_MODE=MEMORY
    ;;
*)
    echo "[ERROR] Unsupported SOC_VERSION: ${SOC_VERSION}"
    echo "[ERROR] Supported: Ascend950*, Ascend910B*, Ascend910_93*, Ascend310P*"
    # shellcheck disable=SC2317
    return 1 2>/dev/null || exit 1
    ;;
esac
export BASE_MODE
unset _SOC_LOWER

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
elif [ -d "$HOME/Ascend/cann/latest" ]; then
    _ASCEND_INSTALL_PATH="$HOME"/Ascend/cann/latest
else
    _ASCEND_INSTALL_PATH=/usr/local/Ascend/cann/latest
fi

if [ ! -f "$_ASCEND_INSTALL_PATH/bin/setenv.bash" ]; then
    echo "[ERROR] No CANN installation at ${_ASCEND_INSTALL_PATH}"
    echo "[ERROR] Set ASCEND_INSTALL_PATH (or ASCEND_HOME_PATH) to the CANN root."
    # shellcheck disable=SC2317
    return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$_ASCEND_INSTALL_PATH"/bin/setenv.bash

# Pin the CANN root for CMake instead of letting it guess.
export ASCEND_CANN_PACKAGE_PATH="$_ASCEND_INSTALL_PATH"

echo "Current compile soc version is ${SOC_VERSION}"
echo "Current base mode is ${BASE_MODE}"
echo "Current CANN package path is ${ASCEND_CANN_PACKAGE_PATH}"
