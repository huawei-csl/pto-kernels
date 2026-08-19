#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

SHORT=v:,m:,
LONG=soc-version:,base-mode:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
# Target SoC. Overridable via the SOC_VERSION environment variable, or the
# --soc-version flag which takes precedence.
SOC_VERSION="${SOC_VERSION:-Ascend910B2}"
# Base addressing mode: MEMORY or REGISTER. Overridable via the BASE_MODE
# environment variable, or the --base-mode flag which takes precedence.
BASE_MODE="${BASE_MODE:-MEMORY}"

while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -m | --base-mode)
        BASE_MODE="$2"
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

BASE_MODE="${BASE_MODE^^}"
if [ "$BASE_MODE" != "MEMORY" ] && [ "$BASE_MODE" != "REGISTER" ]; then
    echo "[ERROR] BASE_MODE must be MEMORY or REGISTER, got: ${BASE_MODE}"
    exit 1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
else
    if [ -d "$HOME/Ascend/cann/latest" ]; then
        _ASCEND_INSTALL_PATH="$HOME"/Ascend/cann/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/cann/latest
    fi
fi
# shellcheck source=/dev/null
source "$_ASCEND_INSTALL_PATH"/bin/setenv.bash
echo "Current compile soc version is ${SOC_VERSION}"
echo "Current base mode is ${BASE_MODE}"

# See https://docs.pytorch.org/cppdocs/installing.html
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
export CMAKE_PREFIX_PATH


echo "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"


set -e
rm -rf build
mkdir -p build
cmake -S "${PARENT_DIR}" \
      -B build \
      -DSOC_VERSION="${SOC_VERSION}" \
      -DBASE_MODE="${BASE_MODE}" \
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
      -DASCEND_CANN_PACKAGE_PATH="${_ASCEND_INSTALL_PATH}"

cmake --build build  -j
