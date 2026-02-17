#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

SHORT=v:,
LONG=soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
SOC_VERSION="Ascend910B4"

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

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH="$HOME"/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
# shellcheck source=/dev/null
source "$_ASCEND_INSTALL_PATH"/bin/setenv.bash
echo "Current compile soc version is ${SOC_VERSION}"

TORCH_DIR=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path, end='')")

set -e
rm -rf build
mkdir -p build
cmake -S "${PARENT_DIR}" \
      -B build \
      -DCMAKE_PREFIX_PATH="${TORCH_DIR}" \
      -DSOC_VERSION="${SOC_VERSION}" \
      -DASCEND_CANN_PACKAGE_PATH="${_ASCEND_INSTALL_PATH}"

cmake --build build  -j
