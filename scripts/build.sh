#!/bin/bash


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
# The flag above takes precedence over the environment. The base addressing
# mode is derived from SOC_VERSION by CMakeLists.txt, so it is not passed here.
# ascend_env.sh also sources the CANN setenv.bash.
export SOC_VERSION

# shellcheck source=./ascend_env.sh
source "${SCRIPT_DIR}/ascend_env.sh"

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
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
      -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH}"

cmake --build build  -j
