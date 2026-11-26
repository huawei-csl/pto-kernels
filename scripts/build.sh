#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

SHORT=v:,
LONG=soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
# Target SoC. Auto-detected from the NPUs installed on this machine, and
# overridable via the SOC_VERSION environment variable or the --soc-version
# flag (which takes precedence). FALLBACK_SOC_VERSION is used only when no NPU
# can be queried, e.g. on a build-only host.
FALLBACK_SOC_VERSION=Ascend910B2

# Print the chip name of the first NPU reported by npu-smi, e.g. Ascend910B2.
detect_soc_version() {
    command -v npu-smi >/dev/null 2>&1 || return 1

    local npu_id
    npu_id=$(npu-smi info -l 2>/dev/null |
        awk -F: '/NPU ID/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')
    [ -n "$npu_id" ] || npu_id=0

    local chip_name
    chip_name=$(npu-smi info -t board -i "$npu_id" 2>/dev/null |
        awk -F: '/Chip Name/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')
    case "$chip_name" in
    Ascend*)
        printf '%s\n' "$chip_name"
        ;;
    *)
        return 1
        ;;
    esac
}

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

if [ -z "$SOC_VERSION" ]; then
    if SOC_VERSION=$(detect_soc_version); then
        echo "[INFO] Detected SoC version from npu-smi: ${SOC_VERSION}"
    else
        SOC_VERSION="$FALLBACK_SOC_VERSION"
        echo "[WARN] Could not detect the SoC version, falling back to ${SOC_VERSION}"
    fi
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
      -DASCEND_CANN_PACKAGE_PATH="${_ASCEND_INSTALL_PATH}"

cmake --build build  -j
