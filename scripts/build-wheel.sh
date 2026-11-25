#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
# Builds a pto_kernels wheel natively on this host (no Docker).
#
# This is the local counterpart of 'scripts/gen-wheel.sh', which builds a
# manylinux wheel inside a container. Use '--install' to force-reinstall the
# resulting wheel into the active Python environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Target SoC. Auto-detected from the NPUs installed on this machine, and
# overridable via the SOC_VERSION environment variable or the --soc-version
# flag (which takes precedence). FALLBACK_SOC_VERSION is used only when no NPU
# can be queried, e.g. on a build-only host.
FALLBACK_SOC_VERSION=Ascend910B2
INSTALL=0

usage() {
    cat <<'EOF'
Usage: ./scripts/build-wheel.sh [--soc-version <version>] [--install]

  -v, --soc-version  Target SoC (default: detected via npu-smi, else Ascend910B2)
      --install      Force-reinstall the built wheel after the build
  -h, --help         Show this help
EOF
    exit 1
}

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

while [[ $# -gt 0 ]]; do
    case "$1" in
    -v | --soc-version)
        [ $# -ge 2 ] || usage
        SOC_VERSION="$2"
        shift 2
        ;;
    --install)
        INSTALL=1
        shift
        ;;
    -h | --help) usage ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        usage
        ;;
    esac
done

if [ -z "${SOC_VERSION:-}" ]; then
    if SOC_VERSION=$(detect_soc_version); then
        echo "[INFO] Detected SoC version from npu-smi: ${SOC_VERSION}"
    else
        SOC_VERSION="$FALLBACK_SOC_VERSION"
        echo "[WARN] Could not detect the SoC version, falling back to ${SOC_VERSION}"
    fi
fi
export SOC_VERSION

if [ -n "${ASCEND_INSTALL_PATH:-}" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
    _ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
elif [ -d "$HOME/Ascend/cann/latest" ]; then
    _ASCEND_INSTALL_PATH="$HOME"/Ascend/cann/latest
else
    _ASCEND_INSTALL_PATH=/usr/local/Ascend/cann/latest
fi

if [ -f "$_ASCEND_INSTALL_PATH/bin/setenv.bash" ]; then
    # shellcheck source=/dev/null
    source "$_ASCEND_INSTALL_PATH"/bin/setenv.bash
else
    echo "[WARN] No setenv.bash under ${_ASCEND_INSTALL_PATH}, continuing with the current environment"
fi
export ASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-$_ASCEND_INSTALL_PATH}"

echo "==> Building pto_kernels wheel"
echo "    repo        : ${REPO_DIR}"
echo "    soc version : ${SOC_VERSION}"
echo "    cann path   : ${ASCEND_CANN_PACKAGE_PATH}"
echo ""

cd "$REPO_DIR"
export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v . --extra-index-url https://download.pytorch.org/whl/cpu

echo ""
echo "==> Wheel written to ${REPO_DIR}:"
ls -1 "${REPO_DIR}"/pto_kernels-*.whl

if [ "$INSTALL" -eq 1 ]; then
    echo ""
    echo "==> Installing the wheel"
    make install
fi
