#!/bin/bash
# Generates a manylinux wheel inside a Docker container, replicating
# the steps from .github/workflows/python-packaging.yml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
ARCH="x86_64"
PY_VER="310"

usage() {
    echo "Usage: $0 [--arch x86_64|aarch64] [--py-ver 310|311]"
    echo ""
    echo "  --arch    Target architecture (default: x86_64)"
    echo "  --py-ver  Python version without dots (default: 310)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)    ARCH="$2";   shift 2 ;;
        --py-ver)  PY_VER="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

case "$PY_VER" in
    310) PYTHON_VERSION="py3.10" ;;
    311) PYTHON_VERSION="py3.11" ;;
    *)   echo "Unsupported --py-ver '$PY_VER'. Use 310 or 311."; exit 1 ;;
esac

IMAGE="quay.io/ascend/manylinux:9.0.0-910b-manylinux_2_28-${PYTHON_VERSION}"

echo "==> Building manylinux wheel"
echo "    image      : ${IMAGE}"
echo "    arch       : ${ARCH}"
echo "    python     : ${PYTHON_VERSION}"
echo "    repo       : ${REPO_DIR}"
echo ""

docker run --rm \
    --platform "linux/${ARCH/x86_64/amd64}" \
    -v "${REPO_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    bash -c "
set -euxo pipefail

# --- env setup (mirrors CI) ---
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/\$(uname -i)-linux/devlib

# --- install build deps ---
pip3 install pyyaml setuptools pytest packaging 'pybind11[global]'
pip3 install -r requirements.txt

# --- build wheel ---
make clean wheel

# --- install & inspect ---
pip install pto_kernels-*.whl

SITE_PACKAGES=\$(python3 -c 'import site; print(site.getsitepackages()[0])')
echo \"\$SITE_PACKAGES\"
yum install -y -q tree
tree \${SITE_PACKAGES}/pto_kernels

# --- show wheel info ---
auditwheel show pto_kernels*.whl

# --- repair wheel ---
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\${SITE_PACKAGES}/pto_kernels/:\${SITE_PACKAGES}/pto_kernels/lib:\${SITE_PACKAGES}/pto_kernels/lib64
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/\$(uname -i)-linux/devlib

auditwheel repair \\
    --plat manylinux_2_27_${ARCH} \\
    --exclude libtorch.so \\
    --exclude libtorch_cpu.so \\
    --exclude libc10.so \\
    --exclude libhccl.so \\
    --exclude libprofapi.so \\
    --exclude libtorch_npu.so \\
    --exclude 'libascend*.so' \\
    --exclude libc_sec.so \\
    --exclude liberror_manager.so \\
    --exclude libmmpa.so \\
    --exclude libmsprofiler.so \\
    --exclude libplatform.so \\
    --exclude libruntime.so \\
    --exclude libruntime_common.so \\
    --exclude libunified_dlog.so \\
    pto_kernels*.whl \\
    -w wheelhouse/

echo ''
echo '==> Repaired wheel written to wheelhouse/:'
ls wheelhouse/pto_kernels*manylinux*.whl
"
