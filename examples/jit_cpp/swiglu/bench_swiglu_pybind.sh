#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SOC_VERSION="${SOC_VERSION:-Ascend910B2}"
CSV_DIR="${CSV_DIR:-outputs/csv/pybind}"
PLOT_DIR="${PLOT_DIR:-outputs/plots/pybind}"
SKIP_BUILD="${SKIP_BUILD:-0}"
RUN_PLOT="${RUN_PLOT:-1}"

if [[ -n "${ASCEND_INSTALL_PATH:-}" ]]; then
    ASCEND_ENV_ROOT="${ASCEND_INSTALL_PATH}"
elif [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    ASCEND_ENV_ROOT="${ASCEND_HOME_PATH}"
elif [[ -d "${HOME}/Ascend/ascend-toolkit/latest" ]]; then
    ASCEND_ENV_ROOT="${HOME}/Ascend/ascend-toolkit/latest"
else
    ASCEND_ENV_ROOT="/usr/local/Ascend/ascend-toolkit/latest"
fi

if [[ -f "${ASCEND_ENV_ROOT}/bin/setenv.bash" ]]; then
    # shellcheck source=/dev/null
    source "${ASCEND_ENV_ROOT}/bin/setenv.bash"
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
    PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir 2>/dev/null || true)"
    if [[ -n "${PYBIND11_CMAKE_DIR}" ]]; then
        if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
            export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${PYBIND11_CMAKE_DIR}"
        else
            export CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}"
        fi
    fi

    (
        cd "${REPO_ROOT}"
        bash scripts/build.sh --soc-version "${SOC_VERSION}"
    )
fi

PYBIND_MODULE="$(ls -t "${REPO_ROOT}"/pto_kernels_ops*.so 2>/dev/null | head -n 1 || true)"
if [[ -z "${PYBIND_MODULE}" ]]; then
    echo "Could not find pto_kernels_ops*.so. Run with SKIP_BUILD=0 or build first." >&2
    exit 1
fi

mkdir -p "${REPO_ROOT}/python/pto_kernels"
cp "${PYBIND_MODULE}" "${REPO_ROOT}/python/pto_kernels/"

export LD_LIBRARY_PATH="${REPO_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

(
    cd "${REPO_ROOT}"
    python "${SCRIPT_DIR}/bench_swiglu_pybind.py" --csv-dir "${CSV_DIR}" "$@"
)

if [[ "${RUN_PLOT}" == "1" ]]; then
    (
        cd "${REPO_ROOT}"
        python "${SCRIPT_DIR}/plot_swiglu.py" \
            --csv-dir "${CSV_DIR}" \
            --plot-dir "${PLOT_DIR}"
    )
fi
