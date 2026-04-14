#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_PACKAGE_DIR="${REPO_ROOT}/python/pto_kernels"

SOC_VERSION="${SOC_VERSION:-Ascend910B2}"
CSV_DIR="${CSV_DIR:-outputs/csv/pybind}"
PLOT_DIR="${PLOT_DIR:-outputs/plots/pybind}"
SKIP_BUILD="${SKIP_BUILD:-0}"
RUN_PLOT="${RUN_PLOT:-1}"

prepend_path() {
    local var_name="$1"
    local entry="$2"
    local current_value="${!var_name:-}"

    if [[ -z "${current_value}" ]]; then
        export "${var_name}=${entry}"
    else
        case ":${current_value}:" in
            *":${entry}:"*) ;;
            *) export "${var_name}=${entry}:${current_value}" ;;
        esac
    fi
}

resolve_ascend_root() {
    if [[ -n "${ASCEND_INSTALL_PATH:-}" ]]; then
        echo "${ASCEND_INSTALL_PATH}"
    elif [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
        echo "${ASCEND_HOME_PATH}"
    elif [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]]; then
        echo "${ASCEND_TOOLKIT_HOME}"
    elif [[ -d "/usr/local/Ascend/cann-8.5.0" ]]; then
        echo "/usr/local/Ascend/cann-8.5.0"
    elif [[ -d "${HOME}/Ascend/ascend-toolkit/latest" ]]; then
        echo "${HOME}/Ascend/ascend-toolkit/latest"
    else
        echo "/usr/local/Ascend/ascend-toolkit/latest"
    fi
}

stage_pybind_module() {
    local pybind_module
    pybind_module="$(ls -t "${REPO_ROOT}"/pto_kernels_ops*.so 2>/dev/null | head -n 1 || true)"
    if [[ -z "${pybind_module}" ]]; then
        echo "Could not find ${REPO_ROOT}/pto_kernels_ops*.so after build." >&2
        exit 1
    fi

    mkdir -p "${PYTHON_PACKAGE_DIR}"
    if [[ ! -f "${PYTHON_PACKAGE_DIR}/__init__.py" ]]; then
        printf 'from .pto_kernels_ops import *  # noqa: F403\n' >"${PYTHON_PACKAGE_DIR}/__init__.py"
    fi
    cp -f "${pybind_module}" "${PYTHON_PACKAGE_DIR}/"
}

ensure_runtime_paths() {
    local torch_lib
    local torch_npu_lib

    torch_lib="$(python - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
    torch_npu_lib="$(python - <<'PY'
from pathlib import Path
import torch_npu
print(Path(torch_npu.__file__).resolve().parent / "lib")
PY
)"

    prepend_path LD_LIBRARY_PATH "${REPO_ROOT}/build/lib"
    prepend_path LD_LIBRARY_PATH "${torch_lib}"
    prepend_path LD_LIBRARY_PATH "${torch_npu_lib}"
    prepend_path PYTHONPATH "${REPO_ROOT}/python"
}

if [[ -n "${ASCEND_INSTALL_PATH:-}" ]]; then
    ASCEND_ENV_ROOT="${ASCEND_INSTALL_PATH}"
else
    ASCEND_ENV_ROOT="$(resolve_ascend_root)"
fi

if [[ -f "${ASCEND_ENV_ROOT}/bin/setenv.bash" ]]; then
    # shellcheck source=/dev/null
    source "${ASCEND_ENV_ROOT}/bin/setenv.bash"
else
    echo "CANN setenv.bash not found at ${ASCEND_ENV_ROOT}/bin/setenv.bash" >&2
    exit 1
fi

export ASCEND_INSTALL_PATH="${ASCEND_ENV_ROOT}"
export ASCEND_HOME_PATH="${ASCEND_ENV_ROOT}"
export ASCEND_CANN_PACKAGE_PATH="${ASCEND_ENV_ROOT}"

if [[ "${SKIP_BUILD}" != "1" ]]; then
    PYBIND11_CMAKE_DIR="$(python -m pybind11 --cmakedir 2>/dev/null || true)"
    if [[ -n "${PYBIND11_CMAKE_DIR}" ]]; then
        prepend_path CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}"
    fi

    echo "Building pto_kernels_ops for ${SOC_VERSION} using ${ASCEND_ENV_ROOT}"
    (
        cd "${REPO_ROOT}"
        bash scripts/build.sh --soc-version "${SOC_VERSION}"
    )
fi

stage_pybind_module
ensure_runtime_paths

echo "Running SwiGLU pybind benchmark"
(
    cd "${REPO_ROOT}"
    python "${SCRIPT_DIR}/bench_swiglu_pybind.py" --csv-dir "${CSV_DIR}" "$@"
)

if [[ "${RUN_PLOT}" == "1" ]]; then
    echo "Plotting SwiGLU pybind benchmark"
    (
        cd "${REPO_ROOT}"
        python "${SCRIPT_DIR}/plot_swiglu.py" \
            --csv-dir "${CSV_DIR}" \
            --plot-dir "${PLOT_DIR}"
    )
fi
