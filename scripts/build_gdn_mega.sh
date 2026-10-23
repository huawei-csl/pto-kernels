#!/usr/bin/env bash
# Incremental GDN-only build. Run inside the CANN Docker environment.
set -eo pipefail
cd "$(dirname "$0")/.."
source /usr/local/Ascend/cann/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
if [[ ! -f build/CMakeCache.txt ]]; then
    cmake -S . -B build -G Ninja -DGDN_MEGA_ONLY=ON -DSOC_VERSION=Ascend910B2 \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch; import pybind11; print(torch.utils.cmake_prefix_path + ";" + pybind11.get_cmake_dir())')"
fi
time cmake --build build -j 4
extension_suffix=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
ln -sf "../../pto_kernels_ops${extension_suffix}" "python/pto_kernels/pto_kernels_ops${extension_suffix}"
