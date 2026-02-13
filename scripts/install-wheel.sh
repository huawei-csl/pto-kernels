#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

AARCH=$(uname -i)
PY_VER=$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("SOABI").split("-")[1])')

python3 -m pip install --force-reinstall "pto_kernels-0.1.0-cp${PY_VER}-cp${PY_VER}-linux_${AARCH}.whl"
