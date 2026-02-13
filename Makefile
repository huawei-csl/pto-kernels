# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
.PHONY: clean setup_once build_wheel install test

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/

setup_once:
	pip3 install -r requirements.txt
	pip3 install torch-npu==2.8.0.post2 --extra-index-url https://download.pytorch.org/whl/cpu # https://github.com/sgl-project/sgl-kernel-npu/pull/326

build_cmake:
	bash scripts/build.sh

build_wheel:
	export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -vvv  . --extra-index-url https://download.pytorch.org/whl/cpu

install: build_wheel
	bash scripts/install-wheel.sh

test:
	pytest -v tests/
