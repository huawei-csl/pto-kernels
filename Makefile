# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
.PHONY: clean setup_once build_cmake build_wheel install docs test test_tri_inv

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/

setup_once:
	pip3 install -r requirements.txt
	pip3 install torch-npu==2.8.0.post2 --extra-index-url https://download.pytorch.org/whl/cpu

build_cmake: clean
	bash scripts/build.sh

build_wheel:
	export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v  . --extra-index-url https://download.pytorch.org/whl/cpu

install:
	python3 -m pip install --force-reinstall pto_kernels-*.whl

docs:
	doxygen doxygen/Doxyfile

test:
	pytest tests/

test_tri_inv:
	pytest tests/test_tri_inv_*.py
