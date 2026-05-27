# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
PTO_LIB_PATH    ?= $(ASCEND_TOOLKIT_HOME)
CSRC_KERNEL_DIR := csrc/kernel

.PHONY: clean setup_once build wheel install docs test test_tri_inv

clean:
	rm -rf build/ dist/ extra-info/ *.so *.egg-info/ kernel_meta/ pto_kernels-*.whl

setup_once:
	pip3 install -r requirements.txt
	pip3 install torch-npu==2.9.0.post2 --extra-index-url https://download.pytorch.org/whl/cpu

build: clean
	bash scripts/build.sh

wheel: clean
	export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v  . --extra-index-url https://download.pytorch.org/whl/cpu


# 'make compile_abs' compiles 'kernel_abs.cpp' into 'libkernel_abs.so' without building the whole wheel package.
# This is useful for development and debugging of individual kernels.
compile_%:
	bisheng -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 \
		-I$(CSRC_KERNEL_DIR) \
		-I$(PTO_LIB_PATH)/include \
		--npu-arch=dav-2201 \
	        -Wno-ignored-attributes \
		$(CSRC_KERNEL_DIR)/kernel_$*.cpp \
		-o libkernel_$*.so

compile_cpu_%:
	g++-15 -fPIC -shared -D__CPU_SIM -std=c++20 \
		-I$(CSRC_KERNEL_DIR) \
		-I$(PTO_LIB_PATH)/include \
		-D_FORTIFY_SOURCE=2 \
		-Wno-macro-redefined \
		-Wno-ignored-attributes \
		-fstack-protector-strong \
		$(CSRC_KERNEL_DIR)/kernel_$*.cpp \
		-o libkernel_$*.so



install:
	python3 -m pip install --force-reinstall pto_kernels-*.whl

docs:
	doxygen doxygen/Doxyfile

test:
	pytest -v tests/

test_tri_inv:
	pytest tests/test_tri_inv_*.py
