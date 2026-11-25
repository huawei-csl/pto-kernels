# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
PTO_LIB_PATH    ?= $(ASCEND_TOOLKIT_HOME)
CSRC_KERNEL_DIR := csrc/kernel
KERNEL_SUBDIR   ?= .

.PHONY: clean setup_once build wheel build_wheel install docs test test_tri_inv

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/ pto_kernels-*.whl

setup_once:
	pip3 install -r requirements.txt

build: clean
	bash scripts/build.sh

wheel: clean
	bash scripts/build-wheel.sh


# 'make compile_abs' compiles 'kernel_abs.cpp' into 'build/lib/libkernel_abs.so' without building the whole wheel package.
# This is useful for development and debugging of individual kernels.
# The default target is A5 (Ascend950); use 'make compile_a2_abs' for A2 (Ascend910B*).
compile_a2_%:
	mkdir -p build/lib/$(KERNEL_SUBDIR)/
	bisheng -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 \
		-I$(CSRC_KERNEL_DIR) \
		-I$(PTO_LIB_PATH)/include \
	        -I${ASCEND_TOOLKIT_HOME}/pkg_inc \
		-I${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime \
		-I${ASCEND_TOOLKIT_HOME}/pkg_inc/profiling \
		--npu-arch=dav-2201 \
		-Wno-ignored-attributes \
		$(CSRC_KERNEL_DIR)/$(KERNEL_SUBDIR)/kernel_$*.cpp \
		-o build/lib/$(KERNEL_SUBDIR)/libkernel_$*.so

# Compiles kernel_$(1).cpp for A5 (Ascend950).
define compile_a5_kernel
	mkdir -p build/lib/$(KERNEL_SUBDIR)/
	bisheng -fPIC -shared -xcce -DREGISTER_BASE -O2 -std=gnu++17 \
		-I$(CSRC_KERNEL_DIR) \
		-I$(PTO_LIB_PATH)/include \
	        -I${ASCEND_TOOLKIT_HOME}/pkg_inc \
		-I${ASCEND_TOOLKIT_HOME}/pkg_inc/runtime \
		-I${ASCEND_TOOLKIT_HOME}/pkg_inc/profiling \
		--cce-aicore-arch=dav-c310 \
		-mllvm -cce-aicore-stack-size=0x8000 \
		-mllvm -cce-aicore-function-stack-size=0x8000 \
		-Wno-ignored-attributes \
		$(CSRC_KERNEL_DIR)/$(KERNEL_SUBDIR)/kernel_$(1).cpp \
		-o build/lib/$(KERNEL_SUBDIR)/libkernel_$(1).so
endef

# 'compile_a5_<name>' is kept as an explicit alias of the default 'compile_<name>'.
compile_a5_%:
	$(call compile_a5_kernel,$*)

compile_%:
	$(call compile_a5_kernel,$*)

install:
	python3 -m pip install --force-reinstall pto_kernels-*.whl

docs:
	doxygen doxygen/Doxyfile

test:
	pytest tests/
