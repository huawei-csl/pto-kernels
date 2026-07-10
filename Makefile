# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
PTO_LIB_PATH    ?= $(ASCEND_TOOLKIT_HOME)
CSRC_KERNEL_DIR := csrc/kernel

.PHONY: clean setup_once build wheel install docs test test_tri_inv compile_all_a5

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/ pto_kernels-*.whl

setup_once:
	pip3 install -r requirements.txt

build: clean
	bash scripts/build.sh

wheel: clean
	export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v --no-build-isolation . --extra-index-url https://download.pytorch.org/whl/cpu


# 'make compile_abs' compiles 'kernel_abs.cpp' into 'libkernel_abs.so' without building the whole wheel package.
# This is useful for development and debugging of individual kernels.
compile_%:
	mkdir -p build/lib/
	bisheng -fPIC -shared -xcce -DMEMORY_BASE -O2 -std=c++17 \
		-I$(CSRC_KERNEL_DIR) \
		-I$(PTO_LIB_PATH)/include \
		--npu-arch=dav-2201 \
		-Wno-ignored-attributes \
		$(CSRC_KERNEL_DIR)/kernel_$*.cpp \
		-o build/lib/libkernel_$*.so

compile_a5_%:
	mkdir -p build/lib/
	bisheng -fPIC -shared -xcce -DREGISTER_BASE -O2 -std=gnu++17 \
		-I$(CSRC_KERNEL_DIR) \
		-I$(PTO_LIB_PATH)/include \
		--cce-aicore-arch=dav-c310 \
		-mllvm -cce-aicore-stack-size=0x8000 \
		-mllvm -cce-aicore-function-stack-size=0x8000 \
		-Wno-ignored-attributes \
		$(CSRC_KERNEL_DIR)/kernel_$*.cpp \
		-o build/lib/libkernel_$*.so

compile_all_a5: compile_a5_abs compile_a5_batch_matrix_square compile_a5_gdn_chunk_cumsum \
	compile_a5_csr_gather compile_a5_gdn_chunk_h compile_a5_gdn_chunk_o \
	compile_a5_gdn_scaled_dot_kkt compile_a5_gdn_wy_fast compile_a5_scan_ul1 \
	compile_a5_simple_matmul compile_a5_swiglu compile_a5_tri_inv_col_sweep \
	compile_a5_tri_inv_ns compile_a5_tri_inv_rec_unroll compile_a5_tri_inv_trick

install:
	python3 -m pip install --force-reinstall pto_kernels-*.whl

docs:
	doxygen doxygen/Doxyfile

test:
	pytest -v tests/

test_tri_inv:
	pytest tests/test_tri_inv_*.py
