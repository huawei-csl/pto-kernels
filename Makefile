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

.PHONY: clean setup_once build wheel install docs test test_tri_inv \
        dev dev-kernels dev-ops dev-sync dev-wheel dev-configure dev-info dev-wheel-install

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/ pto_kernels-*.whl

setup_once:
	pip3 install -r requirements.txt

build: clean
	bash scripts/build.sh

wheel: clean
	export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v . --extra-index-url https://download.pytorch.org/whl/cpu


# 'make compile_abs' compiles 'kernel_abs.cpp' into 'build/lib/libkernel_abs.so' without building the whole wheel package.
# This is useful for development and debugging of individual kernels.
compile_%:
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

compile_a5_%:
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
		$(CSRC_KERNEL_DIR)/$(KERNEL_SUBDIR)/kernel_$*.cpp \
		-o build/lib/$(KERNEL_SUBDIR)/libkernel_$*.so

install:
	python3 -m pip install --force-reinstall pto_kernels-*.whl

# ---------------------------------------------------------------------------
# Incremental development loop
#
# 'make wheel && make install' rebuilds everything from scratch (minutes).
# The targets below reuse the scikit-build-core CMake tree in
# build/<wheel-tag>/, so editing one kernel costs a single bisheng invocation
# plus a copy (seconds).
#
#   make dev          recompile changed kernels -> installed pto_kernels
#   make dev KERNEL_TARGET=pto_kernels_ops   (also rebuild the pybind module)
#   make dev-ops      rebuild kernels + pybind module -> installed pto_kernels
#   make dev-wheel    write a dev wheel to dev-wheelhouse/ (release wheel kept)
#
# Prerequisite: one full 'make wheel' (or 'make dev-configure') to create and
# configure build/<wheel-tag>/. Afterwards it is reused until 'make clean'.
# ---------------------------------------------------------------------------
PYTHON      ?= python3
JOBS        ?= $(shell nproc)
# scikit-build-core uses build-dir = build/{wheel_tag} (see pyproject.toml).
WHEEL_TAG   := $(shell $(PYTHON) -c "import sysconfig,sys;t='cp%d%d'%sys.version_info[:2];print(t+'-'+t+'-'+sysconfig.get_platform().replace('-','_').replace('.','_'))")
SKBUILD_DIR ?= build/$(WHEEL_TAG)
# Directory of the *installed* pto_kernels package (empty if not installed).
SITE_PKG    := $(shell $(PYTHON) -c "import os,pto_kernels;print(os.path.dirname(pto_kernels.__file__))" 2>/dev/null)
KERNEL_LIB  := libno_workspace_kernel.so
OPS_LIB     := $(shell $(PYTHON) -c "import sysconfig;print('pto_kernels_ops'+sysconfig.get_config_var('EXT_SUFFIX'))")

define require_skbuild
	@test -f $(SKBUILD_DIR)/CMakeCache.txt || { \
	  echo "[dev] No configured build tree at $(SKBUILD_DIR)."; \
	  echo "[dev] Run 'make wheel' once (or 'make dev-configure') first."; \
	  exit 1; }
endef

define require_installed
	@test -n "$(SITE_PKG)" || { \
	  echo "[dev] pto_kernels is not importable; run 'make wheel install' once."; \
	  exit 1; }
endef

dev-info:
	@echo "wheel tag   : $(WHEEL_TAG)"
	@echo "build tree  : $(SKBUILD_DIR)"
	@echo "installed to: $(SITE_PKG)"
	@echo "kernel lib  : $(KERNEL_LIB)"
	@echo "ops module  : $(OPS_LIB)"

# Configure (or reconfigure) the scikit-build-core tree without building a
# wheel. Mirrors the defines in pyproject.toml.
dev-configure:
	cmake -S . -B $(SKBUILD_DIR) \
	  -DPIP_INSTALL=ON \
	  -DSOC_VERSION="$${SOC_VERSION:-Ascend910B2}" \
	  -DBASE_MODE="$${BASE_MODE:-MEMORY}" \
	  -DASCEND_CANN_PACKAGE_PATH="$${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann}" \
	  -DCMAKE_PREFIX_PATH="$$($(PYTHON) -c 'import torch;print(torch.utils.cmake_prefix_path)')"

# Recompile only the kernel sources that changed.
dev-kernels:
	$(require_skbuild)
	cmake --build $(SKBUILD_DIR) --target no_workspace_kernel -j$(JOBS)

# Rebuild the pybind11 host module as well (needed after csrc/host changes).
dev-ops: dev-kernels
	$(require_skbuild)
	cmake --build $(SKBUILD_DIR) --target pto_kernels_ops -j$(JOBS)
	$(MAKE) dev-sync SYNC_OPS=1

# Copy the freshly built artefacts over the installed package. The installed
# pto_kernels_ops module resolves the kernel library through RUNPATH
# $ORIGIN/lib, so replacing lib/$(KERNEL_LIB) is enough for kernel-only edits.
dev-sync:
	$(require_installed)
	@mkdir -p $(SITE_PKG)/lib
	install -m 0755 $(SKBUILD_DIR)/lib/$(KERNEL_LIB) $(SITE_PKG)/lib/$(KERNEL_LIB)
	@echo "[dev] updated $(SITE_PKG)/lib/$(KERNEL_LIB)"
	@if [ -n "$(SYNC_OPS)" ]; then \
	  install -m 0755 $(SKBUILD_DIR)/$(OPS_LIB) $(SITE_PKG)/$(OPS_LIB) && \
	  echo "[dev] updated $(SITE_PKG)/$(OPS_LIB)"; \
	fi

# Default fast loop: rebuild changed kernels and push them into the install.
dev: dev-kernels dev-sync

# Take the wheel built by 'make wheel' as a template, swap in the freshly
# built artefacts and write the result to $(DEV_WHEELHOUSE)/. The source wheel
# is read-only here: release wheels are never modified in place.
DEV_WHEELHOUSE ?= dev-wheelhouse

dev-wheel:
	$(require_skbuild)
	@WHL=$$(ls -t pto_kernels-*.whl 2>/dev/null | head -1); \
	test -n "$$WHL" || { echo "[dev] no pto_kernels-*.whl found; run 'make wheel' once."; exit 1; }; \
	TMP=$$(mktemp -d); \
	$(PYTHON) -m wheel unpack -d $$TMP "$$WHL" >/dev/null; \
	PKG=$$(ls -d $$TMP/*/pto_kernels); \
	mkdir -p $$PKG/lib; \
	install -m 0755 $(SKBUILD_DIR)/lib/$(KERNEL_LIB) $$PKG/lib/$(KERNEL_LIB); \
	test -f $(SKBUILD_DIR)/$(OPS_LIB) && install -m 0755 $(SKBUILD_DIR)/$(OPS_LIB) $$PKG/$(OPS_LIB); \
	mkdir -p $(DEV_WHEELHOUSE); \
	$(PYTHON) -m wheel pack -d $(DEV_WHEELHOUSE) $$(ls -d $$TMP/*/) >/dev/null; \
	rm -rf $$TMP; \
	echo "[dev] source wheel (untouched): $$WHL"; \
	echo "[dev] dev wheel: $$(ls -t $(DEV_WHEELHOUSE)/pto_kernels-*.whl | head -1)"

# Install the dev wheel built above.
dev-wheel-install: dev-wheel
	$(PYTHON) -m pip install --force-reinstall --no-deps \
	  $$(ls -t $(DEV_WHEELHOUSE)/pto_kernels-*.whl | head -1)

docs:
	doxygen doxygen/Doxyfile

test:
	pytest -v tests/

test_tri_inv:
	pytest tests/test_tri_inv_*.py
