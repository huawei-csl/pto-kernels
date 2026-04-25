# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
.PHONY: help clean setup setup_once build-wheel build_wheel build_cmake install test test-npu check bootstrap check-env bench-dry-run docs

help:
	@echo "Common pto-kernels targets:"
	@echo "  make setup       Install Python requirements for local development"
	@echo "  make test        Run local tests that do not require NPU/custom ops"
	@echo "  make test-npu    Run NPU/custom-op tests explicitly"
	@echo "  make check       Run local tests plus lightweight repository checks"
	@echo "  make build-wheel Build a Python wheel"
	@echo "  make bootstrap   Clone pinned external repos into external/src"
	@echo "  make check-env   Validate the local CANN/PTO toolchain"
	@echo "  make docs        Build Doxygen documentation"

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/

setup:
	pip3 install -r requirements.txt

setup_once:
	$(MAKE) setup
	pip3 install torch-npu==2.8.0.post2 --extra-index-url https://download.pytorch.org/whl/cpu

build_cmake: clean
	bash scripts/build.sh

build-wheel:
	export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v  . --extra-index-url https://download.pytorch.org/whl/cpu

build_wheel: build-wheel

install:
	python3 -m pip install --force-reinstall pto_kernels-*.whl


docs:
	doxygen doxygen/Doxyfile

test:
	pytest -v \
		tests/test_bench_specs.py \
		tests/test_env_utils.py \
		tests/test_registry_inventory.py \
		tests/test_workflow_docs.py

test-npu:
	pytest -v tests/ -m npu --run-npu

check: test
	git diff --check

bootstrap:
	bash scripts/bootstrap_workspace.sh

check-env:
	bash -lc 'source scripts/source_env.sh && python3 scripts/check_env.py --strict'

bench-dry-run:
	PYTHONPATH=python python3 -m pto_kernels.bench.runner bench/specs/posembedding/apply_rotary_pos_emb.yaml --dry-run
