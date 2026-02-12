.PHONY: clean setup_once build_wheel install test

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/

setup_once:
	pip3 install -r requirements.txt
	pip3 install torch-npu==2.8.0.post2 --extra-index-url https://download.pytorch.org/whl/cpu # https://github.com/sgl-project/sgl-kernel-npu/pull/326

build_wheel:
	export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v  . --extra-index-url https://download.pytorch.org/whl/cpu

install: build_wheel
	bash scripts/install-wheel.sh

test:
	pytest -v tests/
