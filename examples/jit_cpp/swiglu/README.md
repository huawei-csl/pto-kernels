# SwiGLU

This directory contains JIT and CMake/pybind demos for the PTO `SwiGLU`
kernel, plus validation and benchmark workflows against `torch_npu.npu_swiglu`.

The source file here is a symlink to
`../../../csrc/kernel/swiglu.cpp`, so edits made from the demo
directory also update the CMake/pybind kernel source.

Outputs:
- benchmark CSVs: `outputs/csv/`
- plots: `outputs/plots/`
- JIT shared objects: `outputs/so/`

The benchmark reports:
- runtime in microseconds
- effective elementwise TOPS
- PTO speedup over `torch_npu.npu_swiglu`

The TOPS estimate is not a real FLOPS metric: `exp` and `div` are counted as
one effective elementwise operation each so that PTO and `torch_npu` can be
compared on the same scale. The PTO kernel executes `5` vector operations per
output element:
- negate
- exp
- add
- divide
- multiply by the gate

Suggested workflow:

```bash
pytest test_swiglu.py -q --npu npu:0
python bench_swiglu.py --npu 0
python plot_swiglu.py
```

JIT benchmark without repeated Python stream lookup:

```bash
python bench_swiglu.py --cache-stream --npu npu:0 --csv-dir outputs/csv/cached_stream
python plot_swiglu.py --csv-dir outputs/csv/cached_stream --plot-dir outputs/plots/cached_stream
```

CMake/pybind benchmark:

```bash
./bench_swiglu_pybind.sh
```

The script builds the CMake extension, copies `pto_kernels_ops*.so` into
`python/pto_kernels`, sets `LD_LIBRARY_PATH` and `PYTHONPATH`, runs
`bench_swiglu_pybind.py`, then plots the CSV. By default it builds for
`Ascend910B2` and writes to `outputs/csv/pybind` and `outputs/plots/pybind`.

Useful overrides:

```bash
SOC_VERSION=Ascend910B4 ./bench_swiglu_pybind.sh
SKIP_BUILD=1 ./bench_swiglu_pybind.sh --batches 64 --hidden-dims 1024
RUN_PLOT=0 ./bench_swiglu_pybind.sh --warmup 1 --repeats 10 --trials 1
CSV_DIR=outputs/csv/pybind_smoke PLOT_DIR=outputs/plots/pybind_smoke ./bench_swiglu_pybind.sh
```
