## Fast Hadamard

### Usage

```bash
export PTO_LIB_PATH=${ASCEND_TOOLKIT_HOME}
cd examples/jit_cpp/fast_hadamard

# benchmark and write CSV files
python bench_hadamard.py --npu 0
python bench_quantize.py --npu 0

# generate plots from CSV files
python plot_hadamard.py --npu npu:0
python plot_quantize.py --npu npu:0

# correctness tests
pytest test_hadamard.py -v --npu "npu:0"
```

### Output

Benchmark scripts save CSV files to `outputs/csv/`.

Plot script reads CSV files from `outputs/csv/` and saves figures to
`outputs/plots/`.

JIT-compiled shared libraries are saved to `outputs/so/`.
