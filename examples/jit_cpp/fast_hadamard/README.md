## Fast Hadamard

### Usage

```bash
export PTO_LIB_PATH=${ASCEND_TOOLKIT_HOME}
cd examples/jit_cpp/fast_hadamard

# benchmark and write CSV files
python bench_hadamard.py --npu 0
python bench_quantize.py --npu 0
# compares fused PTO, separate PTO, and torch+torch_npu unfused
python bench_hadamard_quant.py --npu 0

# generate plots from CSV files
python plot_hadamard.py
python plot_quantize.py
python plot_hadamard_quant.py

# correctness tests
pytest test_hadamard.py -v --npu "npu:0"
pytest test_hadamard_quant.py -v --npu "npu:0"
pytest test_quantize.py -v --npu "npu:0"
```

### Output

Benchmark scripts save CSV files to `outputs/csv/`, including:

- `fht_pto_bd*.csv` from `bench_hadamard.py`
- `quantize_compare_bd*.csv` from `bench_quantize.py`
- `fht_quant_compare_bd*.csv` from `bench_hadamard_quant.py`

Plot script reads CSV files from `outputs/csv/` and saves figures to
`outputs/plots/`.

JIT-compiled shared libraries are saved to `outputs/so/`.

In `bench_hadamard_quant.py`, the unfused reference is
`hadamard_torch_stagewise(...) + torch_npu.npu_quantize(...)`, reported as
`torch_unfused_*` in the CSV and plots.
