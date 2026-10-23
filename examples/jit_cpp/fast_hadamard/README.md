## Fast Hadamard

### Layout

- `standard/`: plain Hadamard kernel, benchmark, plot, and test files
- `cube/`: the two cube-unit (matmul) kernels, each with the same five kinds of
  file as `standard/`, plus the entry point that picks between them:
  - `fast_hadamard_onesided.cpp`: `Y = X * H_N` as a single right
    multiplication, with `H_N = H_B (x) H_M` applied as fixed-size matmuls
    accumulated in L0C, for `N` in 64..512
  - `fast_hadamard_twosided.cpp`: `Y = H_R * X * H_C`, the balanced Kronecker
    split, for `N` in 256..32768
  - `jit_util_hadamard_auto.py`: one entry point that branches on `N` to
    whichever of the two covers it, for `N` in 64..32768. It owns no kernel of
    its own, so it has a host wrapper and a test suite but no benchmark or
    plot target
- `fuse_int8_quant/`: int8 quantize and fused Hadamard+int8 quant files
- `fuse_int4_quant/`: int4 quantize and fused Hadamard+int4 quant files
- shared helpers stay at the top level:
  - `bench_common.py`
  - `plot_common.py`
  - `jit_util_common.py`
  - `conftest.py`

### Usage

```bash
export PTO_LIB_PATH=${ASCEND_TOOLKIT_HOME}
cd examples/jit_cpp/fast_hadamard

# available generic entry points
python benchmark.py --list
python plot.py --list
python test.py --list

# benchmark and write CSV files
python benchmark.py --target standard --npu 0
# one-sided cube Hadamard over N in {64, 128, 256, 512}, timed against the
# vector kernel and, where N >= 256, the two-sided kernel
python benchmark.py --target onesided --npu 0
# two-sided cube Hadamard over N in 256..16384, timed against the vector kernel
python benchmark.py --target twosided --npu 0
python benchmark.py --target quantize_int8 --npu 0
python benchmark.py --target quantize_int4 --npu 0
# run both standalone quantize suites for later int8-vs-int4 plotting
python benchmark.py --target quantize_compare_int8_int4 --npu 0
# compares fused PTO, separate PTO, and torch+torch_npu unfused
python benchmark.py --target hadamard_quant_int8 --npu 0
python benchmark.py --target hadamard_quant_int4 --npu 0
# run both fused Hadamard+quant suites for later int8-vs-int4 plotting
python benchmark.py --target hadamard_quant_compare_int8_int4 --npu 0
# or run all benchmark suites with one command
python benchmark.py --target all --npu 0

# generate plots from CSV files
python plot.py --target standard
# one-sided-vs-vector plots, plus one-sided-vs-two-sided for the N they share
python plot.py --target onesided
# two-sided-vs-vector duration, bandwidth, and speedup plots
python plot.py --target twosided
python plot.py --target quantize_int8
python plot.py --target quantize_int4
# generate top-level int8-vs-int4 quantize comparison plots
python plot.py --target quantize_compare_int8_int4
python plot.py --target hadamard_quant_int8
python plot.py --target hadamard_quant_int4
# generate top-level int8-vs-int4 fused Hadamard+quant comparison plots
python plot.py --target hadamard_quant_compare_int8_int4
# or run all plot suites with one command
python plot.py --target all

# correctness tests
python test.py --target standard -v --npu "npu:0"
python test.py --target onesided -v --npu "npu:0"
python test.py --target twosided -v --npu "npu:0"
# the size-dispatching entry point: routing, plus correctness through it
python test.py --target auto -v --npu "npu:0"
python test.py --target quantize_int8 -v --npu "npu:0"
python test.py --target quantize_int4 -v --npu "npu:0"
python test.py --target hadamard_quant_int8 -v --npu "npu:0"
python test.py --target hadamard_quant_int4 -v --npu "npu:0"
# or run all test suites with one command
python test.py --target all -v --npu "npu:0"
```

### Output

Benchmark scripts save CSV files into the subdirectories that own the real
implementations:

- `standard/outputs/csv/fht_pto_bd*.csv` from `benchmark.py --target standard`
- `cube/outputs/csv/fht_onesided_compare_bd*.csv` from `benchmark.py --target onesided`
- `cube/outputs/csv/fht_twosided_compare_bd*.csv` from `benchmark.py --target twosided`
- `fuse_int8_quant/outputs/csv/quantize_compare_bd*.csv` from `benchmark.py --target quantize_int8`
- `fuse_int8_quant/outputs/csv/fht_quant_compare_bd*.csv` from `benchmark.py --target hadamard_quant_int8`
- `fuse_int4_quant/outputs/csv/quantize_compare_bd*.csv` from `benchmark.py --target quantize_int4`
- `fuse_int4_quant/outputs/csv/fht_quant_compare_bd*.csv` from `benchmark.py --target hadamard_quant_int4`

Plot scripts read CSV files from the same subdirectories and save figures to:

- `standard/outputs/plots/`
- `cube/outputs/plots/`
- `fuse_int8_quant/outputs/plots/`
- `fuse_int4_quant/outputs/plots/`
- `outputs/plots/` for the top-level int8-vs-int4 comparison plot targets

JIT-compiled shared libraries are saved to:

- `standard/outputs/so/`
- `cube/outputs/so/`, including the kernels the `auto` entry point compiles
  on demand
- `fuse_int8_quant/outputs/so/`
- `fuse_int4_quant/outputs/so/`

In `benchmark.py --target hadamard_quant_int8`, the unfused reference is
`hadamard_torch_stagewise(...) + torch_npu.npu_quantize(...)`, reported as
`torch_unfused_*` in the CSV and plots.

In the int4 targets, outputs use packed-byte storage with shape `[batch, N // 2]`.
The unfused int4 reference path is `hadamard_torch_stagewise(...)` followed by a
torch packed-int4 reference, also reported as `torch_unfused_*` in the fused CSV.

Both cube kernels are out of place (`fn(x, y)`), unlike the in-place vector
kernel (`fn(x)`), and both take `N` 16-aligned so the cube can tile it. The
two-sided kernel additionally needs `N = R * C` with both factors 16-aligned,
so its hidden-dimension grid starts at 256 instead of 128.

In `benchmark.py --target onesided` and `--target twosided`, the same shapes
are also timed with the `standard/` vector kernel, reported as `vector_*` in
the CSV and plots alongside an `*_speedup_vs_vector` column. The one-sided
target carries a third arm, `twosided_*`, which is blank for the `N` the
two-sided kernel cannot run; the one-sided-vs-two-sided plots cover only the
rows that have it.

Both cube kernels pick `block_dim` per launch from the shape, so the `bd` in a
CSV filename is the cap (`cube_core_num`) and the `onesided_block_dim` /
`twosided_block_dim` column records what each shape actually ran at. Passing
`block_dim` explicitly overrides the schedule; it can change only the time,
never the result.
