# JIT C++ Matmul (A x B^T, fp16)

This example integrates the PTO-ISA CubeCore matmul kernel via JIT compile (`bisheng`) + `ctypes`.

## Files

- `matmul_custom_pto.cpp`: custom PTO-ISA kernel source (`call_kernel(blockDim, stream, x, y, z, M, N, K, swizzle_direction, swizzle_count)`).
- `matmul_original_pto.cpp`: upstream-style PTO kernel template compiled per shape via macros.
- `jit_util_matmul.py`: JIT compile + Python wrapper for PTO-ISA backend.
- `jit_util_original_pto.py`: JIT compile + Python wrapper for `original_pto`.
- `bench_kernels.py`: one-shot default benchmark + plot script for `torch`, `custom`, and `original`.
- `tune_swizzle.py`: sweep swizzle configs (`direction in {0,1}`, `count in [0,12]`) over a range of `M` for fixed `N/K`.

Input convention:

- `a`: `[M, K]` fp16
- `b`: `[N, K]` fp16
- output `c`: `[M, N]`, computes `c = a @ b^T`

## Run

```bash
# for example, 910B2 cube cores nums
export PTO_MATMUL_MAX_BLOCK_DIM=24
export ORIG_PTO_MATMUL_BLOCK_DIM=24
# Default benchmark + plots in one run:
python bench_kernels.py

# Swizzle tuning for fixed N/K across a range of M:
python tune_swizzle.py --n 4096 --k 4096
```

Outputs:

- Default CSV: `outputs/csv/matmul_timing.csv`
- Default plots: `outputs/plots/{duration,flops,error}.png`
- Swizzle sweep CSVs: `outputs/csv/swizzle_sweep_n{N}_k{K}.csv` and `outputs/csv/swizzle_summary_n{N}_k{K}.csv`

## Notes
- Best performance for `matmul_custom_pto.cpp` if **M = 128k,  (k ∈ ℤ)**.
- Both wrappers contain padding/tail handling with aligned fast path + dynamic block-dim.
- `original_pto` backend compiles a shape-specialized kernel per padded `(M,N,K)` and emits float32 output.
- Tunables:
  - `PTO_MATMUL_MAX_BLOCK_DIM` (default `20`, PTO backend)
  - `ORIG_PTO_MATMUL_BLOCK_DIM` (default `20`, original_pto backend)
  - `PTO_MATMUL_SWIZZLE_DIRECTION` (default `1`, used by Python wrapper default)
  - `PTO_MATMUL_SWIZZLE_COUNT` (default `3`, used by Python wrapper default)
- Swizzle direction/count are runtime args now; no kernel source edits are needed.
