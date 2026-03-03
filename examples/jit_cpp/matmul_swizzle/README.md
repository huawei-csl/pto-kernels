# JIT C++ Matmul (A x B^T, fp16)

This example integrates the PTO-ISA CubeCore matmul kernel via JIT compile (`bisheng`) + `ctypes`.

## Files

- `matmul_custom_pto.cpp`: custom PTO-ISA kernel source (`call_kernel(blockDim, stream, x, y, z, M, N, K)`).
- `matmul_original_pto.cpp`: upstream-style PTO kernel template compiled per shape via macros.
- `jit_util_matmul.py`: JIT compile + Python wrapper for PTO-ISA backend.
- `jit_util_original_pto.py`: JIT compile + Python wrapper for `original_pto`.
- `bench_kernels.py`: one-shot default benchmark + plot script for `torch`, `custom`, and `original`.

Input convention:

- `a`: `[M, K]` fp16
- `b`: `[N, K]` fp16
- output `c`: `[M, N]`, computes `c = a @ b^T`

## Run

```bash
# Default benchmark + plots in one run:
python bench_kernels.py
```

Outputs:

- Default CSV: `outputs/csv/matmul_timing.csv`
- Default plots: `outputs/plots/{duration,flops,error}.png`

## Notes
- Best performance for `matmul_custom_pto.cpp` if **M = 128k,  (k ∈ ℤ)**.
- Both wrappers contain padding/tail handling with aligned fast path + dynamic block-dim.
- `original_pto` backend compiles a shape-specialized kernel per padded `(M,N,K)` and emits float32 output.
- Tunables:
  - `PTO_MATMUL_MAX_BLOCK_DIM` (default `20`, PTO backend)
  - `ORIG_PTO_MATMUL_BLOCK_DIM` (default `20`, original_pto backend)
- Swizzle Count and Direction have to be adjusted in the .cpp kernel code manually
