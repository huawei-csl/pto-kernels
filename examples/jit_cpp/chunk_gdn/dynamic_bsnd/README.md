# Dynamic BSND PTO Kernels for Chunkwise GatedDeltaNet (GDN)

PTO-ISA C++ kernels for the forward pass of chunk-wise GatedDeltaNet,
operating directly on the `[batch, seq, head, hidden]` (BSND) layout
with runtime-dynamic `batch` and `seq` dimensions and variable-length
sequence support via `cu_seqlens`.

## Kernels

| Kernel | File | Description |
|--------|------|-------------|
| `chunk_cumsum` | `chunk_cumsum_kernel.cpp` | Chunk-local prefix sum of gate values |
| `scaled_dot_kkt` | `scaled_dot_kkt_kernel.cpp` | Gated `K @ K^T` with masking and beta |
| `wy_fast` | `wy_fast_kernel.cpp` | WY-fast recompute: `w = A @ (k·β·exp(g))`, `u = A @ (v·β)` |
| `chunk_h` | `chunk_h_kernel.cpp` | Sequential state recurrence |
| `chunk_o` | `chunk_o_kernel.cpp` | Final output from inter/intra-chunk attention |

Template parameters (`-D` macros at compile time): `GDN_H` (heads),
`GDN_D` (hidden size), `GDN_C` (chunk size, default 128).

Runtime arguments: `batch_size`, `seq_len`, `cu_seqlens`.

## Quick start

```bash
# From the dynamic_bsnd directory:
cd /workdir/pto-kernels-fork/examples/jit_cpp/chunk_gdn/dynamic_bsnd

# Verify numerical correctness
GDN_NPU_DEVICE=npu:7 python3 verify_dynamic_bsnd.py

# Benchmark (N_seq=16, L_seg=16384, H=16, D=128, C=128)
GDN_NPU_DEVICE=npu:7 python3 bench_dynamic_bsnd.py

# Compare with references
GDN_NPU_DEVICE=npu:6 python3 ../triton_baseline/bench_triton_gdn.py
GDN_NPU_DEVICE=npu:5 python3 ../static_baseline/bench_static_gdn.py
```

## Benchmark results

Shape: `(N_seq=16, L_seg=16384, H=16, DK=DV=128, C=128)`, packed varlen
BSND with `T=262144`. The table below reports the median of 3 full
benchmark runs on `npu:7`.

| Kernel | Latency (ms) | #ops (approx) | TFLOPS |
| :-- | --: | --: | --: |
| chunk_cumsum | 0.18 | 4.19e+06 | 0.0233 |
| chunk_scaled_dot_kkt | 4.67 | 6.87e+10 | 14.71 |
| wy_fast | 6.92 | 1.37e+11 | 19.80 |
| chunk_h | 9.68 | 2.75e+11 | 28.41 |
| chunk_o | 11.13 | 3.44e+11 | 30.91 |
| total | 32.57 | 8.25e+11 | 25.33 |

## Design notes

- **BSND layout**: All tensors use `[B=1, T, H, D]` contiguous layout.
  Row stride for QKV tiles is `H * D`; for A tiles `H * C`; for g/beta
  tiles `H`.
- **Variable-length sequences**: `cu_seqlens` (int32) provides cumulative
  sequence boundaries. When non-null, `batch_size` is the number of
  sequences and `seq_len` is ignored.
- **Contiguous per-head G/Beta staging**: The public torch API still
  accepts `g_sum` / `beta` in `[1, T, H]`. Runtime helpers materialize
  contiguous `[H, T]` workspaces so the hot kernels can DMA per-head
  slices directly instead of extracting columns with scalar
  `GetValue`/`SetValue` loops.
- **Vectorized cumsum across heads**: `chunk_cumsum` now keeps a
  `1 x H` running row accumulator in UB and performs row-by-row Vec adds,
  removing the old per-head scalar accumulation loop.
- **Direct PTO source style**: The kernel sources now spell out
  `pto::GlobalTensor`, `pto::Tile`, dynamic-valid tiles, and explicit
  `TLOAD` / `TSTORE` / `TFILLPAD` blocks in place. The old shared
  `include/common.h` wrapper layer has been removed, so the math comments
  in each kernel map directly to PTO primitives or to the tiny local
  K-sliced GEMM helper that stays beside its call sites.
- **Benchmark timing**: `bench_dynamic_bsnd.py` precomputes the
  contiguous `g_t` / `beta_t` workspaces once before the timed kernel
  loop so the reported numbers reflect kernel execution rather than
  one-time layout preparation.
- **Compiler sweep note**: Additional compile-flag experiments
  (`-O3`, `-O3 -funroll-loops`) were tested after the kernel changes, but
  they were not a stable win over the current default build across
  repeated full benchmarks, so the default compile path was left
  unchanged.
- **Grid-stride loop**: Each physical core iterates over multiple logical
  work items to handle dynamic workloads.
- **Per-core workspace**: Intermediate buffers (e.g., K@K^T, state matrices)
  are indexed by `cid` (physical core ID) and reused across iterations.
- **Two-stage cube-vec pipeline**: `scaled_dot_kkt` uses double-buffered
  workspace slots with cross-core synchronization flags to overlap Cube
  matmul (chunk i+1) with Vec gating (chunk i).
- **Vectorized gating**: `chunk_o` uses SIMD operations (`TROWEXPAND`,
  `TCOLEXPAND`, `TSUB`, `TMINS`, `TEXP`, `TMUL`) for gating coefficient
  construction and QS row-scaling, replacing scalar `GetValue`/`SetValue`
  loops.
- **safe_exp via clamp**: `scaled_dot_kkt` and `chunk_o` clamp
  `g_row - g_col` to `min(x, 0)` before `exp()` to prevent IEEE 754
  `Inf * 0 = NaN`.
- **solve_tril omitted**: Consistent with the benchmark configuration.
