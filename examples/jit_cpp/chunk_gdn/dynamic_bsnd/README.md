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
# From the chunk_gdn directory:
cd /workdir/pto-kernels/examples/jit_cpp/chunk_gdn

# Verify numerical correctness
python3 dynamic_bsnd/verify_dynamic_bsnd.py

# Benchmark (N_seq=16, L_seg=16384, H=16, D=128, C=128)
python3 dynamic_bsnd/bench_dynamic_bsnd.py
```

## Benchmark results

Shape: `(N_seq=16, L_seg=16384, H=16, DK=DV=128, C=128)`, packed varlen
BSND with `T=262144`.

| Kernel | Latency (ms) | #ops (approx) | TFLOPS |
| :-- | --: | --: | --: |
| chunk_cumsum | 2.03 | 4.19e+06 | 0.0021 |
| chunk_scaled_dot_kkt | 15.52 | 6.87e+10 | 4.4271 |
| wy_fast | 16.78 | 1.37e+11 | 8.1920 |
| chunk_h | 14.18 | 2.75e+11 | 19.3812 |
| chunk_o | 26.20 | 3.44e+11 | 13.1162 |
| total | 74.71 | 8.25e+11 | 11.0375 |

## Design notes

- **BSND layout**: All tensors use `[B=1, T, H, D]` contiguous layout.
  Row stride for QKV tiles is `H * D`; for A tiles `H * C`; for g/beta
  tiles `H`.
- **Variable-length sequences**: `cu_seqlens` (int32) provides cumulative
  sequence boundaries. When non-null, `batch_size` is the number of
  sequences and `seq_len` is ignored.
- **In-kernel G/beta column extraction**: `g_sum` and `beta` are accepted
  in the original `[1, T, H]` layout (same API as Triton kernels). Each
  kernel loads a `[C, H]` chunk via DMA, then extracts the per-head
  column with scalar `GetValue`/`SetValue` loops (matching `chunk_h`'s
  pattern). This avoids Python-side pre-transpose and keeps PTO kernels
  as drop-in replacements for Triton.
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
