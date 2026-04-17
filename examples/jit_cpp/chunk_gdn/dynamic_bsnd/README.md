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

# Reproduce the full NPU verification sweep used during development
python3 dynamic_bsnd/verify_dynamic_bsnd.py --device npu:7

# Re-run the previously failing ragged-tail regression directly
python3 dynamic_bsnd/verify_dynamic_bsnd.py --device npu:7 --case 21 -v

# Benchmark (N_seq=16, L_seg=16384, H=16, D=128, C=128)
python3 dynamic_bsnd/bench_dynamic_bsnd.py
```

## Benchmark results

Shape: `(N_seq=16, L_seg=16384, H=16, DK=DV=128, C=128)`, packed varlen
BSND with `T=262144`.

| Kernel | PTO (ms) | Triton (ms) | Speedup | TFLOPS |
| :-- | --: | --: | --: | --: |
| chunk_cumsum | 0.37 | 1.00 | 2.7x | 0.012 |
| chunk_scaled_dot_kkt | 4.69 | 4.81 | 1.03x | 14.6 |
| wy_fast | 6.85 | 15.57 | 2.27x | 20.1 |
| chunk_h | 9.57 | 30.82 | 3.22x | 28.7 |
| chunk_o | 10.73 | 16.13 | 1.50x | 32.0 |
| **total** | **32.20** | **68.34** | **2.12x** | **25.6** |

## Design notes

- **BSND layout**: All tensors use `[B=1, T, H, D]` contiguous layout.
  Row stride for QKV tiles is `H * D`; for A tiles `H * C`; for g/beta
  tiles `H`.
- **Variable-length sequences**: `cu_seqlens` (int32) provides cumulative
  sequence boundaries. When non-null, `batch_size` is the number of
  sequences and `seq_len` is ignored.
- **Drop-in Triton replacement**: The Python wrapper functions (`run_*`)
  accept the same argument list and memory layouts as Triton kernels.
  G/beta are accepted as `[1, T, H]` and transposed internally to
  `[H, T]` for efficient contiguous DMA loads per-head. PTO kernels can
  be used as drop-in replacements in production inference.
- **Head-first G/beta layout**: `g_sum` and `beta` are transposed from
  `[1, T, H]` to `[H, T]` inside the Python `run_*` wrappers, enabling
  contiguous DMA loads per-head inside the C++ kernels. This eliminates
  costly scalar `GetValue`/`SetValue` extraction loops.
- **Vectorized cumsum**: `chunk_cumsum` uses SIMD row-wise TADD/TMOV
  operations to process all heads simultaneously per row, replacing
  per-head scalar loops.
- **Vectorized coefficient scaling**: `chunk_h` uses TROWEXPAND + TMUL
  to apply per-row decay coefficients to [HalfC, D] tiles, replacing
  scalar GetValue/TMULS loops.
- **DMA-Cube overlap**: `scaled_dot_kkt` issues G/beta DMA before
  waiting for the Cube GEMM, hiding DMA latency behind Cube compute.
- **Grid-stride loop**: Each physical core iterates over multiple logical
  work items to handle dynamic workloads.
- **Per-core workspace**: Intermediate buffers (e.g., K@K^T, state matrices)
  are indexed by `cid` (physical core ID) and reused across iterations.
- **Two-stage cube-vec pipeline**: `scaled_dot_kkt` uses double-buffered
  workspace slots with cross-core synchronization flags to overlap Cube
  matmul (chunk i+1) with Vec gating (chunk i).
- **Vectorized gating**: `chunk_o` uses SIMD operations (`TROWEXPAND`,
  `TCOLEXPAND`, `TSUB`, `TMINS`, `TEXP`, `TMUL`) for gating coefficient
  construction and QS row-scaling.
- **safe_exp via TMINS**: `scaled_dot_kkt` and `chunk_o` clamp
  `g_row - g_col` to `min(x, 0)` via `TMINS(coeff, coeff, 0.0f)` before
  `TEXP` to prevent IEEE 754 `Inf * 0 = NaN`.
- **solve_tril omitted**: Consistent with the benchmark configuration.
