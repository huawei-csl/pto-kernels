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

# Reproduce the strict per-stage sweep used during development
# (isolated subprocesses + shell timeout help catch rare cross-core deadlocks)
timeout 600s python3 dynamic_bsnd/verify_dynamic_bsnd.py --device npu:7 --isolate

# Re-run the previously failing ragged-tail regression directly
timeout 240s python3 dynamic_bsnd/verify_dynamic_bsnd.py --device npu:7 --isolate --case 21 -v

# End-to-end PTO vs Triton agreement check
timeout 420s python3 pto_e2e_measure/verify_pto_triton_e2e.py --device npu:7 --no-plots

# Benchmark (N_seq=16, L_seg=16384, H=16, D=128, C=128)
python3 dynamic_bsnd/bench_dynamic_bsnd.py
```

## Numerical verification (valid error)

The canonical checker is `verify_dynamic_bsnd.py`. Each pipeline stage is compared to a **PyTorch reference on CPU in float32**; NPU tensors are cast to float before the diff. Inputs use fp16 where the kernel does; references are written to match the same numerics the test expects (for example `chunk_o` uses `exp(min(Δg, 0))` gating consistent with this PTO path).

**Per tensor check** — a stage passes if **either** condition holds, and there is no hard failure (below).

1. **Strict elementwise band** (same shape as [`torch.testing.assert_close`](https://docs.pytorch.org/docs/main/testing.html#torch.testing.assert_close) defaults in spirit: tight absolute, modest relative on fp16/bf16-style work):
   - `|actual − expected| ≤ atol + rtol · |expected|` everywhere,
   - with **`rtol = 1e-2`**, **`atol = 1e-5`**.
   - Large fixed `atol` (for example `1e-2`) is intentionally **not** used: when activations are around `1e-2`, that would allow ~100% relative error and is not an acceptable gate.

2. **Global fallback** (when a few outliers break the strict band but the tensor is still correct overall):
   - Let `RMSE = sqrt(mean((actual − expected)²))` and `mean_abs_ref = mean(|expected|)`.
   - Require **`RMSE / mean_abs_ref ≤ 0.05`** (RMSE should be much smaller than typical magnitude; this ratio is on the order of one to two orders below the scale of the values in many regimes).
   - And **`R² ≥ 0.99`** versus the CPU reference, when the reference has enough variance to define R² meaningfully (`std(expected) ≥ 1e-12`).
   - **Degenerate references:** if `mean(|expected|) < 1e-9`, the fallback uses a small absolute RMSE cap (`RMSE < 5e-4`) instead of R². If the mean is nonzero but `std(expected) < 1e-12`, only the RMSE ratio bound applies (no R² gate).

**Hard failure:** if **`max |actual − expected| > 1.0`** for that stage, the check fails regardless of the above (likely kernel bug or serious corruption).

**Other checks:** selected tensors (`chunk_h` states, `chunk_o`) must be **finite** (`-inf` / `nan` fails). With `-v`, each line shows `rm/|ref|` (RMSE over mean |ref| when defined) and `[allclose]` vs `[stats]` to show which branch passed. With `--fig-dir`, optional per-stage scatter plots (reference on x, kernel on y) are written.

Re-run the same script several times on NPU if you see flakiness; asynchronous execution can make rare races show up as intermittent numerical or hang issues.

## Benchmark results

Shape: `(N_seq=16, L_seg=16384, H=16, DK=DV=128, C=128)`, packed varlen
BSND with `T=262144`.

| Kernel | PTO (ms) | Triton (ms) | Speedup | TFLOPS |
| :-- | --: | --: | --: | --: |
| chunk_cumsum | 0.34 | 1.02 | 3.00x | 0.012 |
| chunk_scaled_dot_kkt | 4.67 | 4.84 | 1.04x | 14.7 |
| solve_tril | 15.90 | — | — | 1.44 |
| wy_fast | 6.82 | 15.63 | 2.29x | 20.1 |
| chunk_h | 10.14 | 30.83 | 3.04x | 27.1 |
| chunk_o | 11.52 | 16.15 | 1.40x | 29.8 |
| **total (exclude solve_tril)** | **33.49** | **68.47** | **2.04x** | **24.6** |

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
- **solve_tril**: Timed separately for PTO only (no Triton equivalent in this split). The **total_summed** row sums the five kernels that appear in both columns so PTO and Triton totals are comparable.
