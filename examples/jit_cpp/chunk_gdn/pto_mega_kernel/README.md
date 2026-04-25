# GDN Mega-Kernel

A single-launch NPU kernel that fuses all 7 stages of the GDN (Gated Delta
Network) chunk pipeline into one `<<<>>>` invocation, eliminating inter-kernel
launch overhead and PyTorch eager calls for transpose / dtype-cast operations.

## Pipeline stages

All stages execute sequentially inside one kernel, separated by `SyncAllImpl`
cross-core barriers that enforce GM write-read ordering.

| # | Stage | Pipes | Description |
|---|-------|-------|-------------|
| 1 | cumsum | Vec | Log-gate cumulative sum: `g` → `g_sum` |
| 2 | transpose | Vec | `g_sum [T,H]→[H,T]`, `beta [T,H]→[H,T]` via `TTRANS` |
| 3 | kkt | Cube+Vec | Scaled-dot KKT: `K, beta_t, g_t, Msk` → `A` |
| 4 | solve_tril | Cube | Triangular inverse: `A` → `A_inv` (fp16 via FIX pipe F322F16) |
| 5 | wy_fast | Vec+Cube | WY factorisation: `K, V, beta_t, g_t, A_inv` → `W, U` |
| 6 | chunk_h | Cube+Vec | Chunk state update: `K, W, U, g_t` → `S, V_new, FS` |
| 7 | chunk_o | Cube+Vec | Chunk output: `Q, K, V_new, S, g_t, Msk` → `O` |

## Files

| File | Purpose |
|------|---------|
| `mega_kernel.cpp` | Fused C++ kernel: sync helpers, in-kernel transpose, all 7 stages |
| `mega_kernel_compile.py` | JIT compilation (`bisheng`), `ctypes` loader, `run_mega_kernel()` API |
| `verify_mega_kernel.py` | Numerical verification against per-stage PTO and CPU fp32 reference |
| `bench_mega_kernel.py` | Wall-clock benchmark: mega-kernel vs per-stage PTO pipeline |

## Quick start

```bash
cd examples/jit_cpp/chunk_gdn/pto_mega_kernel

# Verify accuracy (13 shape configs, uniform + variable-length)
python verify_mega_kernel.py --device npu:0

# Benchmark (8 shape configs, reports speedup vs per-stage PTO)
python bench_mega_kernel.py --device npu:0

# Use a different device
python verify_mega_kernel.py --device npu:4
python bench_mega_kernel.py  --device npu:4 --warmup 10 --iters 50
```

The first run compiles the kernel via `bisheng` (takes ~20 s); subsequent runs
with the same `(H, D, C)` parameters reuse the cached `.so`.

## Performance summary

Measured on Ascend C220, H=16, D=128, C=128, `block_dim=24`:

| Sequence length | Mega-kernel | Per-stage PTO | Speedup |
|-----------------|-------------|---------------|---------|
| T = 128 | 0.86 ms | 1.78 ms | 2.07x |
| T = 256 | 0.83 ms | 1.80 ms | 2.19x |
| T = 512 | 0.83 ms | 1.82 ms | 2.20x |
| T = 1024 | 0.86 ms | 1.88 ms | 2.19x |
| T = 2048 | 1.01 ms | 1.92 ms | 1.91x |
| T = 4096 | 1.43 ms | 2.14 ms | 1.50x |
| T = 8192 | 2.25 ms | 2.89 ms | 1.28x |
| T = 16384 | 4.09 ms | 4.77 ms | 1.17x |
| T = 32768 | 7.78 ms | 8.52 ms | 1.09x |
| T = 65536 | 15.64 ms | 16.27 ms | 1.04x |
| T = 131072 | 30.71 ms | 32.00 ms | 1.04x |
| varlen [256, 256] | 0.82 ms | 1.83 ms | 2.24x |
| varlen long mix (T=2048) | 1.01 ms | 1.96 ms | 1.93x |
| 16×16384 (T=262144) | 55.05 ms | 56.95 ms | 1.03x |

Speedup is largest at short sequences (about 2.2x at T=128) where kernel-launch
overhead dominates, and converges toward 1x for very long sequences where
compute time dwarfs launch cost. Even at T=262144 the mega-kernel is slightly
faster due to eliminating the Python-side transpose and cast operations.

## Implementation considerations

### Cross-core synchronisation

`pipe_barrier(PIPE_ALL)` only orders pipes within a single AI core. Between
stages that share data through GM workspace, a full cross-core barrier
(`SyncAllImpl<false>()`) is required. This uses FFTS flags 11–14 to coordinate
all Cube and Vec sub-cores across every AIC.

### FFTS flag draining

Some original kernels (e.g. `wy_fast`, `chunk_o`, `kkt`) leave residual FFTS
flag counts that are balanced internally under normal stand-alone execution but
accumulate when stages are chained. Idle cores (those with
`get_block_idx() >= num_matrices`) never send these flags, so unconditional
`wait_flag_dev()` calls would deadlock. The mega-kernel drains residual flags
conditionally:

```cpp
#if defined(__DAV_C220_VEC__)
    if (get_block_idx() < num_matrices) {
        pipe_barrier(PIPE_ALL);
        wait_flag_dev(3);
        wait_flag_dev(4);
    }
#endif
```

### In-kernel transpose

The per-stage pipeline performs `g_sum` and `beta` transposes in Python
(`tensor.t().contiguous()`). The mega-kernel replaces this with
`mega_transpose_TH_to_HT`, which loads `[BLOCK, H]` contiguously via MTE2,
transposes in UB via `TTRANS`, then stores each of the `H` rows back to the
`[H, T]` destination with 1-D `TSTORE` per row. The row-by-row store avoids a
known issue with 2-D strided `TSTORE` on fp32 data.

### Direct fp16 output from solve_tril

The triangular-inverse kernel (`kernel_tri_inv_rec_unroll.cpp`) accumulates in
fp32 on L0C and originally wrote fp32 to GM, requiring a separate Vec-side
fp32→fp16 cast. That cast suffered from an L1-coherence issue: the FIX pipe
writes to GM bypass the L1 data cache, so subsequent Vec MTE2 reads could hit
stale L1 entries.

The fix adds a `StoreT` template parameter to `TriInvRecUnrollKernel` (defaults
to `OutputT` for backward compatibility). Setting `StoreT = half` while keeping
`OutputT = float` makes the final `TSTORE` use the FIX pipe's built-in
`F322F16` quantisation mode to write fp16 directly, eliminating the separate
cast stage entirely.

### Workspace allocation

All intermediate tensors that were previously separate PyTorch allocations
(`g_sum`, `g_t`, `beta_t`, `A`, `A_inv`, `w`, `u`, `s`, `v_new`, `fs`) are
pre-allocated on the Python side and passed as GM pointers to the single kernel
launch. Per-stage scratch buffers (`kkt_ws`, `wy_ws_*`, `h_ws`, `o_ws_*`) are
sized by `block_dim` and also pre-allocated.
