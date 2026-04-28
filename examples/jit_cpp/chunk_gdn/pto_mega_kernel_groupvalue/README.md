# GDN mega-kernel (group-value / GQA)

Single-launch NPU mega-kernel for the gated delta chunk pipeline when **queries and keys share `Hg` heads** while **values, gates `β`, and cumulative gates use `H` value heads** (`H ≥ Hg`, `H % Hg == 0`). Implementation mirrors `pto_mega_kernel`, but stages `scaled_dot_kkt`, `wy_fast`, `chunk_h`, and `chunk_o` are included from `dynamic_bsnd_groupvalue`; `chunk_cumsum` stays in `dynamic_bsnd`; triangular inverse is still `csrc/kernel/kernel_tri_inv_rec_unroll.cpp`.

## Pipeline

| # | Stage | Source | Notes |
|---|-------|--------|--------|
| 1 | cumsum | `dynamic_bsnd/chunk_cumsum_kernel.cpp` | `H` gates |
| 2 | transpose | in megakernel | `g_sum`, `beta` `[T,H]` → `[H,T]` |
| 3 | kkt | `dynamic_bsnd_groupvalue/scaled_dot_kkt_kernel.cpp` | `K` has shape `Hg` |
| 4 | solve_tril | `kernel_tri_inv_rec_unroll.cpp` | matrices indexed per value head (`H`) |
| 5 | wy_fast | `dynamic_bsnd_groupvalue/wy_fast_kernel.cpp` | |
| 6 | chunk_h | `dynamic_bsnd_groupvalue/chunk_h_kernel.cpp` | |
| 7 | chunk_o | `dynamic_bsnd_groupvalue/chunk_o_kernel.cpp` | `Q,K` span `Hg` |

Stages are merged with cross-core barriers (`SyncAllImpl`) identical to `pto_mega_kernel`.

## Files

| File | Purpose |
|------|---------|
| `mega_kernel.cpp` | Fused kernel (defines `GDN_H` and `GDN_HG`; includes groupvalue kernels) |
| `mega_kernel_compile.py` | `bisheng` build, ctypes loader, `run_mega_kernel(..., key_heads=Hg)` |
| `verify_mega_kernel_groupvalue.py` | Per-stage PTO + CPU fp32 refs; **`--configs`** default **`16×16,32×16,48×16,64×16`** (see below) |
| `bench_mega_kernel_groupvalue.py` | Wall-clock mega vs sequential PTO chain |

## Quick start

```bash
cd examples/jit_cpp/chunk_gdn/pto_mega_kernel_groupvalue

# Accuracy: 13 uniform/varlen profiles × `--configs` (default: four H×Hg pairs)
python verify_mega_kernel_groupvalue.py --device npu:4

# Subset only
python verify_mega_kernel_groupvalue.py --device npu:4 --configs 32x16

# Benchmark (default: H in 16,32,48,64 with Hg=16)
python bench_mega_kernel_groupvalue.py --device npu:4

# Typical env overrides
export PTO_LIB_PATH=/path/to/pto-isa    # if includes not under ASCEND_TOOLKIT_HOME
export GDN_NPU_DEVICE=npu:7
```

The first `(H, Hg)` build compiles with `bisheng` (~25 s typical); results are cached in `compiled_lib/mega_kernel_groupvalue_H{H}_Hg{Hg}_D128_C128.so`.

## Verification coverage (`Hg = 16`)

The default **`--configs 16x16,32x16,48x16,64x16`** exercises **four** value-head counts **H ∈ {16, 32, 48, 64}**, all **GQA-aligned** with **`Hg = 16`**. **`verify_mega_kernel_groupvalue.py`** runs the same **13** shape profiles against **per-stage PTO** (`run_pto_e2e` from **`verify_pto_triton_e2e_groupvalue`**) **and** a CPU fp32 reference chain (**`ref_*_group`** + **`ref_solve_tril`**).

**Latest run:** **2026-04-28**, device **`npu:4`**, **`52 / 52`** sub-cases passed (`4` configs × **`13`** shapes):

```bash
python verify_mega_kernel_groupvalue.py --device npu:4 --configs 16x16,32x16,48x16,64x16
```

## Benchmark: mega vs per-stage PTO

Measured **2026-04-28**, same device as verification, **`block_dim = 24`**, **D = 128**, **C = 128**. **`warmup = 5`**, **`iters = 20`**, wall time via `time.perf_counter` around the fused launch vs sequential **`run_pto_e2e`**.

```bash
python bench_mega_kernel_groupvalue.py --device npu:4 --configs 16x16,32x16,48x16,64x16
```

### H = 16, Hg = 16

| Scenario | Mega (ms) | Per-stage (ms) | Speedup |
|----------|-----------|----------------|---------|
| T = 128 | 0.81 | 1.78 | 2.18x |
| T = 256 | 0.82 | 1.77 | 2.16x |
| T = 512 | 0.83 | 1.81 | 2.18x |
| T = 1024 | 0.86 | 1.86 | 2.16x |
| T = 2048 | 1.02 | 1.90 | 1.86x |
| T = 4096 | 1.47 | 2.13 | 1.45x |
| T = 8192 | 2.29 | 2.90 | 1.27x |
| T = 16384 | 4.17 | 4.83 | 1.16x |
| T = 32768 | 7.90 | 8.53 | 1.08x |
| T = 65536 | 15.24 | 16.01 | 1.05x |
| varlen [256, 256] | 0.82 | 1.80 | 2.20x |
| varlen long mix (T = 2048) | 0.99 | 1.93 | 1.94x |
| 16 × 16384 (T = 262144) | 54.44 | 56.70 | 1.04x |

### H = 32, Hg = 16

| Scenario | Mega (ms) | Per-stage (ms) | Speedup |
|----------|-----------|----------------|---------|
| T = 128 | 0.79 | 1.74 | 2.22x |
| T = 256 | 0.76 | 1.70 | 2.24x |
| T = 512 | 0.81 | 1.76 | 2.16x |
| T = 1024 | 0.98 | 1.85 | 1.90x |
| T = 2048 | 1.40 | 2.08 | 1.49x |
| T = 4096 | 2.23 | 2.83 | 1.27x |
| T = 8192 | 4.01 | 4.66 | 1.16x |
| T = 16384 | 7.66 | 8.32 | 1.09x |
| T = 32768 | 15.01 | 15.88 | 1.06x |
| T = 65536 | 29.80 | 31.17 | 1.05x |
| varlen [256, 256] | 0.81 | 1.81 | 2.23x |
| varlen long mix (T = 2048) | 1.34 | 2.11 | 1.57x |
| 16 × 16384 (T = 262144) | 108.40 | 112.98 | 1.04x |

### H = 48, Hg = 16

| Scenario | Mega (ms) | Per-stage (ms) | Speedup |
|----------|-----------|----------------|---------|
| T = 128 | 0.81 | 1.77 | 2.19x |
| T = 256 | 0.80 | 1.79 | 2.23x |
| T = 512 | 0.89 | 1.85 | 2.08x |
| T = 1024 | 1.13 | 1.99 | 1.77x |
| T = 2048 | 1.72 | 2.34 | 1.36x |
| T = 4096 | 2.82 | 3.51 | 1.24x |
| T = 8192 | 5.41 | 6.01 | 1.11x |
| T = 16384 | 10.46 | 11.25 | 1.08x |
| T = 32768 | 20.61 | 21.76 | 1.06x |
| T = 65536 | 40.98 | 42.93 | 1.05x |
| varlen [256, 256] | 0.90 | 1.97 | 2.20x |
| varlen long mix (T = 2048) | 1.75 | 2.48 | 1.42x |
| 16 × 16384 (T = 262144) | 163.61 | 170.00 | 1.04x |

### H = 64, Hg = 16

| Scenario | Mega (ms) | Per-stage (ms) | Speedup |
|----------|-----------|----------------|---------|
| T = 128 | 0.79 | 1.78 | 2.26x |
| T = 256 | 0.82 | 1.83 | 2.22x |
| T = 512 | 0.99 | 1.92 | 1.95x |
| T = 1024 | 1.36 | 2.11 | 1.55x |
| T = 2048 | 2.12 | 2.75 | 1.29x |
| T = 4096 | 3.75 | 4.43 | 1.18x |
| T = 8192 | 7.24 | 8.06 | 1.11x |
| T = 16384 | 14.31 | 15.27 | 1.07x |
| T = 32768 | 27.78 | 29.25 | 1.05x |
| T = 65536 | 54.65 | 57.12 | 1.05x |
| varlen [256, 256] | 0.98 | 1.90 | 1.94x |
| varlen long mix (T = 2048) | 2.10 | 2.70 | 1.29x |
| 16 × 16384 (T = 262144) | 212.22 | 221.35 | 1.04x |

At fixed **Hg**, increasing **H** scales work in most stages; mega-kernel stays ahead of the sequential PTO pipeline on every case above, with speedup approaching **1×** only at the longest **T** where raw compute dominates timing.

## Implementation note: `dynamic_kernel_libs` on `PYTHONPATH`

`dynamic_bsnd` and `dynamic_bsnd_groupvalue` both install a sibling module named `dynamic_kernel_libs`. Imports that need `verify_dynamic_bsnd` (cumsum JIT) **must resolve `dynamic_bsnd` ahead of `dynamic_bsnd_groupvalue`** on `sys.path` (see insertion order at the top of the verify/bench scripts).
