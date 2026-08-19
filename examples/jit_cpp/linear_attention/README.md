# PTO-ISA Linear Attention

This directory contains a self-contained PTO-ISA linear attention example and a local step-by-step optimization tutorial.

## What Is Here

- `linear_attention.cpp`: the current optimized kernel
- `run_linear_attention.py`: correctness sweep against a PyTorch reference
- `benchmark_linear_attention.py`: throughput and bandwidth benchmark
- `optimization_lession.md`: reusable optimization notes for future PTO-ISA kernels
- `optimize_step_by_step/`: a tutorial ladder from naive fixed-shape code to the current fast path

## Main Example

The main example:
- compiles `linear_attention.cpp` with `bisheng`
- loads the generated `.so` via `ctypes`
- runs the kernel from PyTorch on NPU
- supports both a cached causal-mask path and a fast on-the-fly `TTRI` mask path
- checks numerical correctness with `torch.testing.assert_close`
- reports effective TFLOP/s and GiB/s on larger shapes

Run correctness:

```bash
python run_linear_attention.py
```

Run the default benchmark table:

```bash
python benchmark_linear_attention.py --warmup 2 --repeats 5 --mask-variant cached_mask
python benchmark_linear_attention.py --warmup 2 --repeats 5 --mask-variant fast_onthefly
```

Quick smoke benchmark:

```bash
python benchmark_linear_attention.py --quick --warmup 1 --repeats 3
```

Throughput hunt:

```bash
python benchmark_linear_attention.py --throughput-hunt --warmup 2 --repeats 5
```

## Current Kernel Shape

The current kernel keeps:
- dynamic `B` and `L`
- compile-time `H`, `D`, and `C`
- fixed `block_dim = num_cores`
- an explicit in-kernel loop over logical work items

The current fast path is `C=128, D=128`.

The directory now contains two PTO execution styles:
- legacy fused `head_first` `(B, H, T, D)` for the highest throughput reference path
- native `seq_first` `(B, T, H, D)` including gated and packed-varlen support without transpose or Python padding

The main performance ideas now present in `linear_attention.cpp` are:
- precomputed causal mask passed from PyTorch and applied with vector tile ops
- an additional fast on-the-fly mask variant that builds the same triangular tile in UB with `TTRI`
- shared L0C reuse so larger tiles fit without changing the math
- cube-side `K=128 -> 2 x 64` L0 ping-pong inside the GEMM helper
- 2-slot cube/vector workspace pipeline for chunk overlap
- in-place mask application on `acc_ub` to reduce UB pressure
- two L1 hidden-state buffers so the next prefix-state tile can be prefetched early
- static strided full-chunk PTO loads/stores for native `seq_first` inputs, with dynamic `TLOAD`/`TSTORE` reserved for only true varlen tail chunks

## Step-By-Step Tutorial

`optimize_step_by_step/` mirrors the optimization path as runnable local examples:

1. `01_naive_static_shape`
2. `02_naive_dynamic_shape`
3. `03_cached_mask`
4. `03a_fast_mask_construct`
5. `04_chunk128`
6. `05_l0_double_buffer`
7. `06_two_slot_cv_pipeline`
8. `07_l1_prefetching`

The tutorial keeps each kernel source self-contained, but now shares common Python compile / test / benchmark helpers through `optimize_step_by_step/common/`.

Start there if you want to understand how the kernel evolved, or if you want a smaller teaching version before reading the main optimized kernel.

## Measured Results

Command used:

```bash
python benchmark_linear_attention.py --warmup 2 --repeats 5 --mask-variant cached_mask
python benchmark_linear_attention.py --warmup 2 --repeats 5 --mask-variant fast_onthefly
```

Current measured default-shape comparison on this machine:

| Shape `(B,H,L,D,C)` | Cached ms | Cached TFLOP/s | Fast on-the-fly ms | Fast on-the-fly TFLOP/s | Faster variant |
| --- | ---: | ---: | ---: | ---: | --- |
| `(32, 20, 2048, 128, 128)` | `2.332` | `73.66` | `2.326` | `73.85` | `fast_onthefly` |
| `(24, 20, 4096, 128, 128)` | `3.442` | `74.87` | `3.379` | `76.26` | `fast_onthefly` |
| `(12, 20, 8192, 128, 128)` | `3.373` | `76.39` | `3.372` | `76.42` | `fast_onthefly` |
| `(24, 20, 6144, 128, 128)` | `4.985` | `77.55` | `5.017` | `77.05` | `cached_mask` |

Best measured points from those two runs:
- `cached_mask`: `77.55 TFLOP/s` / `564.23 GiB/s` at `(24, 20, 6144, 128, 128)`
- `fast_onthefly`: `77.05 TFLOP/s` / `560.62 GiB/s` at `(24, 20, 6144, 128, 128)`

Feature-extension quick table on this machine
using `python benchmark_linear_attention.py --quick --repeats 10 --warmup 3`
and the corresponding `--seq-first`, `--use-g`, and `--varlen-uniform` modes:

| Shape `(B,H,L,D,C)` | PTO path | Median ms | TFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: |
| `(8, 20, 1024, 128, 128)` | `legacy_head_first` | `0.416` | `51.62` | `375.66` |
| `(8, 20, 1024, 128, 128)` | `seq_first` | `0.549` | `39.10` | `284.54` |
| `(8, 20, 1024, 128, 128)` | `seq_first_gated` | `0.535` | `40.11` | `291.90` |
| `(8, 20, 1024, 128, 128)` | `seq_first_varlen_uniform` | `0.529` | `40.61` | `295.50` |
| `(16, 20, 1024, 128, 128)` | `legacy_head_first` | `0.710` | `60.49` | `440.13` |
| `(16, 20, 1024, 128, 128)` | `seq_first` | `0.880` | `48.80` | `355.07` |
| `(16, 20, 1024, 128, 128)` | `seq_first_gated` | `0.872` | `49.24` | `358.30` |
| `(16, 20, 1024, 128, 128)` | `seq_first_varlen_uniform` | `0.872` | `49.27` | `358.50` |

Native `seq_first` larger-shape table on this machine
using `python benchmark_linear_attention.py --throughput-hunt --repeats 5 --warmup 2 --seq-first`:

| Shape `(B,H,L,D,C)` | PTO path | Median ms | TFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: |
| `(24, 20, 2048, 128, 128)` | `seq_first` | `2.154` | `59.81` | `435.15` |
| `(48, 20, 1024, 128, 128)` | `seq_first` | `2.160` | `59.66` | `434.09` |
| `(12, 20, 8192, 128, 128)` | `seq_first` | `4.085` | `63.08` | `458.99` |
| `(24, 20, 1536, 128, 128)` | `seq_first` | `1.661` | `58.19` | `423.39` |

Notes:
- device-local results will vary
- bandwidth here excludes workspace traffic; the cached-mask rows include mask tensor traffic while the fast on-the-fly rows do not
- the same kernel family at `C=64, D=128` is roughly in the `28-31 TFLOP/s` range on large shapes, both for cached-mask and fast on-the-fly masking
- on this machine, the fast on-the-fly `TTRI` path is effectively tied with cached-mask at `C=128`, winning 3 of the 4 default benchmark shapes by a small margin
- the feature-extension rows above benchmark the native `seq_first` / gated / varlen PTO path with precomputed chunk states `h`; after optimization they now reach roughly `49 TFLOP/s` on the quick shapes and `~63 TFLOP/s` on larger seq-first workloads
- the native `seq_first` path still trails the legacy fused `head_first` reference on the smallest quick shapes, but it no longer needs any transpose or Python-side padding and now lands in the same broad throughput class on larger inputs

## Reading Order

If you are new to this directory:

1. Read `optimize_step_by_step/README.md`
2. Run `01` and `02`, including their `numpy_sim.py`
3. Read the current `linear_attention.cpp`
4. Use `optimization_lession.md` as the checklist for future optimization work