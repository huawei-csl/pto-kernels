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
- builds the triangular causal mask once in PyTorch and passes it into the kernel
- checks numerical correctness with `torch.testing.assert_close`
- reports effective TFLOP/s and GiB/s on larger shapes

Run correctness:

```bash
python run_linear_attention.py
```

Run the default benchmark table:

```bash
python benchmark_linear_attention.py --warmup 2 --repeats 5
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

The main performance ideas now present in `linear_attention.cpp` are:
- precomputed causal mask passed from PyTorch and applied with vector tile ops
- shared L0C reuse so larger tiles fit without changing the math
- cube-side `K=128 -> 2 x 64` L0 ping-pong inside the GEMM helper
- 2-slot cube/vector workspace pipeline for chunk overlap
- in-place mask application on `acc_ub` to reduce UB pressure
- two L1 hidden-state buffers so the next prefix-state tile can be prefetched early

## Step-By-Step Tutorial

`optimize_step_by_step/` mirrors the optimization path as runnable local examples:

1. `01_naive_static_shape`
2. `02_naive_dynamic_shape`
3. `03_cached_mask`
4. `04_chunk128`
5. `05_l0_double_buffer`
6. `06_two_slot_cv_pipeline`
7. `07_l1_prefetching`

The tutorial keeps each kernel source self-contained, but now shares common Python compile / test / benchmark helpers through `optimize_step_by_step/common/`.

Start there if you want to understand how the kernel evolved, or if you want a smaller teaching version before reading the main optimized kernel.

## Measured Results

Command used:

```bash
python benchmark_linear_attention.py --warmup 2 --repeats 5
```

Current measured table on this machine:

| Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | ---: | ---: | ---: |
| `(32, 20, 2048, 128, 128)` | `2.282` | `75.28` | `547.76` |
| `(24, 20, 4096, 128, 128)` | `3.362` | `76.64` | `557.64` |
| `(12, 20, 8192, 128, 128)` | `3.327` | `77.45` | `563.54` |
| `(24, 20, 6144, 128, 128)` | `5.063` | `76.35` | `555.55` |

Feature-extension quick table on this machine
using `python benchmark_linear_attention.py --quick --repeats 10 --warmup 3`
and the corresponding `--seq-first`, `--use-g`, and `--varlen-uniform` modes:

| Shape `(B,H,L,D,C)` | PTO path | Median ms | TFLOP/s | GiB/s |
| --- | --- | ---: | ---: | ---: |
| `(8, 20, 1024, 128, 128)` | `legacy_head_first` | `0.416` | `51.62` | `375.66` |
| `(8, 20, 1024, 128, 128)` | `seq_first` | `0.580` | `37.00` | `269.27` |
| `(8, 20, 1024, 128, 128)` | `seq_first_gated` | `0.576` | `37.30` | `271.41` |
| `(8, 20, 1024, 128, 128)` | `seq_first_varlen_uniform` | `0.591` | `36.36` | `264.64` |
| `(16, 20, 1024, 128, 128)` | `legacy_head_first` | `0.710` | `60.49` | `440.13` |
| `(16, 20, 1024, 128, 128)` | `seq_first` | `0.964` | `44.55` | `324.16` |
| `(16, 20, 1024, 128, 128)` | `seq_first_gated` | `0.986` | `43.57` | `317.06` |
| `(16, 20, 1024, 128, 128)` | `seq_first_varlen_uniform` | `0.972` | `44.18` | `321.48` |

Notes:
- device-local results will vary
- bandwidth here excludes workspace traffic, so it reflects external tensor movement plus the mask tensor
- the same kernel family at `C=64, D=128` is roughly in the `28-31 TFLOP/s` range on large shapes
- the best measured default-table point here is `77.45 TFLOP/s` at `(12, 20, 8192, 128, 128)`
- the feature-extension rows above benchmark the new seq-first / gated / varlen PTO path with precomputed chunk states `h`, so they are best compared against each other and against the quick `legacy_head_first` rows, not against the fused large-shape default table

## Reading Order

If you are new to this directory:

1. Read `optimize_step_by_step/README.md`
2. Run `01` and `02`, including their `numpy_sim.py`
3. Read the current `linear_attention.cpp`
4. Use `optimization_lession.md` as the checklist for future optimization work