# naive_separate — Two-stage baseline (no Cube↔Vec pipeline)

Computes `C = A @ B + D` (matmul_add_c2v) and `C = (A + B) @ D` (add_matmul_v2c)
in two **sequential** stages within a **single kernel launch**.  No round-level
overlap between the Cube (GEMM) stage and the Vec (element-wise add) stage.

## Purpose

Provides a slower baseline to demonstrate the benefit of fine-grained pipelining
in the `raw_flag`, `pushpop`, and `gm_pipe` variants.

## Algorithm

**matmul_add_c2v** (`C = A @ B + D`):
1. **Stage 1 — Cube (all rounds)**: compute `A @ B` for every round, write
   results to `workspace[batch, TILE_SIZE]`.  After all rounds, signal Vec
   with one `FLAG_C2V` broadcast.
2. **Stage 2 — Vec (all rounds)**: wait for `FLAG_C2V`, then for each round
   load the GEMM result from workspace, add `D`, write `C`.

**add_matmul_v2c** (`C = (A + B) @ D`):
1. **Stage 1 — Vec (all rounds)**: compute `A + B` for every round, write
   results to `workspace`.  After all rounds each sub-block sends `FLAG_V2C`.
2. **Stage 2 — Cube (all rounds)**: wait for both `FLAG_V2C` signals, then for
   each round load workspace, compute GEMM, write `C`.

## Workspace sizing

`workspace[batch, TILE_SIZE]` fp16 — one full slot per `(core, round)` pair.
This is much larger than the pipelined FIFO buffers (which hold only a handful
of slots regardless of `num_rounds`).

## Reproduce

```bash
BASE=/workdir/pto-kernels-fork/examples/jit_cpp/cross_core_sync_demo/matmul_add
python $BASE/naive_separate/run.py
```

## Key sync difference vs pipelined variants

| | pipelined (raw_flag / pushpop / gm_pipe) | naive_separate |
|---|---|---|
| Signal granularity | one signal **per round** | one signal **after all rounds** |
| Cube↔Vec overlap | yes — GEMM round r overlaps with Vec round r-1 | no — GEMM finishes before Vec starts |
| Workspace size | `num_cores × TILE_SIZE²` (small FIFO) | `batch × TILE_SIZE` (full array) |
| Bandwidth | higher (pipeline hides latency) | lower (sequential stages) |

## Benchmark results (910B2, TILE_SIZE=128, 24 cores)

### matmul_add_c2v (`C = A @ B + D`)

| batch | rounds | naive µs | naive GB/s | torch µs | torch GB/s |
|-------|--------|----------|------------|----------|------------|
| 3072 | 1 | 53.2 | 44.9 | 35.6 | 67.2 |
| 6144 | 2 | 51.3 | 92.7 | 38.1 | 124.9 |
| 12288 | 4 | 51.4 | 184.3 | 37.9 | 250.1 |
| 24576 | 8 | 51.6 | 366.8 | 36.4 | 519.3 |
| 49152 | 16 | 51.9 | 727.7 | 35.9 | 1052.9 |
| 98304 | 32 | 64.4 | **1173.7** | 36.6 | 2061.9 |
| 196608 | 64 | 141.5 | 1067.1 | 76.0 | 1986.2 |

### add_matmul_v2c (`C = (A + B) @ D`)

| batch | rounds | naive µs | naive GB/s | torch µs | torch GB/s |
|-------|--------|----------|------------|----------|------------|
| 3072 | 1 | 52.0 | 46.1 | 36.3 | 65.8 |
| 6144 | 2 | 50.7 | 93.7 | 36.1 | 131.5 |
| 12288 | 4 | 51.1 | 185.2 | 37.4 | 253.2 |
| 24576 | 8 | 51.0 | 371.1 | 36.7 | 514.9 |
| 49152 | 16 | 55.3 | 683.8 | 36.0 | 1051.0 |
| 98304 | 32 | 64.3 | **1174.2** | 36.8 | 2054.4 |
| 196608 | 64 | 124.7 | **1210.7** | 68.7 | 2197.4 |

Peak naive bandwidth: ~**1174–1211 GB/s** vs pipelined variants: ~**1357–1543 GB/s**.

The pipelined kernels are **15–30% faster** than naive_separate because they
overlap Cube and Vec work round-by-round.  Both are constrained by HBM bandwidth
for large batch sizes.
