# matmul_add — Three Cube↔Vector synchronization API styles + naive baseline

Persistent kernels computing `C = A @ B + D` (C2V) and `C = (A + B) @ D` (V2C),
implemented in three pipelined API styles and one non-pipelined naive baseline.

## Variants

| Subdirectory | Sync API | Pipeline | Note |
|---|---|---|---|
| `raw_flag/` | `ffts_cross_core_sync` + `wait_flag_dev` (direct) | round-level overlap | Reference, full multi-round correctness |
| `pushpop/` | `TPipe` TileData — sync + data-move in one call | round-level overlap | `num_rounds=1` scope; multi-round has shared tileIndex issue |
| `gm_pipe/` | `TPipe` GlobalData — `TPUSH`/`TPOP` signal only | round-level overlap | Full multi-round; requires pto-isa-master headers |
| `naive_separate/` | `ffts_cross_core_sync` (one signal per stage) | **none** — stages are sequential | Slower baseline; shows pipeline benefit |

## Kernel Algorithms

| Kernel | Operation | C2V or V2C |
|--------|-----------|------------|
| `matmul_add_c2v` | `C = A @ B + D` | Cube GEMM → workspace → Vec add |
| `add_matmul_v2c` | `C = (A + B) @ D` | Vec add → workspace → Cube GEMM |

## Files

| Subdirectory | Kernels | Python | Note |
|---|---|---|---|
| `raw_flag/` | `matmul_add_c2v.cpp`, `add_matmul_v2c.cpp` | `jit_util_*.py`, `run_*.py` | Reference |
| `pushpop/` | same | `jit_util.py`, `run.py` | Single-round scope |
| `gm_pipe/` | same | `jit_util.py`, `run.py` | pto-isa-master headers |
| `naive_separate/` | `naive_separate.cpp` (both kernels) | `jit_util.py`, `run.py` | No pipeline |

## Reproduce

```bash
BASE=/workdir/pto-kernels-fork/examples/jit_cpp/cross_core_sync_demo/matmul_add

# raw_flag: full multi-round correctness (30/30 seeds × rounds)
python $BASE/raw_flag/run_matmul_add_c2v.py
python $BASE/raw_flag/run_add_matmul_v2c.py

# pushpop: num_rounds=1 scope (5/5 seeds)
python $BASE/pushpop/run.py

# gm_pipe: multi-round correctness (3 batch sizes × 2 kernels)
python $BASE/gm_pipe/run.py

# naive_separate: baseline (30/30 seeds × rounds, both kernels)
python $BASE/naive_separate/run.py
```

## API Syntax Comparison (C2V direction: `C = A @ B + D`)

```
                   Sync API                    │  Data API
──────────────────────────────────────────────────────────────────────────
raw_flag  Cube: ffts_cross_core_sync(FIX, FLAG_C2V)  │  TSTORE(ws_half, c_l0)
          Vec:  wait_flag_dev(FLAG_C2V)               │  TLOAD(c_ub, ws)
                ffts_cross_core_sync(MTE3, FLAG_V2C)  │

pushpop   Cube: TPUSH<C2VPipe, TileL0C, UP_DOWN>(pipe, c_l0)     ← sync + data in one call
          Vec:  TPOP<C2VPipe, VecTile<float>, UP_DOWN>(pipe, c_ub_float)

gm_pipe   Cube: TALLOC<C2VPipe, SlotHalf, UP_DOWN>(pipe, slot)   ← TPipe allocates slot
                TSTORE(slot, c_l0)               ← explicit fp32→fp16 (hardware FIX)
                TPUSH<C2VPipe, SlotHalf, UP_DOWN>(pipe, slot)    ← TPipe signals consumer
          Vec:  TPOP<C2VPipe, PopHalf, UP_DOWN>(pipe, pop)        ← TPipe waits + slot ptr
                TLOAD(c_ub, pop)                 ← explicit load
                TFREE<C2VPipe, PopHalf, UP_DOWN>(pipe, pop)       ← TPipe notifies free

naive     Cube: (all GEMMs) → pipe_barrier → ffts_cross_core_sync(FIX, FLAG_C2V)
          Vec:  wait_flag_dev(FLAG_C2V) → (all adds)
               ↑ one signal after ALL rounds, no round overlap
```

## Measured Bandwidth (910B2, TILE_SIZE=128, 24 Cube cores)

Peak effective external bandwidth (read A+B+D, write C; workspace not counted):

| Variant | matmul_add_c2v peak | add_matmul_v2c peak | Notes |
|---------|--------------------|--------------------|-------|
| `raw_flag` | **1357 GB/s** | **1543 GB/s** | Reference pipelined, 64 rounds |
| `pushpop` | ~50 GB/s | ~30 GB/s | rounds=1 scope only (batch=3072); limited by small-batch overhead, not algorithm |
| `gm_pipe` | **1784 GB/s** | **1478 GB/s** | 64 rounds; requires pto-isa-master headers |
| `naive_separate` | 1174 GB/s | 1211 GB/s | No pipeline — **15–30% lower** |
| `torch.mm + torch.add` | ~2000 GB/s\* | ~2100 GB/s\* | Two separate launches |

\* torch bandwidth appears high because torch's GEMM is a highly tuned library kernel
that may cache intermediate results on-chip; the naive kernel instead round-trips
through full-batch HBM workspace, making it slower than torch for large batches.
The pipelined variants are faster than naive because they overlap Cube and Vec
round-by-round, reducing the effective latency of cross-core data movement.

## Known Limitations

- **pushpop multi-round**: TileData TPUSH/TPOP with `TILE_UP_DOWN` and 2 Vec sub-blocks shares `pipe.prod.tileIndex` between sub-blocks, advancing it by 2 per logical round instead of 1.  This de-syncs producer/consumer FIFO slot indices for `num_rounds > 1`.  Use the `gm_pipe` variant for multi-round workloads.

- **gm_pipe header requirement**: `TALLOC`/`TPOP(GlobalData)`/`TFREE` are in `pto-isa-master` headers, not the default `/sources/pto-isa`.  The `gm_pipe/jit_util.py` uses `-I/workdir/pto-isa-master/include`.

- **naive_separate workspace**: Uses `workspace[batch, TILE_SIZE]` (full-batch allocation) vs `workspace[num_cores * TILE_SIZE, TILE_SIZE]` for pipelined variants.  The larger workspace means more HBM traffic per kernel call at large batch sizes.
