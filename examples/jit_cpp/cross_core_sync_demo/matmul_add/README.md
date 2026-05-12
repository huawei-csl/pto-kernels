# matmul_add — Three Cube↔Vector synchronization API styles

Persistent kernels computing `C = A @ B + D` (C2V) and `C = (A + B) @ D` (V2C),
implemented in three API styles for direct syntax and performance comparison.

## Three API Variants

| Variant | Sync API | Data API | Note |
|---------|----------|----------|------|
| `raw_flag` | `ffts_cross_core_sync` + `wait_flag_dev` (direct) | `TSTORE`/`TLOAD` on workspace | Proven correct for all batch sizes |
| `pushpop` | `TPipe` TileData — sync + data-move in one call | built into `TPUSH`/`TPOP` | num_rounds=1 scope; multi-round has shared tileIndex issue |
| `gm_pipe` | `TPipe` GlobalData — `TPUSH`/`TPOP` handle sync only (C2V); raw ffts fallback for V2C | explicit `TALLOC`+`TSTORE`+`TPUSH` / `TPOP`+`TLOAD`+`TFREE` | Full multi-round correctness; requires pto-isa-master headers |

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
```

## Known Limitations

- **pushpop multi-round**: TileData TPUSH/TPOP with `TILE_UP_DOWN` and 2 Vec sub-blocks shares `pipe.prod.tileIndex` between sub-blocks, advancing it by 2 per logical round instead of 1.  This de-syncs producer/consumer FIFO slot indices for `num_rounds > 1`.  Use the `gm_pipe` variant for multi-round workloads.

- **gm_pipe header requirement**: `TALLOC`/`TPOP(GlobalData)`/`TFREE` are in `pto-isa-master` headers, not the default `/sources/pto-isa`.  The `gm_pipe/jit_util.py` uses `-I/workdir/pto-isa-master/include`.
