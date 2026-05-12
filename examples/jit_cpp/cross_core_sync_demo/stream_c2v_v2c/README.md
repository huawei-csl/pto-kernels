# stream_c2v_v2c — Three Cube↔Vector synchronization API styles

Bandwidth microbenchmarks for the Cube↔Vector workspace handshake, implemented
in three API styles for direct comparison.

## Three API Variants

| Variant | Sync API | Data API | C2V slot | V2C slot |
|---------|----------|----------|----------|----------|
| `raw_flag` | `ffts_cross_core_sync` + `wait_flag_dev` (direct) | `TSTORE`/`TLOAD` on fixed workspace | half 32 KB | half 32 KB |
| `pushpop` | `TPipe` TileData — sync + data-move in one call | built into `TPUSH`/`TPOP` | **float 64 KB** | half 32 KB |
| `gm_pipe` | `TPipe` GlobalData — `TPUSH`/`TPOP` handle sync only | explicit `TALLOC`+`TSTORE`+`TPUSH` / `TPOP`+`TLOAD`+`TFREE` | half 32 KB | half 32 KB |

Key differences:
- **raw_flag**: programmer writes `ffts_cross_core_sync`/`wait_flag_dev` directly + manages workspace manually.
- **pushpop**: one `TPUSH`/`TPOP` call handles both sync and data-move (`TPipe` calls `ffts_cross_core_sync`/`wait_flag_dev` internally). C2V slot stores `AccTile::DType=float` (no implicit fp32→fp16).
- **gm_pipe**: `TPUSH`/`TPOP` with GlobalData overloads handle sync only (`TPipe` manages flags internally); data-move (`TSTORE`/`TLOAD`) is explicit between `TALLOC`+`TPUSH` and `TPOP`+`TFREE`. This allows `TSTORE(slot_half, c_l0)` to perform fp32→fp16 via hardware, matching raw_flag slot size.

## Files

| Subdirectory | Files |
|---|---|
| `raw_flag/` | `stream_c2v.cpp`, `stream_v2c.cpp`, `jit_util_stream.py`, `run_stream_c2v_v2c.py` |
| `pushpop/` | same names |
| `gm_pipe/` | same names (compiled with pto-isa-master headers) |

## Reproduce

```bash
BASE=examples/jit_cpp/cross_core_sync_demo/stream_c2v_v2c

python $BASE/raw_flag/run_stream_c2v_v2c.py
python $BASE/pushpop/run_stream_c2v_v2c.py
python $BASE/gm_pipe/run_stream_c2v_v2c.py

NPU_DEVICE=npu:5 python $BASE/raw_flag/run_stream_c2v_v2c.py  # choose NPU
```

## Results (910B2, 24 Cube cores)

**stream_c2v** — `Cube L0C → workspace → Vec UB`:

| Variant | Slot | Peak (GB/s) | at num_iters |
|---------|------|-------------|--------------|
| raw_flag | half 32 KB | 1148 | 1024 |
| pushpop | **float 64 KB** | **2065** | 1024 (2× slot → 2× bw) |
| gm_pipe | half 32 KB | 1670 | 1024 |

**stream_v2c** — `Vec UB → workspace → Cube L1`:

| Variant | Slot | Peak (GB/s) | at num_iters |
|---------|------|-------------|--------------|
| raw_flag | half 32 KB | 1096 | 128 |
| pushpop | half 32 KB | 1089 | 128 |
| gm_pipe | half 32 KB | 1229 | 512 |

Note: `pushpop` C2V uses a float32 slot (64 KB) so its bandwidth is naturally 2× the half-slot variants. For a like-for-like comparison, divide the `pushpop` C2V bandwidth by 2 (~1033 GB/s), which is comparable to raw_flag (1148 GB/s).

**Sync optimization applied** (vs initial implementation with `pipe_barrier(PIPE_ALL)` everywhere):
- TMATMUL → TSTORE (Cube): replaced `pipe_barrier(PIPE_ALL)` with `SetFlag<M,FIX>; WaitFlag<M,FIX>`
- TLOAD+TLOAD → TADD (Vec): replaced with `SetFlag<MTE2,V>; WaitFlag<MTE2,V>`
- TADD → TSTORE (Vec): replaced with `SetFlag<V,MTE3>; WaitFlag<V,MTE3>`
- DMA → cross-core signal: `pipe_barrier(PIPE_ALL)` **kept** (required for memory visibility)
