# PTO API Known Bugs

This document records confirmed bugs and their workarounds in the PTO-ISA
library (`pto-isa-master`), found while implementing the kernels in this
`cross_core_sync_demo` directory.

---

## Bug 1 — `TPipe` (TileData TPUSH/TPOP): `tileIndex` shared between Vec sub-blocks breaks multi-round kernels

### Status

**Confirmed present in**:
- `/sources/pto-isa/include` (Ascend CANN 8.5.0 bundled headers)
- `pto-isa-master` HEAD as of 2026-05-12 (commit `933ad5d8`)

The pto-isa maintainers acknowledged the issue by changing their own reference
test (`tests/npu/a2a3/src/st/testcase/tpushpop_cv/tpushpop_cv_kernel.cpp`) from
`FIFO_DEPTH=2` to `FIFO_DEPTH=1` in commit `aef3a004` (PR !895, "optimize
reverse dependency with synchronization period", merged 2026-05-07).

### Affected API

`TPUSH` / `TPOP` — TileData overloads (not the GlobalData / gm_pipe overloads):

```cpp
// Producer side (Cube in C2V, Vec in V2C)
TPUSH<PipeType, TileProd, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);

// Consumer side (Vec in C2V, Cube in V2C)
TPOP<PipeType, TileCons, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
```

The bug is specific to `TileSplitAxis::TILE_UP_DOWN` (or any split that causes
2 Vec sub-blocks to call TPUSH or TPOP independently).  `TILE_NO_SPLIT` is
believed to be unaffected.

### Root Cause

`TPipe<FlagID, Dir, SlotSize, FIFO_DEPTH>` stores a single `tileIndex` counter
per `Producer` and per `Consumer` struct (`pipe.prod.tileIndex` /
`pipe.cons.tileIndex`).  With `TILE_UP_DOWN`, a single core has **two** Vec
sub-blocks (`vid = 0` and `vid = 1`); each sub-block independently calls TPUSH
or TPOP in its own code path.

Because `tileIndex++` fires once per TPUSH/TPOP call:

| Direction | Who calls TPUSH | Who calls TPOP | Effect |
|-----------|-----------------|----------------|--------|
| C2V | 1 Cube core — once per round | 2 Vec sub-blocks — once each per round | `cons.tileIndex` advances by **2** per round; `prod.tileIndex` advances by 1 → desync after round 1 |
| V2C | 2 Vec sub-blocks — once each per round | 1 Cube core — once per round | `prod.tileIndex` advances by **2** per round; `cons.tileIndex` advances by 1 → desync after round 1 |

After N logical rounds with `FIFO_DEPTH=2`, `SyncPeriod=2`:
- The side with 2 sub-blocks has `tileIndex = 2N`; the other side has `tileIndex = N`
- The slot selected by `tileIndex % SLOT_NUM` drifts: the 2-sub-block side
  starts reading/writing the wrong FIFO slot from round 2 onwards
- The `shouldWaitFree` / `shouldNotifyFree` conditions also fire at wrong
  intervals, causing the FFTS signal counts to diverge

### Observed Failures

**C2V (`matmul_add_c2v`, `stream_c2v`):**
- `num_rounds = 1`: correct
- `num_rounds = 2`: wrong numerical results (`max_diff ≈ 70` for fp32 output)
- `num_rounds ≥ 4`: hardware exception — `L0C read/write conflict (FIXP reads
  l0c, same address as cube write)`

**V2C (`add_matmul_v2c`, `stream_v2c`):**
- `num_rounds = 1`: correct
- `num_rounds ≥ 2`: wrong numerical results and/or hardware exception

Errors are **deterministic** (reproducible on every run with the same seed).

### Minimal Reproduction

```cpp
// C2V direction — fails at num_rounds=2 with FIFO_DEPTH=2
constexpr uint32_t FIFO_DEPTH = 2;
using C2VPipe = TPipe<0, Direction::DIR_C2V, C2V_SLOT_SIZE, FIFO_DEPTH>;
// ...
for (int32_t r = 0; r < num_rounds; ++r) {
    TPUSH<C2VPipe, TileL0C, TileSplitAxis::TILE_UP_DOWN>(pipe, c_l0);  // Cube
    TPOP<C2VPipe, VecTile,  TileSplitAxis::TILE_UP_DOWN>(pipe, c_ub);  // Vec ×2 sub-blocks
}
```

See `matmul_add/pushpop/matmul_add_c2v.cpp` and `add_matmul_v2c.cpp` for the
full implementations that reproduce the failure.

### Expected Behavior

A kernel with `num_rounds > 1` using `TILE_UP_DOWN` should:
1. Maintain correct FIFO slot selection across all rounds
2. Maintain balanced FFTS signal counts (no accumulation)
3. Produce correct numerical output for any `num_rounds ≥ 1`

### Workarounds

#### Workaround A — `FIFO_DEPTH=1` (pto-isa maintainers' approach)

Change the pipe depth to 1.  With `SlotNum=1`, `SyncPeriod=1` (per
`TPipe::SyncPeriod` formula), and the new `shouldWaitFree` code (PR !895)
always returns `true` for `SlotNum == 1`.  This forces strict producer↔consumer
alternation — no double-buffering — which avoids the tileIndex desync at the
cost of pipeline overlap:

```cpp
constexpr uint32_t FIFO_DEPTH = 1;  // was 2
using C2VPipe = TPipe<0, Direction::DIR_C2V, C2V_SLOT_SIZE, FIFO_DEPTH>;
```

**Important**: the Python-side `fifo_mem` allocation must also reflect
`FIFO_DEPTH=1`:
```python
C2V_FIFO_ELEMS_PER_CORE = 1 * TILE_SIZE * TILE_SIZE  # not 2× anymore
```

**Note**: this workaround also requires fresh `fifo_mem` per kernel call in
Python benchmarks.  Reusing the same `fifo_mem` tensor across repeated calls
accumulates TPipe head/tail state (stored inside `fifo_mem`) and causes wrong
results or hangs.  Pre-allocate one `fifo_mem` per call:
```python
fifos = [torch.zeros(BLOCK_DIM * FIFO_ELEMS_PER_CORE, ...) for _ in range(n_calls)]
for i in range(n_calls):
    kernel(A, B, C, D, fifos[i])
```

#### Workaround B — `gm_pipe` variant (GlobalData TPUSH/TPOP + explicit TALLOC/TFREE)

Use the GlobalData overloads of TPUSH/TPOP together with TALLOC/TFREE and
explicit TSTORE/TLOAD.  The `gm_pipe` implementation in this demo manages FIFO
slot indices manually via `r % FIFO_DEPTH`, completely bypassing the shared
`tileIndex` counter.  This variant supports arbitrary `num_rounds` with
`FIFO_DEPTH=2`.

See `matmul_add/gm_pipe/` and `stream_c2v_v2c/gm_pipe/`.

**Important**: `gm_pipe` requires the newer `pto-isa-master` headers (not the
CANN 8.5.0 bundled headers), because `TALLOC`, `TPOP(GlobalData)`, and `TFREE`
are absent from `/sources/pto-isa/include`.

#### Workaround C — raw FFTS flags (`raw_flag` variant)

Avoid TPipe entirely.  Use `ffts_cross_core_sync` / `wait_flag_dev` directly
with explicit workspace memory.  Supports arbitrary `num_rounds` with no tileIndex
issue.  See `matmul_add/raw_flag/` and `stream_c2v_v2c/raw_flag/`.

### Summary Table

| Variant | API | Multi-round | Notes |
|---------|-----|-------------|-------|
| `pushpop` (FIFO_DEPTH=2) | TileData TPUSH/TPOP | ❌ broken ≥2 rounds | This bug |
| `pushpop` (FIFO_DEPTH=1) | TileData TPUSH/TPOP | ✅ correct | No double-buffer overlap |
| `gm_pipe` | GlobalData TPUSH/TPOP + TALLOC/TFREE | ✅ correct | Newer headers required |
| `raw_flag` | Direct FFTS + manual workspace | ✅ correct | Most portable |

---

## Bug 2 — FFTS flag collision between kernels with the same `FlagID`

### Status

**Design limitation** (not a library bug per se, but a footgun).

### Description

`TPipe<FlagID, ...>` uses FFTS hardware flags `FlagID` (push/data-ready signal)
and `FlagID+1` (free/slot-available signal) internally.  When two different
kernels or pipe types use the same `FlagID`, their FFTS signals contaminate
each other if the kernels are called sequentially in the same process on the
same NPU core.

**Example**: `C2VPipe = TPipe<0, DIR_C2V>` and `V2CPipe = TPipe<0, DIR_V2C>`
both occupy FFTS flags 0 and 1.  A benchmark that calls the C2V kernel many
times accumulates residual FFTS signals on flags 0/1.  The subsequent V2C
kernel's first TPOP fires on a stale signal and reads wrong data.

### Fix

Assign non-overlapping `FlagID` values to pipes that are called from the same
process:

```cpp
using C2VPipe = TPipe<0, Direction::DIR_C2V, ...>;  // uses flags 0, 1
using V2CPipe = TPipe<2, Direction::DIR_V2C, ...>;  // uses flags 2, 3 — no collision
```

This fix is applied in:
- `stream_c2v_v2c/pushpop/stream_v2c.cpp`
- `stream_c2v_v2c/gm_pipe/stream_v2c.cpp`
- `matmul_add/gm_pipe/add_matmul_v2c.cpp` (uses raw FFTS flags 2/3 instead of 0/1)

---

## Bug 3 — `TSTORE(c_global, c_l0)` (FIX pipe) conflicts with next-call `TMATMUL` (M pipe) in benchmark loops

### Status

**Synchronization omission** in the kernel itself, exposed by benchmark loops.

### Description

`TSTORE(dst_gm, c_l0)` on the FIX pipe initiates a DMA that reads from `c_l0`
(L0C) and writes to global memory.  The DMA may still be in-flight when the
kernel "completes" (all pipe instructions issued).  If back-to-back kernel calls
are queued in the same NPU stream (as in a benchmark loop), the **next** call's
`TMATMUL` can start writing to `c_l0` (M pipe) before the **previous** call's
FIX DMA finishes reading it → `L0C read/write conflict` hardware exception.

This does NOT manifest in correctness tests (few calls) but reliably crashes
under benchmark load (`REPEATS=30` calls in a tight loop).

### Fix

Add `pipe_barrier(PIPE_ALL)` immediately after the last `TSTORE(c_global, c_l0)`
in the Cube loop to drain the FIX pipe before kernel exit:

```cpp
for (int32_t r = 0; r < num_rounds; ++r) {
    // ...
    TSTORE(c_global, c_l0);
    pipe_barrier(PIPE_ALL);  // ← drain FIX before kernel exit / next TMATMUL
}
```

Or use the targeted `SetFlag<PIPE_FIX, PIPE_M>(1); WaitFlag<PIPE_FIX, PIPE_M>(1);`
pair after each TSTORE (requires an additional `SetFlag<PIPE_M, PIPE_MTE1>(1);
WaitFlag<PIPE_M, PIPE_MTE1>(1);` guard on L0A reuse — see
`matmul_add/gm_pipe/add_matmul_v2c.cpp` for the full treatment).

This fix is applied in `matmul_add/raw_flag/add_matmul_v2c.cpp` and the
`gm_pipe` variants.
