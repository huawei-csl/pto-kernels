# Porting Raw CCE Intrinsics to PTO-ISA

A practical guide for migrating Ascend NPU kernels from low-level CCE intrinsics to the
higher-level pto-isa tile API (`TLOAD` / `TSTORE` / `TADDS` / `TSync_Custom` / `TPipe`).

The completed reference port lives in [`examples/jit_cpp/c2v_sync/`](../examples/jit_cpp/c2v_sync/).

---

## Intrinsic → wrapper mapping

| Raw CCE intrinsic | PTO-ISA wrapper | Condition |
|---|---|---|
| `copy_gm_to_ubuf(dst, src, sid, nBurst, burstLen, ubGap, gmGap)` | `TLOAD(vec_tile, global_tensor)` | Tile must be `TileType::Vec` |
| `copy_ubuf_to_gm(dst, src, sid, nBurst, burstLen, gmGap, ubGap)` | `TSTORE(global_tensor, vec_tile)` | Tile must be `TileType::Vec` |
| `copy_gm_to_cbuf(dst, src, sid, nBurst, burstLen, l1Gap, gmGap, PAD_NONE)` | `TLOAD(mat_tile, global_tensor)` | Tile must be `TileType::Mat` |
| `copy_cbuf_to_gm(dst, src, sid, nBurst, burstLen, gmGap, l1Gap)` | `TSTORE(global_tensor, mat_tile)` | Tile must be `TileType::Mat` |
| `vadds(dst, src, scalar, repeat, ...)` loop | `TADDS(tile, tile, scalar)` | See [count-mode gotcha](#tadds-count-mode-gotcha) |
| `ffts_cross_core_sync(PIPE_MTE3, msg)` + `wait_flag_dev(id)` | `TSync_Custom<...>::record()` / `::wait()` | See [PIPE_FIX gotcha](#pipe_fix-vs-pipe_mte3-gotcha) |
| same | `C2VPipe::Producer::record()` / `Consumer::wait()` | TPipe alternative |

---

## Tile type selection

`TileType` determines which copy intrinsic TLOAD/TSTORE dispatches to.  A mismatch is
silent — the wrong intrinsic is called with no compile error.

```cpp
// Vec tile — maps to UB (Unified Buffer), used on AIV (vector core, __DAV_C220_VEC__)
using VecTile = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

// Mat tile — maps to L1/cbuf, used on AIC (cube core, __DAV_C220_CUBE__)
using MatTile = Tile<TileType::Mat, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

// Acc tile — maps to matrix accumulator; used only as a TPipe placeholder (see below)
using ProdTile = TileAcc<float, 16, 16>;
```

---

## GlobalTensor burst layout

Design `cols = 256` floats (= 1 KB) to match the original `burstLen = 32` (32-byte blocks).
Set `rows` dynamically via a `DYNAMIC` shape dimension.

```cpp
using GMShape    = Shape<1, 1, 1, DYNAMIC, 256>;   // last dim = 256 floats = 1 KB burst
using GMStride   = Stride<1, 1, 1, 256, 1>;        // contiguous, no gaps
using GlobalFP32 = GlobalTensor<float, GMShape, GMStride>;

// At runtime — pass n_rows as the DYNAMIC argument
GlobalFP32 g(gm_ptr + offset, GMShape(1, 1, 1, n_rows, 256));
```

Only `DYNAMIC` dimensions are written at runtime; static dims are baked into the type.

Typical row counts:
- Cube core (2×N elements): `rows = N * 2 / 256`
- Vector sub-core (N elements): `rows = N / 256`

---

## TADDS count-mode gotcha

> **Silent bug:** if tile `Rows` is `DYNAMIC` (= −1), TADDS produces a no-op.

`TBinSInstr` decides between *norm mode* and *count mode* at template-instantiation time
using the static `Rows` value.  When `Rows = DYNAMIC = −1`:

```
totalRepeats = (−1 × 256 + 63) / 64 = −3   ≤ 255  →  NormMode selected
→  vadds(..., repeat=256, ...)   ← 256 overflows uint8_t to 0   → no-op
```

**Fix:** use a static `Rows` large enough that `totalRepeats > 255`:

```cpp
// Rows=256, Cols=256  →  totalRepeats = (256×256+63)/64 = 1024 > 255  →  count mode
using VecTile = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
//                                         ^^^  ^^^  static — force count mode
//                                                   DYNAMIC valid dims still used at runtime
```

Count mode internally calls `set_mask_count()` + `SetVectorCount(n)` + `vadds(..., repeats=0)`,
handling arbitrary runtime element counts without overflow.

---

## PIPE_FIX vs PIPE_MTE3 gotcha

> **Ordering hazard:** the original kernel uses `PIPE_MTE3`; pto-isa sync wrappers use `PIPE_FIX`.

The original:
```cpp
ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (0 << 8));  // fires after GM writes drain
```

`TSync_Custom::record()` and `TPipe::Producer::record()` both emit:
```cpp
ffts_cross_core_sync(PIPE_FIX, ...);  // different pipe — GM writes may not have drained yet
```

**Fix:** drain MTE3 explicitly before calling `.record()`:

```cpp
pipe_barrier(PIPE_MTE3);   // ← drain GM-write pipeline
c2v_sync.record();         // now safe: vector will see the written data
```

---

## TSync_Custom (TSYNC version)

```cpp
#include <pto/npu/a2a3/custom/TSync_Custom.hpp>
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>

// flag_id = 0, baked in via aggregate initializer
constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> c2v_sync = {0};

// Cube side (__DAV_C220_CUBE__)
pipe_barrier(PIPE_MTE3);
c2v_sync.record();   // → ffts_cross_core_sync(PIPE_FIX, _getFFTSMsg(CV_CORE_SYNC, 0))

// Vector side (__DAV_C220_VEC__)
c2v_sync.wait();     // → wait_flag_dev(0)
```

Template args `<TSTORE_C2GM, TLOAD>` encode the data-flow direction: cube stores to GM,
vector loads from GM.

---

## TPipe Producer / Consumer (TPUSH/TPOP version)

```cpp
#include <pto/npu/a2a3/TPop.hpp>   // includes TPush.hpp → TPipe

// ProdTile=Acc, ConsTile=Vec → is_c2v = true
using ProdTile = TileAcc<float, 16, 16>;
using ConsTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
using C2VPipe  = TPipe<0, FIFOType::GM_FIFO, 1, 1, ProdTile, ConsTile>;
//                     ^ FLAG_ID=0

// Cube side
pipe_barrier(PIPE_MTE3);
C2VPipe::Producer prod;
prod.record();   // → ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(CV_CORES_SYNC, 0))

// Vector side
C2VPipe::Consumer cons;
cons.wait();     // → wait_flag_dev(0)
```

> **Note:** ProdTile / ConsTile are **placeholders only** — they influence direction inference
> (`is_c2v`), not data movement.  Actual data transfer is done by TLOAD / TSTORE.

---

## Full kernel skeleton

```cpp
#include <pto/pto-inst.hpp>
// + one of:
//   #include <pto/npu/a2a3/custom/TSync_Custom.hpp>  (TSYNC version)
//   #include <pto/npu/a2a3/TPop.hpp>                 (TPUSH/TPOP version)
#include "runtime/rt.h"

using namespace pto;

using VecTile  = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
using MatTile  = Tile<TileType::Mat, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
using GMShape  = Shape<1, 1, 1, DYNAMIC, 256>;
using GMStride = Stride<1, 1, 1, 256, 1>;
using GlobalFP32 = GlobalTensor<float, GMShape, GMStride>;

extern "C" __global__ __aicore__ void my_kernel(
    __gm__ float * __restrict__ gm_input,
    __gm__ float * __restrict__ gm_output,
    __gm__ uint8_t * __restrict__ ffts_addr,
    int32_t N)
{
#ifdef __DAV_C220_CUBE__
    set_ffts_base_addr((uint64_t)ffts_addr);
    set_padding(0);
    set_atomic_none();

    int rows = N * 2 / 256;
    GlobalFP32 gIn (gm_input  + get_block_idx() * N * 2, GMShape(1, 1, 1, rows, 256));
    GlobalFP32 gOut(gm_output + get_block_idx() * N * 2, GMShape(1, 1, 1, rows, 256));
    MatTile l1(rows, 256);
    TASSIGN(l1, (uint32_t)0x0);          // bind to cbuf base

    TLOAD(l1, gIn);                      // GM → L1  (MTE2)
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(gOut, l1);                    // L1 → GM  (MTE3)

    pipe_barrier(PIPE_MTE3);            // ← drain before record()
    /* sync.record() or prod.record() */
    pipe_barrier(PIPE_ALL);
#endif

#ifdef __DAV_C220_VEC__
    set_atomic_none();
    set_mask_norm();
    set_ffts_base_addr((uint64_t)ffts_addr);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    int id   = get_block_idx() * get_subblockdim() + get_subblockid();
    int rows = N / 256;
    GlobalFP32 gOut(gm_output + id * N, GMShape(1, 1, 1, rows, 256));
    VecTile ub(rows, 256);
    TASSIGN(ub, (uint32_t)0x0);          // bind to UB base

    /* sync.wait() or cons.wait() */
    TLOAD(ub, gOut);                     // GM → UB  (MTE2)
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADDS(ub, ub, scalar);               // replaces vadds loop
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gOut, ub);                    // UB → GM  (MTE3)
    pipe_barrier(PIPE_ALL);
#endif
}
```

---

## File-scope type alias gotcha

`pto_tile.hpp` (which defines `DYNAMIC`, `Shape`, `Stride`, `GlobalTensor`) is only
included by `pto-inst.hpp` when `__CCE_AICORE__` is defined — i.e., in device-only
compilation contexts.  File-scope `using` aliases are compiled in host context too, where
those types do not exist, causing `use of undeclared identifier 'DYNAMIC'` errors.

**Fix:** declare tile and global type aliases **inside** the appropriate `#ifdef __DAV_C220_*__`
block, not at file scope.  Aliases needed in both Cube and Vec blocks must be repeated in each:

```cpp
// ✗ Wrong — file scope, fails in host compilation
using VecTile = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

// ✓ Correct — inside device-only block
#ifdef __DAV_C220_VEC__
    using VecTile    = Tile<TileType::Vec, float, 256, 256, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using GMShape    = Shape<1, 1, 1, DYNAMIC, 256>;
    using GlobalFP32 = GlobalTensor<float, GMShape, Stride<1, 1, 1, 256, 1>>;
    ...
#endif
```

`TPipe` / `TSync_Custom` type aliases that reference only file-scope types (e.g.,
`TileAcc`, static-dim `Tile`) are safe at file scope because they don't use `DYNAMIC`.

---

## Porting checklist

- [ ] `copy_gm_to_ubuf` → `TLOAD(VecTile, GlobalTensor)`
- [ ] `copy_ubuf_to_gm` → `TSTORE(GlobalTensor, VecTile)`
- [ ] `copy_gm_to_cbuf` → `TLOAD(MatTile, GlobalTensor)`
- [ ] `copy_cbuf_to_gm` → `TSTORE(GlobalTensor, MatTile)`
- [ ] `vadds` loop → `TADDS(tile, tile, scalar)`
- [ ] `ffts_cross_core_sync` + `wait_flag_dev` → `TSync_Custom` or `TPipe`
- [ ] **Static `Rows=256` (not `DYNAMIC`) on any tile passed to `TADDS`**
- [ ] **`pipe_barrier(PIPE_MTE3)` before every `.record()` call**
- [ ] `DYNAMIC` valid dims (`RowValid`, `ColValid`) for runtime tile sizes
- [ ] `Shape<..., DYNAMIC, 256>` with `n_rows` passed to GlobalTensor constructor
- [ ] `TASSIGN(tile, (uint32_t)0x0)` to bind tile to buffer base address
- [ ] Compile: `bisheng -xcce --npu-arch=dav-2201 -DMEMORY_BASE -I${PTO_LIB_PATH}/include`

---

## Reference files in pto-isa

| File | Purpose |
|---|---|
| `include/pto/npu/a2a3/TLoad.hpp` | TLOAD dispatch (Vec→UB, Mat→L1, Acc→matrix) |
| `include/pto/npu/a2a3/TStore.hpp` | TSTORE dispatch |
| `include/pto/npu/a2a3/TAddS.hpp` | TADDS entry point |
| `include/pto/npu/a2a3/TBinSOp.hpp` | TADDS / count-mode selection logic |
| `include/pto/npu/a2a3/custom/TSync_Custom.hpp` | TSync_Custom record/wait |
| `include/pto/npu/a2a3/custom/TSyncCVID.hpp` | `_getFFTSMsg`, `CVSyncMode` |
| `include/pto/npu/a2a3/TPush.hpp` + `TPop.hpp` | TPipe Producer/Consumer |
| `include/pto/common/pto_tile.hpp` | `Tile`, `GlobalTensor`, `Shape`, `DYNAMIC` |
