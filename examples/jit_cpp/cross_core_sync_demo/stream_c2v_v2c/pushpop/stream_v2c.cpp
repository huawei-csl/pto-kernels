// =============================================================================
// stream_v2c.cpp — Vector→Cube bandwidth microbenchmark  [pushpop variant]
//
// Same algorithm as raw_flag/stream_v2c.cpp:
//   Vec loads A+D from GM, adds them in UB, writes to FIFO; Cube pops into L1
//   and discards the data (pure bandwidth test, no GEMM on Cube side).
//
// ── API variant: TileData TPUSH / TPOP ──────────────────────────────────────
//
//  raw_flag equivalent               │  pushpop (this file)
//  ──────────────────────────────────┼────────────────────────────────────────
//  TSTORE(ws_half, a_ub)             │  TPUSH<V2CPipe, VecTile, UP_DOWN>(pipe, a_ub)
//  pipe_barrier(PIPE_ALL)            │    └─ internally: TSTORE(GlobalTensor<half>, a_ub)
//  SetCrossFlag<MTE3>(FLAG_V2C)      │       then data-ready signal
//  ──────────────────────────────────┼────────────────────────────────────────
//  WaitCrossFlag(FLAG_V2C)           │  TPOP<V2CPipe, TileL1, UP_DOWN>(pipe, ws_l1)
//  TLOAD(ws_l1, ws_half)             │    └─ internally: wait, TLOAD(GlobalTensor<half>, ws_l1)
//  SetCrossFlag<MTE2>(FLAG_C2V)      │       then free-space notify
//
// Key difference from raw_flag:
//   VecTile::DType = half — TPUSH stores half into the slot (same as raw_flag).
//   SlotSize = T²×sizeof(half) = 32 KB (identical to raw_flag workspace size).
//   TPipe manages double-buffering automatically; no explicit FLAG_C2V/FLAG_V2C.
//
// Memory budget (per core):
//   L1 (512 KB): ws_l1 — managed by TPOP via V2C_CONSUMER_BUF
//                Slot 0 at L1:0x00000 (32 KB), Slot 1 at L1:0x08000 (32 KB)
//   UB (192 KB): a_ub 16 KB + b_ub 16 KB = 32 KB used
//   FIFO GM    : FIFO_DEPTH × SLOT_SIZE = 2 × 32 KB = 64 KB per core
// =============================================================================

#define MEMORY_BASE
#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>

using namespace pto;

#define TILE_SIZE 128
#define HALF_TILE  64
#define VEC_NUM     2

#ifdef __CCE_AICORE__

// ── On-chip UB buffer offsets (bytes) ─────────────────────────────────────────
constexpr uint32_t UB_A_OFFSET = 0u;                                     // a_ub: 16 KB
constexpr uint32_t UB_B_OFFSET = HALF_TILE * TILE_SIZE * sizeof(half);   // b_ub: 16 KB

// ── FIFO configuration ────────────────────────────────────────────────────────
// Slot holds VecTile::DType = half.  Two slots for double-buffered overlap.
constexpr uint32_t FIFO_DEPTH      = 2u;
constexpr uint32_t V2C_SLOT_SIZE   = TILE_SIZE * TILE_SIZE * sizeof(half); // 32 KB
constexpr uint32_t V2C_FIFO_BYTES  = FIFO_DEPTH * V2C_SLOT_SIZE;           // 64 KB/core

// L1 base for Cube to receive TPOP'd V2C tiles (managed by TPipe internally).
constexpr uint32_t V2C_L1_BASE     = 0x0;

// ── Tile types ────────────────────────────────────────────────────────────────
// L1 consumer tile: Cube pops the full T×T half tile from the FIFO.
using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;

// Vec UB tiles: half, each sub-block owns HALF_TILE rows.
using TileVecUB = Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE,
                       BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                       SLayout::NoneBox, 512, PadValue::Null>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }

// ── FIFO pipe type ────────────────────────────────────────────────────────────
// FlagID=0, DIR_V2C, slot=half32KB, depth=2.
// TPUSH calls TSTORE(GlobalTensor<half>, a_ub) — same dtype, no conversion.
// TPOP  calls TLOAD(GlobalTensor<half>, ws_l1) — loads half into L1.
using V2CPipe = TPipe<0, Direction::DIR_V2C, V2C_SLOT_SIZE, FIFO_DEPTH>;

using HalfTileGlobal =
    GlobalTensor<half,
                 TileShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

// ─────────────────────────────────────────────────────────────────────────────
AICORE void run_stream_v2c(
    __gm__ half    *A,         // [num_iters × num_cores × T, T]  Vec input 1
    __gm__ half    *D,         // [num_iters × num_cores × T, T]  Vec input 2
    __gm__ uint8_t *fifo_mem,  // [num_cores × V2C_FIFO_BYTES]  V2C FIFO buffer
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows = num_cores * TILE_SIZE;

    // Each core owns its own FIFO region.
    __gm__ void *core_fifo = fifo_mem + cid * V2C_FIFO_BYTES;
    V2CPipe pipe(core_fifo, /*c2v_ub_base=*/0x0, /*v2c_l1_base=*/V2C_L1_BASE);

    // ── Allocate UB buffers ────────────────────────────────────────────────────
    TileVecUB a_ub, b_ub;
    TASSIGN(a_ub, UB_A_OFFSET);
    TASSIGN(b_ub, UB_B_OFFSET);

    // ws_l1 is assigned internally by TPOP (no manual TASSIGN needed).
    TileL1 ws_l1;

    // ── Cube: pop loop — receives half tiles from FIFO, discards data ─────────
#if defined(__DAV_C220_CUBE__)

    for (int32_t r = 0; r < num_iters; ++r) {
        // ── pushpop replaces raw_flag: ───────────────────────────────────────
        // raw_flag: wait_flag_dev(FLAG_V2C)
        //           TLOAD(ws_l1, ws_half)
        //           ffts_cross_core_sync(PIPE_MTE2, FLAG_C2V)  ← after TLOAD
        // pushpop:
        TPOP<V2CPipe, TileL1, TileSplitAxis::TILE_UP_DOWN>(pipe, ws_l1);
        // └─ internally: wait, TLOAD(GlobalTensor<half>, ws_l1), free-space notify.
        //    ws_l1 assigned to V2C_L1_BASE rotation (slot 0 or 1).
        //    Rotating L1 addresses avoid read-after-write: no barrier needed here.
    }

#endif  // __DAV_C220_CUBE__

    // ── Vec: load A+D, add, push to FIFO ──────────────────────────────────────
#if defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    for (int32_t r = 0; r < num_iters; ++r) {
        const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        HalfTileGlobal a_global(A + row_v * TILE_SIZE);
        TLOAD(a_ub, a_global);

        HalfTileGlobal d_global(D + row_v * TILE_SIZE);
        TLOAD(b_ub, d_global);

        SetFlag<PIPE_MTE2, PIPE_V>(0);
        WaitFlag<PIPE_MTE2, PIPE_V>(0);   // MTE2→V: both TLOADs done before TADD

        TADD(a_ub, a_ub, b_ub);
        SetFlag<PIPE_V, PIPE_MTE3>(0);
        WaitFlag<PIPE_V, PIPE_MTE3>(0);   // V→MTE3: TADD done before TPUSH writes to GM

        // ── pushpop replaces raw_flag: ───────────────────────────────────────
        // raw_flag: (if r>0) wait_flag_dev(FLAG_C2V)
        //           pipe_barrier(PIPE_ALL)
        //           TSTORE(ws_half, a_ub)
        //           pipe_barrier(PIPE_ALL)
        //           ffts_cross_core_sync(PIPE_MTE3, FLAG_V2C)
        // pushpop:
        TPUSH<V2CPipe, TileVecUB, TileSplitAxis::TILE_UP_DOWN>(pipe, a_ub);
        // └─ internally: (if needed) wait for free space, TSTORE(GlobalTensor<half>, a_ub),
        //    then data-ready signal.  Double-buffer back-pressure is automatic.
    }

#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__

extern "C" __global__ AICORE void stream_v2c_kernel(
    __gm__ uint8_t *A,
    __gm__ uint8_t *D,
    __gm__ uint8_t *fifo_mem,
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    run_stream_v2c(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(D),
        fifo_mem,
        num_iters, ffts_addr);
}

extern "C" void call_stream_v2c(uint32_t block_dim, void *stream,
                                 uint8_t *A, uint8_t *D,
                                 uint8_t *fifo_mem, int32_t num_iters)
{
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    stream_v2c_kernel<<<block_dim, nullptr, stream>>>(A, D, fifo_mem, num_iters, ffts_addr);
}
