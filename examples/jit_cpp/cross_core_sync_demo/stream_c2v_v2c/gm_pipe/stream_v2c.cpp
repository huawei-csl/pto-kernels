// =============================================================================
// stream_v2c.cpp — Vector→Cube bandwidth microbenchmark  [gm_pipe variant]
//
// Same algorithm as raw_flag/stream_v2c.cpp.
//
// ── API variant: GlobalData TALLOC + TSTORE + TPUSH / TPOP + TLOAD + TFREE ──
//
//  raw_flag equivalent               │  gm_pipe (this file)
//  ──────────────────────────────────┼─────────────────────────────────────────
//  TSTORE(ws_half, a_ub)             │  TALLOC<V2CPipe, HalfSlot, UP_DOWN>(pipe, slot)
//  SetCrossFlag<MTE3>(FLAG_V2C)      │  TSTORE(slot, a_ub)  ← explicit half store
//                                    │  TPUSH<V2CPipe, HalfSlot, UP_DOWN>(pipe, slot)
//                                    │    (TSTORE and TPUSH are both MTE3-pipe; ordered)
//  ──────────────────────────────────┼─────────────────────────────────────────
//  WaitCrossFlag(FLAG_V2C)           │  TPOP<V2CPipe, FullSlot, NO_SPLIT>(pipe, slot)
//  TLOAD(ws_l1, ws_half)             │  TLOAD(ws_l1, slot)  ← explicit L1 load
//  SetCrossFlag<MTE2>(FLAG_C2V)      │  TFREE<V2CPipe, FullSlot, NO_SPLIT>(pipe, slot)
//                                    │    (TLOAD and TFREE are both MTE2-pipe; ordered)
//
// Vec TALLOC TILE_UP_DOWN: vid=0→slot_base+0, vid=1→slot_base+T/2×T×sizeof(half).
// Cube TPOP TILE_NO_SPLIT: Cube always gets the full T×T slot address.
// Slot size = T²×sizeof(half) = 32 KB — identical to raw_flag.
//
// Uses `if constexpr` for Cube/Vec branching (required for GlobalData dispatch).
// =============================================================================

#define MEMORY_BASE
#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>

using namespace pto;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif
#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

#define TILE_SIZE 128
#define HALF_TILE  64
#define VEC_NUM     2

#ifdef __CCE_AICORE__

constexpr uint32_t L1_WS_OFFSET = 0u;
constexpr uint32_t UB_A_OFFSET  = 0u;
constexpr uint32_t UB_B_OFFSET  = HALF_TILE * TILE_SIZE * sizeof(half);  // 16 KB

constexpr uint32_t FIFO_DEPTH     = 2u;
constexpr uint32_t V2C_SLOT_SIZE  = TILE_SIZE * TILE_SIZE * sizeof(half); // 32 KB
constexpr uint32_t V2C_FIFO_BYTES = FIFO_DEPTH * V2C_SLOT_SIZE;           // 64 KB/core

using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;

using TileVecUB = Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE,
                       BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                       SLayout::NoneBox, 512, PadValue::Null>;

// Use FlagID=2 (FFTS flags 2 and 3) to avoid collision with stream_c2v's
// C2VPipe = TPipe<0, DIR_C2V> which occupies flags 0 (push) and 1 (free).
using V2CPipe = TPipe<2, Direction::DIR_V2C, V2C_SLOT_SIZE, FIFO_DEPTH>;

// Vec writes T/2 rows per sub-block (TILE_UP_DOWN).
using HalfSlot = GlobalTensor<half, pto::Shape<1, 1, 1, HALF_TILE, TILE_SIZE>,
                               pto::Stride<1, 1, 1, TILE_SIZE, 1>>;
// Cube reads the full T×T slot (TILE_NO_SPLIT).
using FullSlot = GlobalTensor<half, pto::Shape<1, 1, 1, TILE_SIZE, TILE_SIZE>,
                               pto::Stride<1, 1, 1, TILE_SIZE, 1>>;

using HalfTileGlobal =
    GlobalTensor<half,
                 TileShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }

AICORE void run_stream_v2c(
    __gm__ half    *A,
    __gm__ half    *D,
    __gm__ uint8_t *fifo_mem,
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows = num_cores * TILE_SIZE;

    __gm__ void *core_fifo = fifo_mem + cid * V2C_FIFO_BYTES;
    V2CPipe pipe(core_fifo, /*c2v_ub_base=*/0x0, /*v2c_l1_base=*/0x0);

    TileVecUB a_ub, b_ub;
    TASSIGN(a_ub, UB_A_OFFSET);
    TASSIGN(b_ub, UB_B_OFFSET);

    TileL1 ws_l1;
    TASSIGN(ws_l1, L1_WS_OFFSET);

    // ── Cube: pop half tiles from FIFO, discard (bandwidth test) ─────────────
    if constexpr (DAV_CUBE) {
        FullSlot pop_slot;
        for (int32_t r = 0; r < num_iters; ++r) {
            TPOP<V2CPipe, FullSlot, TileSplitAxis::TILE_NO_SPLIT>(pipe, pop_slot);
            //   Waits for data-ready; assigns the full T×T slot address.
            TLOAD(ws_l1, pop_slot);
            //   Explicit TLOAD: TileL1<half> ← GlobalTensor<half>
            pipe_barrier(PIPE_ALL);  // MTE2: wait for DMA to complete before freeing slot
            //   TFREE fires from MTE2 pipe after TLOAD DMA completes.
            TFREE<V2CPipe, FullSlot, TileSplitAxis::TILE_NO_SPLIT>(pipe, pop_slot);
            //   Emits free-space notification to Vec (conditional on SyncPeriod)
        }
    }

    // ── Vec: load A+D, add, write to FIFO slot ────────────────────────────────
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask(-1, -1);

        HalfSlot push_slot;
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
            WaitFlag<PIPE_V, PIPE_MTE3>(0);   // V→MTE3: TADD done before TSTORE

            // ── gm_pipe replaces raw_flag: ───────────────────────────────────
            // raw_flag: (if r>0) wait_flag_dev(FLAG_C2V), TSTORE(ws_half, a_ub),
            //           ffts_cross_core_sync(PIPE_MTE3, FLAG_V2C)
            // gm_pipe:
            TALLOC<V2CPipe, HalfSlot, TileSplitAxis::TILE_UP_DOWN>(pipe, push_slot);
            //   vid=0→slot_base+0, vid=1→slot_base+T/2×T×sizeof(half)
            TSTORE(push_slot, a_ub);
            //   Explicit TSTORE: GlobalTensor<half> ← TileVecUB<half> (same dtype)
            pipe_barrier(PIPE_ALL);  // MTE3: wait for DMA to complete before TPUSH signals Cube
            TPUSH<V2CPipe, HalfSlot, TileSplitAxis::TILE_UP_DOWN>(pipe, push_slot);
            //   Sync-only: emits data-ready signal
        }
    }
}

#endif  // __CCE_AICORE__

extern "C" __global__ AICORE void stream_v2c_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *D,
    __gm__ uint8_t *fifo_mem, int32_t num_iters, uint64_t ffts_addr)
{
    run_stream_v2c(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(D),
        fifo_mem, num_iters, ffts_addr);
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
