// =============================================================================
// stream_c2v.cpp — Cube→Vector bandwidth microbenchmark  [gm_pipe variant]
//
// Same algorithm as raw_flag/stream_c2v.cpp.
//
// ── API variant: GlobalData TALLOC + TSTORE + TPUSH / TPOP + TLOAD + TFREE ──
//
//  raw_flag equivalent          │  gm_pipe (this file)
//  ─────────────────────────────┼─────────────────────────────────────────────
//  TSTORE(ws_half, c_l0)        │  TALLOC<C2VPipe, SlotFull, UP_DOWN>(pipe, slot)
//  pipe_barrier(PIPE_ALL)       │  TSTORE(slot, c_l0)   ← explicit fp32→fp16
//  SetCrossFlag<FIX>(FLAG_C2V)  │  pipe_barrier(PIPE_ALL)
//                               │  TPUSH<C2VPipe, SlotFull, UP_DOWN>(pipe, slot)
//  ─────────────────────────────┼─────────────────────────────────────────────
//  WaitCrossFlag(FLAG_C2V)      │  TPOP<C2VPipe, PopHalf, UP_DOWN>(pipe, pop)
//  TLOAD(c_ub, ws_half)         │    └─ sync-only: assigns slot address to pop
//  SetCrossFlag<MTE3>(FLAG_V2C) │  TLOAD(c_ub, pop)     ← explicit half load
//                               │  pipe_barrier(PIPE_ALL)
//                               │  TFREE<C2VPipe, PopHalf, UP_DOWN>(pipe, pop)
//
// Key property:
//   TSTORE(slot_half, c_l0) — GlobalTensor<half> ← TileAcc<float>:
//   hardware FIX unit performs fp32→fp16.  SlotSize = T²×sizeof(half) = 32 KB.
//   Same slot size and dtype as raw_flag; direct bandwidth comparison is valid.
//
// Slot type summary:
//   raw_flag : half slot, 32 KB/slot  →  workspace half tensor
//   pushpop  : float slot, 64 KB/slot
//   gm_pipe  : half slot, 32 KB/slot  →  FIFO memory half tensor  ← this file
//
// NOTE: Uses `if constexpr` (not #if/#endif) for Cube/Vec branching, matching
//   the style of the pto-isa unit tests.  This avoids overload-resolution
//   issues in the GlobalData TPOP/TFREE template dispatch.
//
// Memory budget (per core):
//   L1  (512 KB): b_l1 32 KB + a_l1 32 KB = 64 KB
//   L0A ( 64 KB): a_l0 32 KB
//   L0B ( 64 KB): b_l0 32 KB
//   L0C (128 KB): c_l0 64 KB
//   UB  (192 KB): c_ub 16 KB
//   FIFO GM    : FIFO_DEPTH × SLOT_SIZE = 2 × 32 KB = 64 KB per core
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

constexpr uint32_t L1_B_OFFSET  = 0u;
constexpr uint32_t L1_A_OFFSET  = TILE_SIZE * TILE_SIZE * sizeof(half);  // 32 KB
constexpr uint32_t L0_OFFSET    = 0u;
constexpr uint32_t UB_C_OFFSET  = 0u;

constexpr uint32_t FIFO_DEPTH     = 2u;
constexpr uint32_t C2V_SLOT_SIZE  = TILE_SIZE * TILE_SIZE * sizeof(half); // 32 KB
constexpr uint32_t C2V_FIFO_BYTES = FIFO_DEPTH * C2V_SLOT_SIZE;           // 64 KB/core

using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;
using TileL0A = TileLeft<half,  TILE_SIZE, TILE_SIZE>;
using TileL0B = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C = TileAcc<float,  TILE_SIZE, TILE_SIZE>;

using TileVecUB = Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE,
                       BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                       SLayout::NoneBox, 512, PadValue::Null>;

using C2VPipe = TPipe<0, Direction::DIR_C2V, C2V_SLOT_SIZE, FIFO_DEPTH>;

// Cube writes the full T×T slot; Vec reads its T/2-row subslot.
using SlotFull = GlobalTensor<half, pto::Shape<1, 1, 1, TILE_SIZE, TILE_SIZE>,
                               pto::Stride<1, 1, 1, TILE_SIZE, 1>>;
using PopHalf  = GlobalTensor<half, pto::Shape<1, 1, 1, HALF_TILE, TILE_SIZE>,
                               pto::Stride<1, 1, 1, TILE_SIZE, 1>>;

using TileGlobal =
    GlobalTensor<half,
                 TileShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }

AICORE void run_stream_c2v(
    __gm__ half    *A,
    __gm__ half    *B,
    __gm__ uint8_t *fifo_mem,
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    const int32_t cid = static_cast<int32_t>(get_block_idx());
    set_ffts_base_addr(ffts_addr);

    __gm__ void *core_fifo = fifo_mem + cid * C2V_FIFO_BYTES;
    C2VPipe pipe(core_fifo, /*c2v_ub_base=*/0x0, /*v2c_l1_base=*/0x0);

    TileL1  b_l1, a_l1;
    TASSIGN(b_l1, L1_B_OFFSET);
    TASSIGN(a_l1, L1_A_OFFSET);

    TileL0A a_l0;  TileL0B b_l0;  TileL0C c_l0;
    TASSIGN(a_l0, L0_OFFSET);
    TASSIGN(b_l0, L0_OFFSET);
    TASSIGN(c_l0, L0_OFFSET);

    TileVecUB c_ub;
    TASSIGN(c_ub, UB_C_OFFSET);

    // ── Cube: one-time GEMM, then push loop ───────────────────────────────────
    if constexpr (DAV_CUBE) {
        TileGlobal b_global(B);
        TLOAD(b_l1, b_global);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TMOV(b_l0, b_l1);
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        TileGlobal a_global(A + cid * TILE_SIZE * TILE_SIZE);
        TLOAD(a_l1, a_global);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TMOV(a_l0, a_l1);
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        TMATMUL(c_l0, a_l0, b_l0);
        pipe_barrier(PIPE_ALL);  // M→FIX: c_l0 ready for TSTORE

        SlotFull push_slot;
        for (int32_t r = 0; r < num_iters; ++r) {
            // ── gm_pipe replaces raw_flag: ───────────────────────────────────
            // raw_flag: TSTORE(ws_half, c_l0)   (fp32→fp16 via FIX)
            //           pipe_barrier(PIPE_ALL)
            //           ffts_cross_core_sync(PIPE_FIX, FLAG_C2V)
            // gm_pipe:
            TALLOC<C2VPipe, SlotFull, TileSplitAxis::TILE_UP_DOWN>(pipe, push_slot);
            TSTORE(push_slot, c_l0);
            //   GlobalTensor<half> ← TileAcc<float>: fp32→fp16 (hardware FIX unit)
            pipe_barrier(PIPE_ALL);  // FIX: TSTORE complete before TPUSH signals Vec
            TPUSH<C2VPipe, SlotFull, TileSplitAxis::TILE_UP_DOWN>(pipe, push_slot);
            //   Sync-only: emits data-ready signal (no internal TSTORE)
        }
    }

    // ── Vec: pop half tiles from FIFO, discard (bandwidth test) ──────────────
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask(-1, -1);

        PopHalf pop_slot;
        for (int32_t r = 0; r < num_iters; ++r) {
            // ── gm_pipe replaces raw_flag: ───────────────────────────────────
            // raw_flag: wait_flag_dev(FLAG_C2V)
            //           pipe_barrier(PIPE_ALL)
            //           TLOAD(c_ub, ws_half)
            //           pipe_barrier(PIPE_ALL)
            //           ffts_cross_core_sync(PIPE_MTE3, FLAG_V2C)
            // gm_pipe:
            TPOP<C2VPipe, PopHalf, TileSplitAxis::TILE_UP_DOWN>(pipe, pop_slot);
            //   Waits for data-ready; assigns this sub-block's T/2-row slice address.
            TLOAD(c_ub, pop_slot);
            //   Explicit TLOAD: TileVecUB<half> ← GlobalTensor<half>
            pipe_barrier(PIPE_ALL);  // MTE2: TLOAD complete before slot is released
            TFREE<C2VPipe, PopHalf, TileSplitAxis::TILE_UP_DOWN>(pipe, pop_slot);
            //   Emits free-space notification to Cube (conditional on SyncPeriod)
        }
    }
}

#endif  // __CCE_AICORE__

extern "C" __global__ AICORE void stream_c2v_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *B,
    __gm__ uint8_t *fifo_mem, int32_t num_iters, uint64_t ffts_addr)
{
    run_stream_c2v(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(B),
        fifo_mem, num_iters, ffts_addr);
}

extern "C" void call_stream_c2v(uint32_t block_dim, void *stream,
                                 uint8_t *A, uint8_t *B,
                                 uint8_t *fifo_mem, int32_t num_iters)
{
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    stream_c2v_kernel<<<block_dim, nullptr, stream>>>(A, B, fifo_mem, num_iters, ffts_addr);
}
