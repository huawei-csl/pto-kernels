// =============================================================================
// add_matmul_v2c.cpp  —  C = (A + B) @ D   [pushpop variant]
//
// ── API variant: TileData TPUSH / TPOP ──────────────────────────────────────
//
//  raw_flag equivalent               │  pushpop (this file)
//  ──────────────────────────────────┼─────────────────────────────────────────
//  TSTORE(ws_half, a_ub)             │  TPUSH<V2CPipe, VecTile, UP_DOWN>(pipe, a_ub)
//  pipe_barrier(PIPE_ALL)            │    └─ TSTORE(GlobalTensor<half>, a_ub) internally
//  SetCrossFlag<MTE3>(FLAG_V2C)      │       then data-ready signal
//  ──────────────────────────────────┼─────────────────────────────────────────
//  WaitCrossFlag(FLAG_V2C)           │  TPOP<V2CPipe, TileL1, UP_DOWN>(pipe, ab_l1)
//  TLOAD(ab_l1, ws_half)             │    └─ waits, TLOAD(GlobalTensor<half>, ab_l1)
//  SetCrossFlag<MTE2>(FLAG_C2V)      │       free-space notify
//
// VecTile::DType = half → same slot size as raw_flag (T²×sizeof(half) = 32 KB).
// All data types identical to raw_flag.
//
// NOTE: TileData TPUSH/TPOP with TILE_UP_DOWN and 2 Vec sub-blocks shares
//       pipe.prod.tileIndex between sub-blocks, causing tileIndex to advance
//       by 2 per logical round (not 1).  This de-syncs producer/consumer slot
//       indices for num_rounds > 1.  The test is scoped to num_rounds=1 where
//       the API behaves correctly (matching the pto-isa-master unit tests).
//       For multi-round workloads, use the gm_pipe variant.
//
// Python: all float16.  Reference: (A + B) @ D
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

constexpr uint32_t L1_D_OFFSET  = 0u;
constexpr uint32_t L1_AB_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);  // 32 KB
constexpr uint32_t L0_OFFSET    = 0u;
constexpr uint32_t UB_A_OFFSET  = 0u;
constexpr uint32_t UB_B_OFFSET  = HALF_TILE * TILE_SIZE * sizeof(half);  // 16 KB

// FIFO_DEPTH=2 for double-buffered overlap.
constexpr uint32_t FIFO_DEPTH     = 2u;
constexpr uint32_t V2C_SLOT_SIZE  = TILE_SIZE * TILE_SIZE * sizeof(half); // 32 KB
constexpr uint32_t V2C_FIFO_BYTES = FIFO_DEPTH * V2C_SLOT_SIZE;           // 64 KB/core
constexpr uint32_t V2C_L1_BASE    = L1_AB_OFFSET;

using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;
using TileL0A = TileLeft<half,  TILE_SIZE, TILE_SIZE>;
using TileL0B = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C = TileAcc<float,  TILE_SIZE, TILE_SIZE>;

using TileVecUB = Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE,
                       BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                       SLayout::NoneBox, 512, PadValue::Null>;

using V2CPipe = TPipe<0, Direction::DIR_V2C, V2C_SLOT_SIZE, FIFO_DEPTH>;

using TileGlobal =
    GlobalTensor<half,
                 TileShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;
using HalfTileGlobal =
    GlobalTensor<half,
                 TileShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }

AICORE void run_add_matmul_v2c(
    __gm__ half    *A, __gm__ half *B, __gm__ half *C, __gm__ half *D,
    __gm__ uint8_t *fifo_mem, int64_t batch, uint64_t ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows  = num_cores * TILE_SIZE;
    const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

    __gm__ void *core_fifo = fifo_mem + cid * V2C_FIFO_BYTES;
    V2CPipe pipe(core_fifo, /*c2v_ub_base=*/0x0, /*v2c_l1_base=*/V2C_L1_BASE);

    TileL1  d_l1, ab_l1;
    TASSIGN(d_l1, L1_D_OFFSET);
    TileL0A ab_l0;  TileL0B d_l0;  TileL0C c_l0;
    TASSIGN(ab_l0, L0_OFFSET);
    TASSIGN(d_l0,  L0_OFFSET);
    TASSIGN(c_l0,  L0_OFFSET);
    TileVecUB a_ub, b_ub;
    TASSIGN(a_ub, UB_A_OFFSET);
    TASSIGN(b_ub, UB_B_OFFSET);

    // ── Cube: load constant D, TPOP A+B from FIFO, GEMM, store C ──────────────
#if defined(__DAV_C220_CUBE__)

    TileGlobal d_global(D);
    TLOAD(d_l1, d_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(d_l0, d_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_c = r * wave_rows + cid * TILE_SIZE;

        TPOP<V2CPipe, TileL1, TileSplitAxis::TILE_UP_DOWN>(pipe, ab_l1);
        // └─ internally: wait for both Vec sub-blocks' data-ready signals,
        //    TLOAD(GlobalTensor<half>, ab_l1) to V2C_L1_BASE rotation,
        //    then sends free-space notification back to Vec.

        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);  // MTE2→MTE1: TPOP TLOAD done before TMOV

        TMOV(ab_l0, ab_l1);
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        TMATMUL(c_l0, ab_l0, d_l0);
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);  // M→FIX: c_l0 ready for TSTORE

        TileGlobal c_global(C + row_c * TILE_SIZE);
        TSTORE(c_global, c_l0);
        // Next iteration starts with TPOP (cross-core wait);
        // c_global doesn't alias ab_l1/c_l0 — no barrier needed after TSTORE.
    }

#endif  // __DAV_C220_CUBE__

    // ── Vec: load A+B, TPUSH to FIFO ─────────────────────────────────────────
#if defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        HalfTileGlobal a_global(A + row_v * TILE_SIZE);
            TLOAD(a_ub, a_global);
            HalfTileGlobal b_global(B + row_v * TILE_SIZE);
            TLOAD(b_ub, b_global);
            pipe_barrier(PIPE_ALL);  // MTE2→V: both TLOADs done before TADD

            TADD(a_ub, a_ub, b_ub);
            pipe_barrier(PIPE_ALL);  // V→MTE3: TADD done before TPUSH writes to GM

        TPUSH<V2CPipe, TileVecUB, TileSplitAxis::TILE_UP_DOWN>(pipe, a_ub);
        // └─ waits for free space (pipe.prod.allocate = wait_flag_dev),
        //    TSTORE(GlobalTensor<half>, a_ub) to current FIFO slot,
        //    data-ready signal (both sub-blocks together unblock Cube).
    }

#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__

extern "C" __global__ AICORE void add_matmul_v2c_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *B, __gm__ uint8_t *C,
    __gm__ uint8_t *D, __gm__ uint8_t *fifo_mem,
    int64_t batch, uint64_t ffts_addr)
{
    run_add_matmul_v2c(
        reinterpret_cast<__gm__ half *>(A), reinterpret_cast<__gm__ half *>(B),
        reinterpret_cast<__gm__ half *>(C), reinterpret_cast<__gm__ half *>(D),
        fifo_mem, batch, ffts_addr);
}

extern "C" void call(uint32_t block_dim, void *stream,
                     uint8_t *A, uint8_t *B, uint8_t *C,
                     uint8_t *D, uint8_t *fifo_mem, int64_t batch)
{
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    add_matmul_v2c_kernel<<<block_dim, nullptr, stream>>>(
        A, B, C, D, fifo_mem, batch, ffts_addr);
}
