// =============================================================================
// matmul_add_c2v.cpp  —  C = A @ B + D   [gm_pipe variant]
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
//  TLOAD(c_ub, ws_half)         │  TLOAD(c_ub, pop)
//  SetCrossFlag<MTE3>(FLAG_V2C) │  pipe_barrier(PIPE_ALL)
//                               │  TFREE<C2VPipe, PopHalf, UP_DOWN>(pipe, pop)
//
// Requires: pto-isa-master headers (GlobalData TPOP/TALLOC/TFREE APIs).
// Half slot (32 KB/slot) — same size as raw_flag; direct bandwidth comparison valid.
// Python: all float16.  Reference: A @ B + D
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

constexpr uint32_t L1_B_OFFSET = 0u;
constexpr uint32_t L1_A_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);   // 32 KB
constexpr uint32_t L0_OFFSET   = 0u;
constexpr uint32_t UB_C_OFFSET = 0u;   // c_ub: 16 KB
constexpr uint32_t UB_D_OFFSET = HALF_TILE * TILE_SIZE * sizeof(half);   // 16 KB

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

using SlotFull = GlobalTensor<half, pto::Shape<1, 1, 1, TILE_SIZE, TILE_SIZE>,
                               pto::Stride<1, 1, 1, TILE_SIZE, 1>>;
using PopHalf  = GlobalTensor<half, pto::Shape<1, 1, 1, HALF_TILE, TILE_SIZE>,
                               pto::Stride<1, 1, 1, TILE_SIZE, 1>>;

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

AICORE void run_matmul_add_c2v(
    __gm__ half    *A, __gm__ half *B, __gm__ half *C, __gm__ half *D,
    __gm__ uint8_t *fifo_mem, int64_t batch, uint64_t ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows  = num_cores * TILE_SIZE;
    const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

    __gm__ void *core_fifo = fifo_mem + cid * C2V_FIFO_BYTES;
    C2VPipe pipe(core_fifo, /*c2v_ub_base=*/0x0, /*v2c_l1_base=*/0x0);

    TileL1  b_l1, a_l1;
    TASSIGN(b_l1, L1_B_OFFSET);
    TASSIGN(a_l1, L1_A_OFFSET);
    TileL0A a_l0;  TileL0B b_l0;  TileL0C c_l0;
    TASSIGN(a_l0, L0_OFFSET);
    TASSIGN(b_l0, L0_OFFSET);
    TASSIGN(c_l0, L0_OFFSET);
    TileVecUB c_ub, d_ub;
    TASSIGN(c_ub, UB_C_OFFSET);
    TASSIGN(d_ub, UB_D_OFFSET);

    if constexpr (DAV_CUBE) {
        TileGlobal b_global(B);
        TLOAD(b_l1, b_global);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TMOV(b_l0, b_l1);
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        SlotFull push_slot;
        for (int32_t r = 0; r < num_rounds; ++r) {
            const int32_t row_c = r * wave_rows + cid * TILE_SIZE;

            TileGlobal a_global(A + row_c * TILE_SIZE);
            TLOAD(a_l1, a_global);
            SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
            WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
            TMOV(a_l0, a_l1);
            SetFlag<PIPE_MTE1, PIPE_M>(0);
            WaitFlag<PIPE_MTE1, PIPE_M>(0);

            TMATMUL(c_l0, a_l0, b_l0);
            pipe_barrier(PIPE_ALL);

            TALLOC<C2VPipe, SlotFull, TileSplitAxis::TILE_UP_DOWN>(pipe, push_slot);
            TSTORE(push_slot, c_l0);
            //   AccTile<float> → GlobalTensor<half>: fp32→fp16 via hardware FIX
            pipe_barrier(PIPE_ALL);
            TPUSH<C2VPipe, SlotFull, TileSplitAxis::TILE_UP_DOWN>(pipe, push_slot);
        }
    }

    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask(-1, -1);

        PopHalf pop_slot;
        for (int32_t r = 0; r < num_rounds; ++r) {
            const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

            TPOP<C2VPipe, PopHalf, TileSplitAxis::TILE_UP_DOWN>(pipe, pop_slot);
            TLOAD(c_ub, pop_slot);

            HalfTileGlobal d_global(D + row_v * TILE_SIZE);
            TLOAD(d_ub, d_global);

            pipe_barrier(PIPE_ALL);

            TADD(c_ub, c_ub, d_ub);
            pipe_barrier(PIPE_ALL);

            TFREE<C2VPipe, PopHalf, TileSplitAxis::TILE_UP_DOWN>(pipe, pop_slot);

            HalfTileGlobal c_out(C + row_v * TILE_SIZE);
            TSTORE(c_out, c_ub);
            pipe_barrier(PIPE_ALL);
        }
    }
}

#endif  // __CCE_AICORE__

extern "C" __global__ AICORE void matmul_add_c2v_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *B, __gm__ uint8_t *C,
    __gm__ uint8_t *D, __gm__ uint8_t *fifo_mem,
    int64_t batch, uint64_t ffts_addr)
{
    run_matmul_add_c2v(
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
    matmul_add_c2v_kernel<<<block_dim, nullptr, stream>>>(
        A, B, C, D, fifo_mem, batch, ffts_addr);
}
