// =============================================================================
// matmul_add_c2v.cpp  —  C = A @ B + D   [pushpop variant]
//
// ── API variant: TileData TPUSH / TPOP ──────────────────────────────────────
//
//  raw_flag equivalent          │  pushpop (this file)
//  ─────────────────────────────┼─────────────────────────────────────────────
//  TSTORE(ws_half, c_l0)        │  TPUSH<C2VPipe, TileL0C, UP_DOWN>(pipe, c_l0)
//  pipe_barrier(PIPE_ALL)       │    └─ stores AccTile::DType=float into slot
//  SetCrossFlag<FIX>(FLAG_C2V)  │
//  ─────────────────────────────┼─────────────────────────────────────────────
//  WaitCrossFlag(FLAG_C2V)      │  TPOP<C2VPipe, VecTileFloat, UP_DOWN>(pipe, c_ub_float)
//  TLOAD(c_ub_half, ws_half)    │    └─ receives float tile  ← dtype differs from raw_flag
//  SetCrossFlag<MTE3>(FLAG_V2C) │
//
// Dtype note: TPUSH stores AccTile::DType=float32 into the slot (no fp32→fp16
// in the TileData path).  Vec receives float32.  D must be float32 to match.
// Output C is also float32.  Use gm_pipe variant for half-precision output.
//
// Python: A,B: float16;  D,C: float32.
// Reference: (A @ B) + D  computed in float32.
//
// Slot type: float32, 64 KB/slot  (vs 32 KB half-slot in raw_flag / gm_pipe).
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

// ── On-chip buffer offsets (bytes) ────────────────────────────────────────────
constexpr uint32_t L1_B_OFFSET = 0u;
constexpr uint32_t L1_A_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);   // 32 KB
constexpr uint32_t L0_OFFSET   = 0u;
constexpr uint32_t UB_D_OFFSET = 0u;  // d_ub_float: 32 KB (float, HALF_TILE rows)

// ── FIFO configuration ────────────────────────────────────────────────────────
constexpr uint32_t FIFO_DEPTH      = 2u;
constexpr uint32_t C2V_SLOT_SIZE   = TILE_SIZE * TILE_SIZE * sizeof(float); // 64 KB
constexpr uint32_t C2V_FIFO_BYTES  = FIFO_DEPTH * C2V_SLOT_SIZE;            // 128 KB/core
constexpr uint32_t C2V_UB_BASE     = 0x20000;  // 128 KB offset: after d_ub (32 KB)
// TPOP assigns c_ub_float to C2V_UB_BASE + rotation * HALF_TILE*TILE_SIZE*sizeof(float)
// Slot 0: UB[0x20000], Slot 1: UB[0x28000] — within the 192 KB UB budget.

// ── Tile types ────────────────────────────────────────────────────────────────
using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;

using TileL0A = TileLeft<half,  TILE_SIZE, TILE_SIZE>;
using TileL0B = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C = TileAcc<float,  TILE_SIZE, TILE_SIZE>;

// Vec receives float (matching the float slot written by TPUSH).
using VecTileFloat = Tile<TileType::Vec, float, HALF_TILE, TILE_SIZE,
                          BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                          SLayout::NoneBox, 512, PadValue::Null>;
// D is also float so TADD types match.
using VecTileFloatD = Tile<TileType::Vec, float, HALF_TILE, TILE_SIZE,
                           BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                           SLayout::NoneBox, 512, PadValue::Null>;

using C2VPipe = TPipe<0, Direction::DIR_C2V, C2V_SLOT_SIZE, FIFO_DEPTH>;

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
// D input and C output are declared as half in GM but loaded/stored as float
// via float GlobalTensors with the same byte offset.
using FloatHalfTileGlobal =
    GlobalTensor<float,
                 TileShape2D<float, HALF_TILE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<float, HALF_TILE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }

// ─────────────────────────────────────────────────────────────────────────────
AICORE void run_matmul_add_c2v(
    __gm__ half    *A,
    __gm__ half    *B,
    __gm__ float   *C,        // float32 — matches VecTileFloat output dtype
    __gm__ float   *D,        // float32 — must match VecTileFloat dtype
    __gm__ uint8_t *fifo_mem,
    int64_t         batch,
    uint64_t        ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows  = num_cores * TILE_SIZE;
    const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

    __gm__ void *core_fifo = fifo_mem + cid * C2V_FIFO_BYTES;
    C2VPipe pipe(core_fifo, /*c2v_ub_base=*/C2V_UB_BASE, /*v2c_l1_base=*/0x0);

    TileL1  b_l1, a_l1;
    TASSIGN(b_l1, L1_B_OFFSET);
    TASSIGN(a_l1, L1_A_OFFSET);

    TileL0A a_l0;  TileL0B b_l0;  TileL0C c_l0;
    TASSIGN(a_l0, L0_OFFSET);
    TASSIGN(b_l0, L0_OFFSET);
    TASSIGN(c_l0, L0_OFFSET);

    VecTileFloatD d_ub;
    TASSIGN(d_ub, UB_D_OFFSET);
    VecTileFloat c_ub_float;  // TPOP assigns this internally via C2V_UB_BASE

    // ── Cube: GEMM per round, push result to FIFO ─────────────────────────────
#if defined(__DAV_C220_CUBE__)

    TileGlobal b_global(B);
    TLOAD(b_l1, b_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(b_l0, b_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

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
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);   // M→FIX: c_l0 ready before TPUSH stores it

        TPUSH<C2VPipe, TileL0C, TileSplitAxis::TILE_UP_DOWN>(pipe, c_l0);
        // └─ internally: TSTORE(GlobalTensor<float>, c_l0) + data-ready signal
    }

#endif  // __DAV_C220_CUBE__

    // ── Vec: pop GEMM result (float), add D (float), store C (half) ───────────
#if defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        TPOP<C2VPipe, VecTileFloat, TileSplitAxis::TILE_UP_DOWN>(pipe, c_ub_float);
        // └─ wait + TLOAD(GlobalTensor<float>, c_ub_float) + free notify

        // Load D as float32 to match c_ub_float dtype for TADD.
        FloatHalfTileGlobal d_global(D + row_v * TILE_SIZE);
        TLOAD(d_ub, d_global);

        pipe_barrier(PIPE_ALL);  // MTE2: both TPOP-TLOAD and D-TLOAD complete

        TADD(c_ub_float, c_ub_float, d_ub);
        pipe_barrier(PIPE_ALL);  // V→MTE3: TADD done before TSTORE

        // TSTORE float → float: Vec UB tile → GM C (float32 output).
        // Note: Vec→GM TSTORE requires matching dtypes (no implicit conversion
        // for the MTE3 pipe).  Float32 C output is the natural result of float
        // FIFO slots in this pushpop variant.  See gm_pipe for half16 output.
        using FloatTileGlobal =
            GlobalTensor<float,
                         TileShape2D<float, HALF_TILE, TILE_SIZE, Layout::ND>,
                         BaseShape2D<float, HALF_TILE, TILE_SIZE, Layout::ND>,
                         Layout::ND>;
        FloatTileGlobal c_out(C + row_v * TILE_SIZE);
        TSTORE(c_out, c_ub_float);
        pipe_barrier(PIPE_ALL);  // MTE3: TSTORE complete before next round
    }

#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__

extern "C" __global__ AICORE void matmul_add_c2v_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *B, __gm__ uint8_t *C,
    __gm__ uint8_t *D, __gm__ uint8_t *fifo_mem,
    int64_t batch, uint64_t ffts_addr)
{
    run_matmul_add_c2v(
        reinterpret_cast<__gm__ half    *>(A),
        reinterpret_cast<__gm__ half    *>(B),
        reinterpret_cast<__gm__ float   *>(C),
        reinterpret_cast<__gm__ float   *>(D),
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
