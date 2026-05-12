// =============================================================================
// stream_c2v.cpp — Cube→Vector bandwidth microbenchmark  [pushpop variant]
//
// Same algorithm as raw_flag/stream_c2v.cpp:
//   • Cube runs one GEMM to fill c_l0, then spills it num_iters times.
//   • Vec pops each tile from the FIFO and discards it (pure bandwidth test).
//
// ── API variant: TileData TPUSH / TPOP ──────────────────────────────────────
//
//  raw_flag equivalent          │  pushpop (this file)
//  ─────────────────────────────┼────────────────────────────────────────────
//  TSTORE(ws_half, c_l0)        │  TPUSH<C2VPipe, TileL0C, UP_DOWN>(pipe, c_l0)
//  pipe_barrier(PIPE_ALL)       │    └─ internally: TSTORE(GlobalTensor<float>, c_l0)
//  SetCrossFlag<FIX>(FLAG_C2V)  │       then data-ready signal
//  ─────────────────────────────┼────────────────────────────────────────────
//  WaitCrossFlag(FLAG_C2V)      │  TPOP<C2VPipe, VecTile<float>, UP_DOWN>(pipe, c_ub)
//  TLOAD(c_ub, ws_half)         │    └─ internally: wait, TLOAD(GlobalTensor<float>, c_ub)
//  SetCrossFlag<MTE3>(FLAG_V2C) │       then free-space notify
//
// Key difference from raw_flag:
//   TPUSH stores AccTile::DType = float32 into the FIFO slot (no fp32→fp16).
//   SlotSize = T²×sizeof(float) = 64 KB  (vs 32 KB half-slot in raw_flag).
//   Vec receives a float VecTile.  TPipe manages double-buffering automatically.
//
// Slot type summary:
//   raw_flag : half slot,  32 KB/slot  →  workspace half tensor
//   pushpop  : float slot, 64 KB/slot  →  FIFO memory float tensor  ← this file
//   gm_pipe  : half slot,  32 KB/slot  →  FIFO memory half tensor
//
// Memory budget (per core):
//   L1  (512 KB): b_l1 32 KB + a_l1 32 KB = 64 KB  (initial GEMM setup)
//   L0A ( 64 KB): a_l0 32 KB
//   L0B ( 64 KB): b_l0 32 KB
//   L0C (128 KB): c_l0 64 KB  (spilled every iteration, never overwritten)
//   UB  (192 KB): c_ub_float — managed by TPOP via C2V_CONSUMER_BUF
//                 Slot 0 at 0x00000 (32 KB), Slot 1 at 0x08000 (32 KB) = 64 KB used
//   FIFO GM     : FIFO_DEPTH × SLOT_SIZE = 2 × 64 KB = 128 KB per core
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

// ── On-chip L1/L0 buffer offsets (bytes) ──────────────────────────────────────
constexpr uint32_t L1_B_OFFSET = 0u;                                      // B: 32 KB
constexpr uint32_t L1_A_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);    // A: 32 KB
constexpr uint32_t L0_OFFSET   = 0u;                                      // shared origin

// ── FIFO configuration ────────────────────────────────────────────────────────
// Slot holds AccTile::DType = float32.  Two slots for double-buffered overlap.
constexpr uint32_t FIFO_DEPTH      = 2u;
constexpr uint32_t C2V_SLOT_SIZE   = TILE_SIZE * TILE_SIZE * sizeof(float); // 64 KB
constexpr uint32_t C2V_FIFO_BYTES  = FIFO_DEPTH * C2V_SLOT_SIZE;            // 128 KB/core

// UB base for Vec to receive TPOP'd C2V tiles (managed by TPipe internally).
constexpr uint32_t C2V_UB_BASE     = 0x0;

// ── Tile types ────────────────────────────────────────────────────────────────
using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;

using TileL0A   = TileLeft<half,  TILE_SIZE, TILE_SIZE>;
using TileL0B   = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C   = TileAcc<float,  TILE_SIZE, TILE_SIZE>;

// Vec receives float (matching AccTile::DType stored by TPUSH).
using VecTileFloat = Tile<TileType::Vec, float, HALF_TILE, TILE_SIZE,
                          BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                          SLayout::NoneBox, 512, PadValue::Null>;

// ── FIFO pipe type ────────────────────────────────────────────────────────────
// FlagID=0, DIR_C2V, slot=float64KB, depth=2.
// TPUSH calls TSTORE(GlobalTensor<float>, c_l0) internally — no fp32→fp16.
// TPOP  calls TLOAD(GlobalTensor<float>, c_ub) internally.
using C2VPipe = TPipe<0, Direction::DIR_C2V, C2V_SLOT_SIZE, FIFO_DEPTH>;

// ── Global tensor type for GEMM input ─────────────────────────────────────────
using TileGlobal =
    GlobalTensor<half,
                 TileShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

// ── Intra-pipe sync helpers ────────────────────────────────────────────────────
template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }

// ─────────────────────────────────────────────────────────────────────────────
AICORE void run_stream_c2v(
    __gm__ half    *A,         // [num_cores × T, T]  initial GEMM input
    __gm__ half    *B,         // [T, T]              weight matrix
    __gm__ uint8_t *fifo_mem,  // [num_cores × C2V_FIFO_BYTES]  C2V FIFO buffer
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    const int32_t cid = static_cast<int32_t>(get_block_idx());

    set_ffts_base_addr(ffts_addr);

    // Each core owns its own FIFO region to avoid inter-core aliasing.
    __gm__ void *core_fifo = fifo_mem + cid * C2V_FIFO_BYTES;
    C2VPipe pipe(core_fifo, /*c2v_ub_base=*/C2V_UB_BASE, /*v2c_l1_base=*/0x0);

    // ── Allocate on-chip buffers ───────────────────────────────────────────────
    TileL1  b_l1, a_l1;
    TASSIGN(b_l1, L1_B_OFFSET);
    TASSIGN(a_l1, L1_A_OFFSET);

    TileL0A a_l0;  TileL0B b_l0;  TileL0C c_l0;
    TASSIGN(a_l0, L0_OFFSET);
    TASSIGN(b_l0, L0_OFFSET);
    TASSIGN(c_l0, L0_OFFSET);

    // c_ub_float is assigned internally by TPOP (no manual TASSIGN needed).
    VecTileFloat c_ub_float;

    // ── Cube: one-time GEMM fills c_l0, then push loop ────────────────────────
#if defined(__DAV_C220_CUBE__)

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
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);    // M→FIX: c_l0 ready before TPUSH reads it

    for (int32_t r = 0; r < num_iters; ++r) {
        // ── pushpop replaces raw_flag: ───────────────────────────────────────
        // raw_flag: TSTORE(ws_half, c_l0)  (fp32→fp16)
        //           pipe_barrier(PIPE_ALL)
        //           ffts_cross_core_sync(PIPE_FIX, FLAG_C2V)
        // pushpop:
        TPUSH<C2VPipe, TileL0C, TileSplitAxis::TILE_UP_DOWN>(pipe, c_l0);
        // └─ internally: TSTORE(GlobalTensor<float>, c_l0) + data-ready signal
        //    Note: slot stores float32 (AccTile::DType), NOT half.
    }

#endif  // __DAV_C220_CUBE__

    // ── Vec: pop loop — receives float tiles, discards data ───────────────────
#if defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    for (int32_t r = 0; r < num_iters; ++r) {
        // ── pushpop replaces raw_flag: ───────────────────────────────────────
        // raw_flag: wait_flag_dev(FLAG_C2V)
        //           TLOAD(c_ub_half, ws_half)
        //           ffts_cross_core_sync(PIPE_MTE3, FLAG_V2C)
        // pushpop:
        TPOP<C2VPipe, VecTileFloat, TileSplitAxis::TILE_UP_DOWN>(pipe, c_ub_float);
        // └─ internally: wait, TLOAD(GlobalTensor<float>, c_ub_float),
        //    free-space notify.  c_ub_float assigned to C2V_UB_BASE rotation.
    }

#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__

// ── Kernel entry point ─────────────────────────────────────────────────────────
extern "C" __global__ AICORE void stream_c2v_kernel(
    __gm__ uint8_t *A,
    __gm__ uint8_t *B,
    __gm__ uint8_t *fifo_mem,
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    run_stream_c2v(
        reinterpret_cast<__gm__ half    *>(A),
        reinterpret_cast<__gm__ half    *>(B),
        fifo_mem,
        num_iters, ffts_addr);
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
