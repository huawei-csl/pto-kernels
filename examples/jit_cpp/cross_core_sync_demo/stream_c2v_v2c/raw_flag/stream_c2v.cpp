// =============================================================================
// stream_c2v.cpp — Cube→Vector bandwidth microbenchmark
//
// Measures the sustained throughput of the C2V path:
//   Cube L0C  →  GM workspace  →  Vector UB
//
// Setup (once, outside the timed loop):
//   Cube loads A[cid*T:] → L1 → L0A and B → L1 → L0B, then GEMM to fill c_l0.
//
// Inner loop (num_iters times, this is the timed section):
//   Cube:  if r > 0: WaitCrossFlag(FLAG_V2C)
//          TSTORE c_l0 → workspace[cid*T:]   (FIX pipe, fp32 → fp16)
//          SetCrossFlag<PIPE_FIX>(FLAG_C2V)   → both Vec sub-blocks
//   Vec:   WaitCrossFlag(FLAG_C2V)
//          TLOAD workspace[ws_row:] → c_ub    (MTE2 pipe)
//          SetCrossFlag<PIPE_MTE3>(FLAG_V2C)  → Cube
//
// c_l0 is filled once by the GEMM and re-spilled every iteration unchanged.
// There is no global C write — the only GM traffic is the workspace round-trip.
//
// Effective bandwidth per iteration (per core):
//   write  TILE_SIZE × TILE_SIZE × sizeof(fp16)   (Cube  → workspace)
//   read   TILE_SIZE × TILE_SIZE × sizeof(fp16)   (Vec   ← workspace)
//   total  2 × T² × 2  bytes  ×  num_cores  across all cores
//
// Input tensors:
//   A         [num_cores * T, T]  fp16   (initial GEMM input, read once)
//   B         [T, T]              fp16   (weight, read once)
//   workspace [num_cores * T, T]  fp16   (C2V ping-pong buffer)
//   num_iters int32               runtime loop count
//
// Memory budget (per core):
//   L1 (512 KB): b_l1 32 KB + a_l1 32 KB = 64 KB used
//   L0A (64 KB): a_l0  32 KB
//   L0B (64 KB): b_l0  32 KB
//   L0C (128 KB): c_l0 64 KB   (spilled every iteration, never overwritten)
//   UB  (192 KB): c_ub 16 KB   (workspace result, discarded after load)
// =============================================================================

#define MEMORY_BASE
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>

using namespace pto;

#define TILE_SIZE 128
#define HALF_TILE  64
#define VEC_NUM     2

#ifdef __CCE_AICORE__

constexpr uint32_t L1_B_OFFSET  = 0u;
constexpr uint32_t L1_A_OFFSET  = TILE_SIZE * TILE_SIZE * sizeof(half);  // 32 KB
constexpr uint32_t L0_OFFSET    = 0u;
constexpr uint32_t UB_C_OFFSET  = 0u;

constexpr int32_t FLAG_C2V = 0;  // Cube → Vec: workspace tile written
constexpr int32_t FLAG_V2C = 1;  // Vec → Cube: workspace tile consumed

using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;

using TileL0A = TileLeft<half,  TILE_SIZE, TILE_SIZE>;
using TileL0B = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C = TileAcc<float,  TILE_SIZE, TILE_SIZE>;

using TileVecUB =
    Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE,
         BLayout::RowMajor, HALF_TILE, TILE_SIZE, SLayout::NoneBox, 512, PadValue::Null>;

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

template <pipe_t Pipe>
AICORE inline void SetCrossFlag(int32_t flag) {
    ffts_cross_core_sync(Pipe, 1 | (VEC_NUM << 4) | (flag << 8));
}
AICORE inline void WaitCrossFlag(int32_t flag) { wait_flag_dev(flag); }

template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) { set_flag(Src, Dst, static_cast<event_t>(id)); }
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) { wait_flag(Src, Dst, static_cast<event_t>(id)); }

AICORE void run_stream_c2v(
    __gm__ half  *A,          // [num_cores * T, T]  initial GEMM input (A[cid*T:])
    __gm__ half  *B,          // [T, T]              weight matrix
    __gm__ half  *workspace,  // [num_cores * T, T]  C2V ping-pong buffer
    int32_t       num_iters,  // inner loop count (runtime)
    uint64_t      ffts_addr)
{
    const int32_t cid = static_cast<int32_t>(get_block_idx());
    const int32_t vid = static_cast<int32_t>(get_subblockid());

    set_ffts_base_addr(ffts_addr);

    TileL1  b_l1, a_l1;
    TASSIGN(b_l1, L1_B_OFFSET);
    TASSIGN(a_l1, L1_A_OFFSET);

    TileL0A a_l0;  TileL0B b_l0;  TileL0C c_l0;
    TASSIGN(a_l0, L0_OFFSET);
    TASSIGN(b_l0, L0_OFFSET);
    TASSIGN(c_l0, L0_OFFSET);

    TileVecUB c_ub;
    TASSIGN(c_ub, UB_C_OFFSET);

    // ── Cube: one-time setup — GEMM fills c_l0 ────────────────────────────────
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

    // c_l0 is filled here and re-spilled every iteration without recomputing.
    TMATMUL(c_l0, a_l0, b_l0);
    pipe_barrier(PIPE_ALL);

    // ── Cube: inner bandwidth loop ─────────────────────────────────────────────
    TileGlobal ws_out(workspace + cid * TILE_SIZE * TILE_SIZE);
    for (int32_t r = 0; r < num_iters; ++r) {
        // Wait for Vec to finish reading workspace before overwriting it.
        // (Skip round 0: Vec hasn't touched the slot yet.)
        if (r > 0) {
            WaitCrossFlag(FLAG_V2C);
            pipe_barrier(PIPE_ALL);
        }
        TSTORE(ws_out, c_l0);           // L0C → workspace  (FIX pipe, fp32 → fp16)
        pipe_barrier(PIPE_ALL);
        SetCrossFlag<PIPE_FIX>(FLAG_C2V);  // signal Vec: workspace tile is ready
    }

#endif  // __DAV_C220_CUBE__

    // ── Vec: inner bandwidth loop ──────────────────────────────────────────────
#if defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    const int32_t ws_row = cid * TILE_SIZE + vid * HALF_TILE;
    HalfTileGlobal ws_in(workspace + ws_row * TILE_SIZE);

    for (int32_t r = 0; r < num_iters; ++r) {
        WaitCrossFlag(FLAG_C2V);         // workspace tile is ready
        pipe_barrier(PIPE_ALL);
        TLOAD(c_ub, ws_in);              // workspace → UB  (MTE2 pipe)
        pipe_barrier(PIPE_ALL);
        SetCrossFlag<PIPE_MTE3>(FLAG_V2C);  // signal Cube: workspace slot freed
    }

#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__

// ── Kernel entry point ─────────────────────────────────────────────────────────
extern "C" __global__ AICORE void stream_c2v_kernel(
    __gm__ uint8_t *A,
    __gm__ uint8_t *B,
    __gm__ uint8_t *workspace,
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    run_stream_c2v(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(B),
        reinterpret_cast<__gm__ half *>(workspace),
        num_iters, ffts_addr);
}

extern "C" void call_stream_c2v(uint32_t block_dim, void *stream,
                                 uint8_t *A, uint8_t *B,
                                 uint8_t *workspace, int32_t num_iters)
{
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    stream_c2v_kernel<<<block_dim, nullptr, stream>>>(A, B, workspace, num_iters, ffts_addr);
}
