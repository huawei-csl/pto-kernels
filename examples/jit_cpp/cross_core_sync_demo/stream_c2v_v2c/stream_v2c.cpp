// =============================================================================
// stream_v2c.cpp — Vector→Cube bandwidth microbenchmark
//
// Measures the sustained throughput of the V2C path:
//   Vector UB  →  GM workspace  →  Cube L1
//
// Unlike the TileLang reference (which required a GEMM on the Cube side for
// compiler reasons), this PTO C++ version strips the Cube work down to the
// bare minimum: load workspace into L1, then immediately free the slot.
//
// Inner loop (num_iters times, this is the timed section):
//   Vec:   TLOAD A[row_v:] → a_ub, TLOAD D[row_v:] → b_ub
//          TADD a_ub = a_ub + b_ub
//          if r > 0: WaitCrossFlag(FLAG_C2V)
//          TSTORE a_ub → workspace[ws_row:]     (MTE3 pipe)
//          SetCrossFlag<PIPE_MTE3>(FLAG_V2C)     → Cube
//   Cube:  WaitCrossFlag(FLAG_V2C)
//          TLOAD workspace[cid*T:] → ws_l1      (MTE2 pipe, data discarded)
//          SetCrossFlag<PIPE_MTE2>(FLAG_C2V)     → Vec  (fires after TLOAD)
//
// The SetCrossFlag<PIPE_MTE2> fires in the MTE2 instruction stream right after
// the TLOAD, so it signals Vec the moment the workspace slot is captured in L1.
// Vec can then write fresh data while Cube is already done with the slot.
//
// Effective bandwidth (same definition as stream_c2v for a fair comparison):
//   write  num_cores × T² × sizeof(fp16)   (Vec  → workspace)
//   read   num_cores × T² × sizeof(fp16)   (Cube ← workspace)
//   total  2 × num_cores × T² × 2  bytes  per iteration
//
// Input tensors:
//   A         [num_iters * num_cores * T, T]  fp16   (Vec input 1)
//   D         [num_iters * num_cores * T, T]  fp16   (Vec input 2)
//   workspace [num_cores * T, T]              fp16   (V2C ping-pong buffer)
//   num_iters int32                           runtime loop count
//
// Memory budget (per core):
//   L1  (512 KB): ws_l1 32 KB  (workspace read buffer, discarded each iteration)
//   UB  (192 KB): a_ub 16 KB + b_ub 16 KB = 32 KB used
//   L0A / L0B / L0C: unused
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

constexpr uint32_t L1_WS_OFFSET = 0u;           // workspace read buffer in L1
constexpr uint32_t UB_A_OFFSET  = 0u;
constexpr uint32_t UB_B_OFFSET  = HALF_TILE * TILE_SIZE * sizeof(half);  // 16 KB

constexpr int32_t FLAG_C2V = 0;  // Cube → Vec: workspace slot consumed into L1
constexpr int32_t FLAG_V2C = 1;  // Vec → Cube: workspace tile written to GM

using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;

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

AICORE void run_stream_v2c(
    __gm__ half  *A,          // [num_iters * num_cores * T, T]  Vec input 1
    __gm__ half  *D,          // [num_iters * num_cores * T, T]  Vec input 2
    __gm__ half  *workspace,  // [num_cores * T, T]              V2C ping-pong buffer
    int32_t       num_iters,
    uint64_t      ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows = num_cores * TILE_SIZE;

    TileL1    ws_l1;
    TASSIGN(ws_l1, L1_WS_OFFSET);

    TileVecUB a_ub, b_ub;
    TASSIGN(a_ub, UB_A_OFFSET);
    TASSIGN(b_ub, UB_B_OFFSET);

    // ── Cube: load workspace into L1, discard data, free slot immediately ──────
#if defined(__DAV_C220_CUBE__)

    TileGlobal ws_in(workspace + cid * TILE_SIZE * TILE_SIZE);
    for (int32_t r = 0; r < num_iters; ++r) {
        // Wait for both Vec sub-blocks to write their halves of the workspace tile.
        WaitCrossFlag(FLAG_V2C);

        // Capture workspace into L1 (the measured read).
        TLOAD(ws_l1, ws_in);

        // Signal Vec immediately from the MTE2 pipe: workspace slot is consumed
        // (in L1) — Vec can overwrite it for the next round.
        SetCrossFlag<PIPE_MTE2>(FLAG_C2V);

        // Drain MTE2 before the next iteration's TLOAD touches ws_l1 again.
        pipe_barrier(PIPE_MTE2);
    }

#endif  // __DAV_C220_CUBE__

    // ── Vec: load A+D, add, write to workspace ─────────────────────────────────
#if defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    const int32_t ws_row = cid * TILE_SIZE + vid * HALF_TILE;
    HalfTileGlobal ws_out(workspace + ws_row * TILE_SIZE);

    for (int32_t r = 0; r < num_iters; ++r) {
        const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        // Load A and D — independent of the workspace handshake, so prefetch.
        HalfTileGlobal a_global(A + row_v * TILE_SIZE);
        TLOAD(a_ub, a_global);

        HalfTileGlobal d_global(D + row_v * TILE_SIZE);
        TLOAD(b_ub, d_global);

        pipe_barrier(PIPE_ALL);

        TADD(a_ub, a_ub, b_ub);
        pipe_barrier(PIPE_ALL);

        // Wait for Cube to free the workspace slot (skip on round 0).
        if (r > 0) {
            WaitCrossFlag(FLAG_C2V);
            pipe_barrier(PIPE_ALL);
        }

        TSTORE(ws_out, a_ub);
        pipe_barrier(PIPE_ALL);

        // Signal Cube: workspace tile is ready to read.
        SetCrossFlag<PIPE_MTE3>(FLAG_V2C);
    }

#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__

// ── Kernel entry point ─────────────────────────────────────────────────────────
extern "C" __global__ AICORE void stream_v2c_kernel(
    __gm__ uint8_t *A,
    __gm__ uint8_t *D,
    __gm__ uint8_t *workspace,
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    run_stream_v2c(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(D),
        reinterpret_cast<__gm__ half *>(workspace),
        num_iters, ffts_addr);
}

extern "C" void call_stream_v2c(uint32_t block_dim, void *stream,
                                 uint8_t *A, uint8_t *D,
                                 uint8_t *workspace, int32_t num_iters)
{
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    stream_v2c_kernel<<<block_dim, nullptr, stream>>>(A, D, workspace, num_iters, ffts_addr);
}
