// =============================================================================
// stream_v2c.cpp — Vector→Cube bandwidth microbenchmark
//
// Measures the sustained throughput of the V2C path:
//   Vector UB  →  GM workspace  →  Cube L1
//
// Per iteration the Vec sub-block loads A[row_v:] and D[row_v:] from GM,
// computes element-wise sum in UB, and writes it to workspace.  The Cube reads
// workspace into L1 and executes a GEMM — the GEMM result is discarded (no
// global C write).  The GEMM is required to keep the hardware pipeline active;
// it is the same as in add_matmul_v2c but without the final TSTORE to C.
//
// Inner loop (num_iters times, this is the timed section):
//   Vec:   TLOAD A[row_v:] → a_ub, TLOAD D[row_v:] → b_ub
//          TADD a_ub = a_ub + b_ub
//          if r > 0: WaitCrossFlag(FLAG_C2V)
//          TSTORE a_ub → workspace[ws_row:]      (MTE3 pipe)
//          SetCrossFlag<PIPE_MTE3>(FLAG_V2C)      → Cube
//   Cube:  WaitCrossFlag(FLAG_V2C)
//          TLOAD workspace[cid*T:] → ab_l1        (MTE2 pipe)
//          SetCrossFlag<PIPE_MTE2>(FLAG_C2V)       → Vec  (fires after TLOAD)
//          TMOV ab_l1 → ab_l0,  TMATMUL c_l0 = ab_l0 @ b_l0  (result discarded)
//
// Effective bandwidth per iteration is measured as the workspace round-trip
// (same definition as stream_c2v for a fair C2V vs V2C comparison):
//   write  num_cores × T² × sizeof(fp16)   (Vec  → workspace)
//   read   num_cores × T² × sizeof(fp16)   (Cube ← workspace)
//   total  2 × num_cores × T² × 2  bytes
//
// Input tensors:
//   A         [num_iters * num_cores * T, T]  fp16   (Vec input 1)
//   D         [num_iters * num_cores * T, T]  fp16   (Vec input 2)
//   B         [T, T]                          fp16   (Cube GEMM weight, constant)
//   workspace [num_cores * T, T]              fp16   (V2C ping-pong buffer)
//   num_iters int32                           runtime loop count
//
// Memory budget (per core):
//   L1 (512 KB): b_l1 32 KB + ab_l1 32 KB = 64 KB used
//   L0A (64 KB): ab_l0 32 KB
//   L0B (64 KB): b_l0  32 KB
//   L0C (128 KB): c_l0 64 KB  (GEMM result — discarded each iteration)
//   UB  (192 KB): a_ub 16 KB + b_ub 16 KB = 32 KB used
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
constexpr uint32_t L1_AB_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);  // 32 KB
constexpr uint32_t L0_OFFSET    = 0u;
constexpr uint32_t UB_A_OFFSET  = 0u;
constexpr uint32_t UB_B_OFFSET  = HALF_TILE * TILE_SIZE * sizeof(half);  // 16 KB

constexpr int32_t FLAG_C2V = 0;  // Cube → Vec: workspace slot consumed into L1
constexpr int32_t FLAG_V2C = 1;  // Vec → Cube: workspace tile written to GM

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

AICORE void run_stream_v2c(
    __gm__ half  *A,          // [num_iters * num_cores * T, T]  Vec input 1
    __gm__ half  *D,          // [num_iters * num_cores * T, T]  Vec input 2
    __gm__ half  *B,          // [T, T]                          Cube GEMM weight
    __gm__ half  *workspace,  // [num_cores * T, T]              V2C ping-pong buffer
    int32_t       num_iters,  // runtime loop count
    uint64_t      ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows = num_cores * TILE_SIZE;

    TileL1  b_l1, ab_l1;
    TASSIGN(b_l1,  L1_B_OFFSET);
    TASSIGN(ab_l1, L1_AB_OFFSET);

    TileL0A ab_l0;  TileL0B b_l0;  TileL0C c_l0;
    TASSIGN(ab_l0, L0_OFFSET);
    TASSIGN(b_l0,  L0_OFFSET);
    TASSIGN(c_l0,  L0_OFFSET);

    TileVecUB a_ub, b_ub;
    TASSIGN(a_ub, UB_A_OFFSET);
    TASSIGN(b_ub, UB_B_OFFSET);

    // ── Cube: load constant weight B once, then loop ───────────────────────────
#if defined(__DAV_C220_CUBE__)

    TileGlobal b_global(B);
    TLOAD(b_l1, b_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(b_l0, b_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    TileGlobal ws_in(workspace + cid * TILE_SIZE * TILE_SIZE);
    for (int32_t r = 0; r < num_iters; ++r) {
        // Wait for Vec to write its half-tiles to workspace.
        WaitCrossFlag(FLAG_V2C);

        // Load workspace (A+D sum) into ab_l1.
        TLOAD(ab_l1, ws_in);

        // Signal Vec: workspace slot is captured in L1 — safe to overwrite.
        // SetCrossFlag fires in the MTE2 pipe right after the TLOAD completes,
        // while GEMM (M pipe) can run concurrently with Vec's next iteration.
        SetCrossFlag<PIPE_MTE2>(FLAG_C2V);

        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TMOV(ab_l0, ab_l1);
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        // GEMM: c_l0 = (A+D) @ B — result intentionally discarded (no TSTORE to C)
        TMATMUL(c_l0, ab_l0, b_l0);
        pipe_barrier(PIPE_ALL);
    }

#endif  // __DAV_C220_CUBE__

    // ── Vec: load, add, and store to workspace ─────────────────────────────────
#if defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    const int32_t ws_row = cid * TILE_SIZE + vid * HALF_TILE;
    HalfTileGlobal ws_out(workspace + ws_row * TILE_SIZE);

    for (int32_t r = 0; r < num_iters; ++r) {
        const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        // Load A and D from GM (independent of workspace — prefetch freely).
        HalfTileGlobal a_global(A + row_v * TILE_SIZE);
        TLOAD(a_ub, a_global);

        HalfTileGlobal d_global(D + row_v * TILE_SIZE);
        TLOAD(b_ub, d_global);

        pipe_barrier(PIPE_ALL);

        // Element-wise add: a_ub = A + D
        TADD(a_ub, a_ub, b_ub);
        pipe_barrier(PIPE_ALL);

        // Wait for Cube to free the workspace slot (skip on round 0).
        if (r > 0) {
            WaitCrossFlag(FLAG_C2V);
            pipe_barrier(PIPE_ALL);
        }

        // Write (A+D) sum to workspace.
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
    __gm__ uint8_t *B,
    __gm__ uint8_t *workspace,
    int32_t         num_iters,
    uint64_t        ffts_addr)
{
    run_stream_v2c(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(D),
        reinterpret_cast<__gm__ half *>(B),
        reinterpret_cast<__gm__ half *>(workspace),
        num_iters, ffts_addr);
}

extern "C" void call_stream_v2c(uint32_t block_dim, void *stream,
                                 uint8_t *A, uint8_t *D, uint8_t *B,
                                 uint8_t *workspace, int32_t num_iters)
{
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    stream_v2c_kernel<<<block_dim, nullptr, stream>>>(A, D, B, workspace, num_iters, ffts_addr);
}
