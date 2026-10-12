// =============================================================================
// add_matmul_v2c.cpp — Persistent kernel: C = (A + B) @ D (Vector-to-Cube stream)
//
// Computes  C[batch, T] = (A[batch, T] + B[batch, T]) @ D[T, T]  (fp16)
// where T = TILE_SIZE = 128.
//
// Algorithm (persistent kernel, block_dim == num_cube_cores):
//
//   Vec sub-block (vid ∈ {0,1}), each owns HALF_TILE = T/2 rows — PRODUCER:
//     For each round r:
//       Load A slice → a_ub, Load B slice → b_ub  (prefetch, no dependency on ws)
//       TADD  a_ub = a_ub + b_ub
//       if r > 0: WaitCrossFlag(FLAG_C2V)  ← Cube freed workspace (loaded into L1)
//       TSTORE a_ub → workspace[cid*T + vid*HT :]  (MTE3 pipe)
//       SetCrossFlag FLAG_V2C              ← signal Cube: workspace tile written
//
//   Cube core (cid) — CONSUMER:
//     Load D → d_l1 → d_l0  (once, constant weight)
//     For each round r:
//       WaitCrossFlag(FLAG_V2C)            ← Vec wrote workspace
//       TLOAD workspace[cid*T:] → ab_l1   (MTE2 pipe)
//       SetCrossFlag<PIPE_MTE2> FLAG_C2V   ← signal Vec: workspace freed (right
//                                            after TLOAD, while GEMM is in flight)
//       TMOV ab_l1 → ab_l0  (L1 → L0A)
//       TMATMUL c_l0 = ab_l0 @ d_l0       (GEMM: (A+B) @ D)
//       TSTORE c_l0 → C[row_c:]            (fp32 → fp16, FIX pipe)
//
// Cross-core flags (FFTS):
//   FLAG_V2C = 1  Vec → Cube: workspace tile written, safe to read
//   FLAG_C2V = 0  Cube → Vec: workspace tile consumed into L1, safe to overwrite
//
// Cube signals Vec via PIPE_MTE2 immediately after the workspace TLOAD — this
// lets Vec begin loading A and B and computing A+B for the next round while Cube
// is still executing the GEMM on the already-captured a_l1 data.
//
// Memory budget (per core):
//   L1 (512 KB): d_l1 (32 KB at 0) + ab_l1 (32 KB at 32 KB) = 64 KB used
//   L0A ( 64 KB): ab_l0 (32 KB at 0)
//   L0B ( 64 KB): d_l0  (32 KB at 0)
//   L0C (128 KB): c_l0  (64 KB at 0)
//   UB  (192 KB): a_ub  (16 KB at 0) + b_ub (16 KB at 16 KB) = 32 KB used
// =============================================================================

#define MEMORY_BASE
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>

using namespace pto;

// ── Tile dimensions ────────────────────────────────────────────────────────────
#define TILE_SIZE 128   // rows/cols per matrix tile
#define HALF_TILE  64   // rows per Vec sub-block  (TILE_SIZE / VEC_NUM)
#define VEC_NUM     2   // Vec sub-blocks per Cube core

#ifdef __CCE_AICORE__

// ── On-chip buffer base addresses (bytes) ─────────────────────────────────────
// L1: d_l1 (constant weight D) followed by ab_l1 (workspace: A+B result)
constexpr uint32_t L1_D_OFFSET  = 0u;
constexpr uint32_t L1_AB_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);  // 32 KB

// L0A / L0B / L0C are independent scratchpads; each starts at byte 0
constexpr uint32_t L0_OFFSET = 0u;

// UB: a_ub and b_ub for the Vec add
constexpr uint32_t UB_A_OFFSET = 0u;
constexpr uint32_t UB_B_OFFSET = HALF_TILE * TILE_SIZE * sizeof(half);   // 16 KB

// ── Cross-core FFTS flags ──────────────────────────────────────────────────────
constexpr int32_t FLAG_C2V = 0;  // Cube → Vec: workspace slot consumed into L1
constexpr int32_t FLAG_V2C = 1;  // Vec → Cube: workspace tile written to GM

// ── Tile type aliases ──────────────────────────────────────────────────────────
// L1 tile — NZ (ColMajor/RowMajor) layout required by the Cube engine
using TileL1 = Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE,
                    BLayout::ColMajor, TILE_SIZE, TILE_SIZE,
                    SLayout::RowMajor, 512, PadValue::Zero>;

// L0 tiles — one per independent Cube scratchpad
using TileL0A = TileLeft<half,  TILE_SIZE, TILE_SIZE>;
using TileL0B = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C = TileAcc<float,  TILE_SIZE, TILE_SIZE>;  // fp32 accumulator

// UB Vec tile — row-major, HALF_TILE rows × TILE_SIZE cols, fp16
using TileVecUB = Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE,
                       BLayout::RowMajor, HALF_TILE, TILE_SIZE,
                       SLayout::NoneBox, 512, PadValue::Null>;

// GlobalTensor aliases — contiguous 2D row-major in GM
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

// ── Cross-core sync helpers ────────────────────────────────────────────────────
template <pipe_t Pipe>
AICORE inline void SetCrossFlag(int32_t flag) {
    ffts_cross_core_sync(Pipe, 1 | (VEC_NUM << 4) | (flag << 8));
}

AICORE inline void WaitCrossFlag(int32_t flag) {
    wait_flag_dev(flag);
}

// ── Intra-pipe sync helpers ────────────────────────────────────────────────────
template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) {
    set_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) {
    wait_flag(Src, Dst, static_cast<event_t>(id));
}

// ── Kernel implementation ──────────────────────────────────────────────────────
AICORE void run_add_matmul_v2c(
    __gm__ half    *A,          // [batch, TILE_SIZE]              input
    __gm__ half    *B,          // [batch, TILE_SIZE]              input
    __gm__ half    *C,          // [batch, TILE_SIZE]              output
    __gm__ half    *D,          // [TILE_SIZE, TILE_SIZE]          weight (constant)
    __gm__ half    *workspace,  // [num_cores*TILE_SIZE, TILE_SIZE] V2C buffer
    int64_t         batch,
    uint64_t        ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());   // Cube core id
    const int32_t vid       = static_cast<int32_t>(get_subblockid());  // Vec sub-block: 0 or 1
    const int32_t num_cores = static_cast<int32_t>(block_num);         // launched Cube cores

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows  = num_cores * TILE_SIZE;
    const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

    // ── Allocate on-chip buffers ───────────────────────────────────────────────
    TileL1  d_l1, ab_l1;
    TASSIGN(d_l1,  L1_D_OFFSET);
    TASSIGN(ab_l1, L1_AB_OFFSET);

    TileL0A ab_l0;
    TileL0B d_l0;
    TileL0C c_l0;
    TASSIGN(ab_l0, L0_OFFSET);
    TASSIGN(d_l0,  L0_OFFSET);
    TASSIGN(c_l0,  L0_OFFSET);

    TileVecUB a_ub, b_ub;
    TASSIGN(a_ub, UB_A_OFFSET);
    TASSIGN(b_ub, UB_B_OFFSET);

    // ── Cube core: GEMM ───────────────────────────────────────────────────────
#if defined(__DAV_C220_CUBE__)

    // Load the constant weight D once — reused for every round.
    TileGlobal d_global(D);
    TLOAD(d_l1, d_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(d_l0, d_l1);                   // L1 → L0B (MTE1 pipe)
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_c = r * wave_rows + cid * TILE_SIZE;

        // Wait for both Vec sub-blocks to write their halves of the workspace tile.
        WaitCrossFlag(FLAG_V2C);

        // Load workspace (A+B sum) from GM → ab_l1  (MTE2 pipe).
        TileGlobal ws_in(workspace + cid * TILE_SIZE * TILE_SIZE);
        TLOAD(ab_l1, ws_in);

        // Signal Vec immediately after workspace TLOAD (via MTE2): the workspace slot
        // is now captured in L1, Vec can overwrite it for the next round.
        // This fires in the MTE2 pipe right after the preceding TLOAD — the GEMM and
        // C store (MTE1, M, FIX pipes) run concurrently with Vec's next A+B load.
        SetCrossFlag<PIPE_MTE2>(FLAG_C2V);

        // Sync MTE2 → MTE1: wait for TLOAD to finish before TMOV reads L1.
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);

        TMOV(ab_l0, ab_l1);             // L1 → L0A  (MTE1 pipe)
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0); // M pipe waits for ab_l0 to be ready

        // GEMM: c_l0 = (A+B) @ D  (initialises c_l0)
        TMATMUL(c_l0, ab_l0, d_l0);
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);  // M→FIX: c_l0 ready for TSTORE

        // Store result (fp32 → fp16) to global memory C.
        TileGlobal c_global(C + row_c * TILE_SIZE);
        TSTORE(c_global, c_l0);
        // Drain FIX pipe before the loop back-edge (or kernel exit on the last
        // round): the next TMATMUL writes c_l0, so FIX must finish reading it.
        // Back-to-back benchmark invocations would otherwise trigger an L0C
        // read/write conflict (same pattern that raw_flag matmul_add_c2v avoids
        // with its pipe_barrier before SetCrossFlag).
        pipe_barrier(PIPE_ALL);
    }
    // pipe_barrier(PIPE_ALL) inside the loop already drained the last round.

#endif  // __DAV_C220_CUBE__

    // ── Vec sub-block: element-wise add + store to workspace ──────────────────
#if defined(__DAV_C220_VEC__)

    set_mask_norm();
    set_vector_mask(-1, -1);

    // This sub-block's fixed workspace row offset (constant across rounds).
    const int32_t ws_row = cid * TILE_SIZE + vid * HALF_TILE;

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        // Load A and B slices — independent of the workspace handshake; prefetch
        // them while Cube may still be draining the previous round's workspace.
        HalfTileGlobal a_global(A + row_v * TILE_SIZE);
        TLOAD(a_ub, a_global);

        HalfTileGlobal b_global(B + row_v * TILE_SIZE);
        TLOAD(b_ub, b_global);

        pipe_barrier(PIPE_ALL);  // MTE2→V: both TLOADs done before TADD

        // Compute element-wise sum: a_ub = A + B
        TADD(a_ub, a_ub, b_ub);
        pipe_barrier(PIPE_ALL);  // V→MTE3: TADD done before TSTORE

        // Wait for Cube to signal that it has loaded the previous workspace tile
        // into L1 (slot is free to overwrite). Round 0: no previous tile, skip.
        if (r > 0) {
            WaitCrossFlag(FLAG_C2V);
            pipe_barrier(PIPE_ALL);
        }

        // Write (A+B) sum to the workspace slot for this sub-block.
        HalfTileGlobal ws_out(workspace + ws_row * TILE_SIZE);
        TSTORE(ws_out, a_ub);
        pipe_barrier(PIPE_ALL);  // MTE3: TSTORE complete before SetCrossFlag

        // Signal Cube: workspace tile is fully written, safe to read.
        SetCrossFlag<PIPE_MTE3>(FLAG_V2C);
    }

#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__

// ── Kernel entry point ─────────────────────────────────────────────────────────
extern "C" __global__ AICORE void add_matmul_v2c_kernel(
    __gm__ uint8_t *A,
    __gm__ uint8_t *B,
    __gm__ uint8_t *C,
    __gm__ uint8_t *D,
    __gm__ uint8_t *workspace,
    int64_t         batch,
    uint64_t        ffts_addr)
{
    run_add_matmul_v2c(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(B),
        reinterpret_cast<__gm__ half *>(C),
        reinterpret_cast<__gm__ half *>(D),
        reinterpret_cast<__gm__ half *>(workspace),
        batch, ffts_addr);
}

// ── Host-side launcher (called from Python via ctypes) ─────────────────────────
extern "C" void call(uint32_t block_dim, void *stream,
                     uint8_t *A, uint8_t *B, uint8_t *C,
                     uint8_t *D, uint8_t *workspace, int64_t batch)
{
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    add_matmul_v2c_kernel<<<block_dim, nullptr, stream>>>(
        A, B, C, D, workspace, batch, ffts_addr);
}
