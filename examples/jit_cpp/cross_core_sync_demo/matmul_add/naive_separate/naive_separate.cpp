// =============================================================================
// naive_separate.cpp — Naive two-stage baseline: no Cube↔Vec pipeline
//
// Computes the same operations as the pipelined matmul_add kernels, but the
// two stages (GEMM and element-wise add) are executed sequentially within each
// core pair.  There is no fine-grained round-by-round overlap between Cube and
// Vec: the first stage completes ALL rounds before the second stage starts.
//
// Two kernels (sharing the four stage-helper functions below):
//
//   matmul_add_c2v  "C = A @ B + D"
//     Stage 1 (Cube): GEMM A@B for every round → workspace
//     Stage 2 (Vec):  C = workspace + D for every round
//
//   add_matmul_v2c  "C = (A + B) @ D"
//     Stage 1 (Vec):  (A + B) for every round → workspace
//     Stage 2 (Cube): GEMM workspace@D for every round → C
//
// Workspace layout — both kernels use the same shape:
//   workspace[batch, TILE_SIZE]  fp16   (== same size as A, C, D)
//   Slot for core cid at round r:
//     workspace[r * num_cores * TILE_SIZE + cid * TILE_SIZE : ..., :]
//   Each slot is TILE_SIZE × TILE_SIZE elements; Vec sub-block vid reads/writes
//   the half-rows at vid * HALF_TILE inside that slot.
//
// Cross-core synchronization (one signal per direction, not per round):
//   matmul_add_c2v:
//     Cube signals FLAG_C2V *once* after all GEMM rounds are in workspace.
//     Both Vec sub-blocks wait for that one signal then drain workspace.
//   add_matmul_v2c:
//     Each Vec sub-block signals FLAG_V2C *once* after writing all its rows.
//     Cube waits for VEC_NUM=2 signals then drains workspace.
//
// Compared with pipelined versions: within each core pair the two stages are
// serialized — no round-level overlap — so effective bandwidth is lower.
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
// L1: constant weight first, data tile second
constexpr uint32_t L1_CONST_OFFSET = 0u;
constexpr uint32_t L1_DATA_OFFSET  = TILE_SIZE * TILE_SIZE * sizeof(half);  // 32 KB

// L0A / L0B / L0C are independent scratchpads; each starts at byte 0
constexpr uint32_t L0_OFFSET = 0u;

// UB: two HALF_TILE × TILE_SIZE half tiles
constexpr uint32_t UB_SLOT0_OFFSET = 0u;
constexpr uint32_t UB_SLOT1_OFFSET = HALF_TILE * TILE_SIZE * sizeof(half);  // 16 KB

// ── Cross-core FFTS flag IDs ───────────────────────────────────────────────────
// For matmul_add_c2v: Cube → Vec  (workspace filled with GEMM results)
// For add_matmul_v2c: Vec → Cube  (workspace filled with A+B results)
constexpr int32_t FLAG_C2V = 0;
constexpr int32_t FLAG_V2C = 1;

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
// One call from Cube signals FLAG and unblocks all VEC_NUM Vec sub-blocks.
// One call from each Vec sub-block signals FLAG; Cube unblocks after VEC_NUM.
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

// =============================================================================
// Stage helpers — shared by both kernels
//
// Each helper is guarded by the compilation pass it belongs to
// (__DAV_C220_CUBE__ or __DAV_C220_VEC__).  bisheng compiles the source twice
// (once per pass) so we must keep Cube-only and Vec-only instructions in their
// respective guards.  Wrapping helpers this way lets run_* call them within the
// matching #if blocks, providing code reuse across both kernels.
// =============================================================================

// ── Cube-side helpers ──────────────────────────────────────────────────────────
#if defined(__DAV_C220_CUBE__)

// Stage 1 of matmul_add_c2v (Cube side):
//   Compute A @ B for every round and write results to workspace.
//   After the final TSTORE, send FLAG_C2V to signal both Vec sub-blocks.
//
// Workspace offset for round r, core cid:
//   workspace + (r * num_cores * TILE_SIZE + cid * TILE_SIZE) * TILE_SIZE
AICORE void cube_gemm_all_rounds_to_ws(
    __gm__ half *A,         // [batch, TILE_SIZE]              input
    __gm__ half *B,         // [TILE_SIZE, TILE_SIZE]          constant weight
    __gm__ half *workspace, // [batch, TILE_SIZE]              output of this stage
    int32_t cid, int32_t num_cores, int32_t num_rounds)
{
    TileL1  b_l1, a_l1;
    TileL0A a_l0;
    TileL0B b_l0;
    TileL0C c_l0;
    TASSIGN(b_l1, L1_CONST_OFFSET);
    TASSIGN(a_l1, L1_DATA_OFFSET);
    TASSIGN(a_l0, L0_OFFSET);
    TASSIGN(b_l0, L0_OFFSET);
    TASSIGN(c_l0, L0_OFFSET);

    const int32_t wave_rows = num_cores * TILE_SIZE;

    // Load constant weight B once — reused for all rounds
    TileGlobal b_global(B);
    TLOAD(b_l1, b_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(b_l0, b_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_a  = r * wave_rows + cid * TILE_SIZE;
        const int32_t ws_row = r * wave_rows + cid * TILE_SIZE;

        // Load A tile: GM → L1 → L0A
        TileGlobal a_global(A + row_a * TILE_SIZE);
        TLOAD(a_l1, a_global);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TMOV(a_l0, a_l1);
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        // GEMM: c_l0 = A @ B  (fp32 accumulator)
        TMATMUL(c_l0, a_l0, b_l0);
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);  // M → FIX: c_l0 ready for TSTORE

        // Write result to workspace (fp32 → fp16 via FIX pipe)
        TileGlobal ws_out(workspace + ws_row * TILE_SIZE);
        TSTORE(ws_out, c_l0);

        // Drain all pipes before the next round.  Without this barrier the
        // next round's TMOV(a_l0) on MTE1 would race with the current round's
        // TMATMUL still reading a_l0 on M (no cross-core wait provides the
        // implicit ordering that the pipelined version relies on).
        pipe_barrier(PIPE_ALL);
    }

    // All GEMM rounds written; pipe_barrier at end of last iteration already
    // flushed DMAs — signal both Vec sub-blocks.
    SetCrossFlag<PIPE_FIX>(FLAG_C2V);
}

// Stage 2 of add_matmul_v2c (Cube side):
//   Wait for both Vec sub-blocks to finish stage 1 (FLAG_V2C × VEC_NUM),
//   then for each round: load workspace, GEMM, store C.
AICORE void cube_gemm_from_ws_all_rounds(
    __gm__ half *workspace, // [batch, TILE_SIZE]  A+B results from stage 1
    __gm__ half *D,         // [TILE_SIZE, TILE_SIZE]  constant weight
    __gm__ half *C,         // [batch, TILE_SIZE]  output
    int32_t cid, int32_t num_cores, int32_t num_rounds)
{
    TileL1  d_l1, ab_l1;
    TileL0A ab_l0;
    TileL0B d_l0;
    TileL0C c_l0;
    TASSIGN(d_l1,  L1_CONST_OFFSET);
    TASSIGN(ab_l1, L1_DATA_OFFSET);
    TASSIGN(ab_l0, L0_OFFSET);
    TASSIGN(d_l0,  L0_OFFSET);
    TASSIGN(c_l0,  L0_OFFSET);

    const int32_t wave_rows = num_cores * TILE_SIZE;

    // Load constant weight D once — reused for all rounds
    TileGlobal d_global(D);
    TLOAD(d_l1, d_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(d_l0, d_l1);
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);

    // Wait for BOTH Vec sub-blocks to finish all A+B rounds (VEC_NUM signals)
    WaitCrossFlag(FLAG_V2C);
    pipe_barrier(PIPE_ALL);  // ensure workspace writes are visible to this Cube core

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_c  = r * wave_rows + cid * TILE_SIZE;
        const int32_t ws_row = r * wave_rows + cid * TILE_SIZE;

        // Load workspace (A+B): GM → L1 → L0A
        TileGlobal ws_in(workspace + ws_row * TILE_SIZE);
        TLOAD(ab_l1, ws_in);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TMOV(ab_l0, ab_l1);
        SetFlag<PIPE_MTE1, PIPE_M>(0);
        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        // GEMM: c_l0 = (A+B) @ D
        TMATMUL(c_l0, ab_l0, d_l0);
        SetFlag<PIPE_M, PIPE_FIX>(0);
        WaitFlag<PIPE_M, PIPE_FIX>(0);

        // Store result (fp32 → fp16) to global memory C
        TileGlobal c_global(C + row_c * TILE_SIZE);
        TSTORE(c_global, c_l0);
        // Drain all pipes before the next round (same reasoning as
        // cube_gemm_all_rounds_to_ws: prevent L0A/L0C conflicts on loop back-edge).
        pipe_barrier(PIPE_ALL);
    }
}

#endif  // __DAV_C220_CUBE__

// ── Vec-side helpers ───────────────────────────────────────────────────────────
#if defined(__DAV_C220_VEC__)

// Stage 2 of matmul_add_c2v (Vec side):
//   Wait for FLAG_C2V (Cube has filled workspace), then for each round:
//   load the GEMM result and D, compute c = gemm_result + D, store C.
AICORE void vec_add_from_ws_all_rounds(
    __gm__ half *workspace, // [batch, TILE_SIZE]  GEMM results from stage 1
    __gm__ half *D,         // [batch, TILE_SIZE]  bias
    __gm__ half *C,         // [batch, TILE_SIZE]  output
    int32_t cid, int32_t vid, int32_t num_cores, int32_t num_rounds)
{
    TileVecUB c_ub, d_ub;
    TASSIGN(c_ub, UB_SLOT0_OFFSET);
    TASSIGN(d_ub, UB_SLOT1_OFFSET);

    const int32_t wave_rows = num_cores * TILE_SIZE;

    // Wait until all GEMM rounds are in workspace (one-shot signal from Cube)
    WaitCrossFlag(FLAG_C2V);
    pipe_barrier(PIPE_ALL);  // ensure workspace writes are visible to this Vec sub-block

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_v  = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;
        const int32_t ws_row = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        // Load GEMM result slice and D slice from GM → UB (both issue to MTE2)
        HalfTileGlobal ws_in(workspace + ws_row * TILE_SIZE);
        TLOAD(c_ub, ws_in);

        HalfTileGlobal d_global(D + row_v * TILE_SIZE);
        TLOAD(d_ub, d_global);

        SetFlag<PIPE_MTE2, PIPE_V>(0);
        WaitFlag<PIPE_MTE2, PIPE_V>(0);  // both TLOADs done before TADD

        // C = gemm_result + D  (element-wise, Vec engine)
        TADD(c_ub, c_ub, d_ub);
        SetFlag<PIPE_V, PIPE_MTE3>(0);
        WaitFlag<PIPE_V, PIPE_MTE3>(0);  // TADD done before TSTORE

        HalfTileGlobal c_out(C + row_v * TILE_SIZE);
        TSTORE(c_out, c_ub);

        // Wait for TSTORE to complete before the next iteration reuses c_ub
        SetFlag<PIPE_MTE3, PIPE_MTE2>(0);
        WaitFlag<PIPE_MTE3, PIPE_MTE2>(0);
    }
}

// Stage 1 of add_matmul_v2c (Vec side):
//   For each round: load A and B slices, compute a_ub = A + B, store to workspace.
//   After the final TSTORE, send FLAG_V2C to signal Cube.
AICORE void vec_add_all_rounds_to_ws(
    __gm__ half *A,         // [batch, TILE_SIZE]  input
    __gm__ half *B,         // [batch, TILE_SIZE]  input
    __gm__ half *workspace, // [batch, TILE_SIZE]  output of this stage
    int32_t cid, int32_t vid, int32_t num_cores, int32_t num_rounds)
{
    TileVecUB a_ub, b_ub;
    TASSIGN(a_ub, UB_SLOT0_OFFSET);
    TASSIGN(b_ub, UB_SLOT1_OFFSET);

    const int32_t wave_rows = num_cores * TILE_SIZE;

    for (int32_t r = 0; r < num_rounds; ++r) {
        const int32_t row_v  = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;
        const int32_t ws_row = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

        // Load A and B slices from GM → UB
        HalfTileGlobal a_global(A + row_v * TILE_SIZE);
        TLOAD(a_ub, a_global);

        HalfTileGlobal b_global(B + row_v * TILE_SIZE);
        TLOAD(b_ub, b_global);

        SetFlag<PIPE_MTE2, PIPE_V>(0);
        WaitFlag<PIPE_MTE2, PIPE_V>(0);  // both TLOADs done before TADD

        // a_ub = A + B  (element-wise, Vec engine)
        TADD(a_ub, a_ub, b_ub);
        SetFlag<PIPE_V, PIPE_MTE3>(0);
        WaitFlag<PIPE_V, PIPE_MTE3>(0);  // TADD done before TSTORE

        HalfTileGlobal ws_out(workspace + ws_row * TILE_SIZE);
        TSTORE(ws_out, a_ub);

        // Wait for TSTORE before next iteration reuses a_ub
        SetFlag<PIPE_MTE3, PIPE_MTE2>(0);
        WaitFlag<PIPE_MTE3, PIPE_MTE2>(0);
    }

    // All A+B rounds written — flush DMAs then signal Cube
    pipe_barrier(PIPE_ALL);
    SetCrossFlag<PIPE_MTE3>(FLAG_V2C);
}

#endif  // __DAV_C220_VEC__

// =============================================================================
// Kernel entry points
// =============================================================================

// matmul_add_c2v: C = A @ B + D
//   Stage 1 (Cube):  A @ B → workspace   (all rounds)
//   Stage 2 (Vec):   workspace + D → C   (all rounds, after stage 1)
AICORE void run_matmul_add_c2v(
    __gm__ half *A,         // [batch, TILE_SIZE]
    __gm__ half *B,         // [TILE_SIZE, TILE_SIZE]
    __gm__ half *C,         // [batch, TILE_SIZE]
    __gm__ half *D,         // [batch, TILE_SIZE]
    __gm__ half *workspace, // [batch, TILE_SIZE]
    int64_t batch, uint64_t ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows  = num_cores * TILE_SIZE;
    const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

#if defined(__DAV_C220_CUBE__)
    cube_gemm_all_rounds_to_ws(A, B, workspace, cid, num_cores, num_rounds);
#endif

#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    vec_add_from_ws_all_rounds(workspace, D, C, cid, vid, num_cores, num_rounds);
#endif
}

// add_matmul_v2c: C = (A + B) @ D
//   Stage 1 (Vec):  A + B → workspace   (all rounds)
//   Stage 2 (Cube): workspace @ D → C   (all rounds, after stage 1)
AICORE void run_add_matmul_v2c(
    __gm__ half *A,         // [batch, TILE_SIZE]
    __gm__ half *B,         // [batch, TILE_SIZE]
    __gm__ half *C,         // [batch, TILE_SIZE]
    __gm__ half *D,         // [TILE_SIZE, TILE_SIZE]
    __gm__ half *workspace, // [batch, TILE_SIZE]
    int64_t batch, uint64_t ffts_addr)
{
    const int32_t cid       = static_cast<int32_t>(get_block_idx());
    const int32_t vid       = static_cast<int32_t>(get_subblockid());
    const int32_t num_cores = static_cast<int32_t>(block_num);

    set_ffts_base_addr(ffts_addr);

    const int32_t wave_rows  = num_cores * TILE_SIZE;
    const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    vec_add_all_rounds_to_ws(A, B, workspace, cid, vid, num_cores, num_rounds);
#endif

#if defined(__DAV_C220_CUBE__)
    cube_gemm_from_ws_all_rounds(workspace, D, C, cid, num_cores, num_rounds);
#endif
}

#endif  // __CCE_AICORE__

// ── Kernel entry points (extern "C" __global__) ───────────────────────────────
extern "C" __global__ AICORE void matmul_add_c2v_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *B,
    __gm__ uint8_t *C, __gm__ uint8_t *D,
    __gm__ uint8_t *workspace, int64_t batch, uint64_t ffts_addr)
{
    run_matmul_add_c2v(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(B),
        reinterpret_cast<__gm__ half *>(C),
        reinterpret_cast<__gm__ half *>(D),
        reinterpret_cast<__gm__ half *>(workspace),
        batch, ffts_addr);
}

extern "C" __global__ AICORE void add_matmul_v2c_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *B,
    __gm__ uint8_t *C, __gm__ uint8_t *D,
    __gm__ uint8_t *workspace, int64_t batch, uint64_t ffts_addr)
{
    run_add_matmul_v2c(
        reinterpret_cast<__gm__ half *>(A),
        reinterpret_cast<__gm__ half *>(B),
        reinterpret_cast<__gm__ half *>(C),
        reinterpret_cast<__gm__ half *>(D),
        reinterpret_cast<__gm__ half *>(workspace),
        batch, ffts_addr);
}

// ── Host-side launchers (called from Python via ctypes) ───────────────────────
static inline uint64_t _get_ffts_addr() {
    uint32_t ffts_len  = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
    return ffts_addr;
}

extern "C" void call_matmul_add_c2v(
    uint32_t block_dim, void *stream,
    uint8_t *A, uint8_t *B, uint8_t *C,
    uint8_t *D, uint8_t *workspace, int64_t batch)
{
    matmul_add_c2v_kernel<<<block_dim, nullptr, stream>>>(
        A, B, C, D, workspace, batch, _get_ffts_addr());
}

extern "C" void call_add_matmul_v2c(
    uint32_t block_dim, void *stream,
    uint8_t *A, uint8_t *B, uint8_t *C,
    uint8_t *D, uint8_t *workspace, int64_t batch)
{
    add_matmul_v2c_kernel<<<block_dim, nullptr, stream>>>(
        A, B, C, D, workspace, batch, _get_ffts_addr());
}
