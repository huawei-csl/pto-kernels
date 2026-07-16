// =============================================================================
// matmul_add_c2v.cpp — Persistent kernel: C = A @ B + D (Cube-to-Vec stream)
//
// Computes  C[batch, T] = A[batch, T] @ B[T, T] + D[batch, T]
// where T = TILE_SIZE = 128.
//
// Algorithm (persistent kernel, block_dim == num_cube_cores):
//   Each "round" all Cube cores process one wave of TILE_SIZE rows in parallel.
//   After num_rounds waves the full batch is consumed.
//
//   Cube core (cid):
//     1. Load B[0:T, 0:T] → L1 → L0B  (once, before the loop)
//     2. For each round r:
//          a. Load A[row_c:row_c+T, :] → L1 → L0A
//          b. GEMM: c_l0 = A @ B  (L0C ← L0A × L0B)
//          c. If r > 0: wait for Vec signal (FLAG_V2C) confirming workspace
//          freed d. TSTORE c_l0 → workspace[cid*T : (cid+1)*T, :]  (FIX pipe,
//          f32→f16) e. SetCrossFlag FLAG_C2V  (signals both Vec sub-blocks:
//          tile ready)
//
//   Vec sub-block (cid, vid ∈ {0,1}):
//     Each sub-block owns half the tile rows: vid*T/2 .. (vid+1)*T/2
//     For each round r:
//          a. WaitCrossFlag FLAG_C2V  (Cube has written workspace)
//          b. TLOAD workspace slice → c_ub
//          c. TLOAD D slice         → d_ub
//          d. pipe_barrier(ALL)  — both loads complete
//          e. SetCrossFlag FLAG_V2C  (signals Cube: workspace slot freed)
//          f. TADD c_ub = c_ub + d_ub
//          g. TSTORE c_ub → C output
//
// Cross-core sync uses FFTS (Fast Fine-grained Task Synchronization).
// Flag mode = VEC_NUM = 2:
//   • Cube sends FLAG_C2V once; hardware delivers it to both Vec sub-blocks.
//   • Each Vec sub-block sends FLAG_V2C once; Cube unblocks after VEC_NUM
//   signals.
//
// Memory budget:
//   L1  (512 KB): b_l1 (32 KB at 0) + a_l1 (32 KB at 32 KB) = 64 KB used
//   L0A ( 64 KB): a_l0 (32 KB at 0)
//   L0B ( 64 KB): b_l0 (32 KB at 0)
//   L0C (128 KB): c_l0 (64 KB at 0)
//   UB  (192 KB): c_ub (16 KB at 0) + d_ub (16 KB at 16 KB) = 32 KB used
// =============================================================================

#include <runtime/rt_ffts.h>

#include <pto/pto-inst.hpp>

#include "acl/acl.h"

using namespace pto;

// ── Tile dimensions
// ────────────────────────────────────────────────────────────
#define TILE_SIZE 128  // rows/cols per matrix tile
#define HALF_TILE 64   // rows per Vec sub-block  (TILE_SIZE / VEC_NUM)
#define VEC_NUM 2      // Vec sub-blocks per Cube core

#ifdef __CCE_AICORE__

// ── On-chip buffer base addresses (bytes)
// ───────────────────────────────────── L1: two back-to-back
// TILE_SIZE×TILE_SIZE half tiles
constexpr uint32_t L1_B_OFFSET = 0u;
constexpr uint32_t L1_A_OFFSET = TILE_SIZE * TILE_SIZE * sizeof(half);  // 32 KB

// L0A / L0B / L0C are independent scratchpads; each starts at byte 0
constexpr uint32_t L0_OFFSET = 0u;

// UB: two HALF_TILE×TILE_SIZE half tiles
constexpr uint32_t UB_C_OFFSET = 0u;
constexpr uint32_t UB_D_OFFSET = HALF_TILE * TILE_SIZE * sizeof(half);  // 16 KB

// ── Cross-core FFTS flags
// ──────────────────────────────────────────────────────
constexpr int32_t FLAG_C2V = 0;  // Cube → Vec: GEMM result written to workspace
constexpr int32_t FLAG_V2C = 1;  // Vec → Cube: workspace slot has been read

// ── Tile type aliases
// ────────────────────────────────────────────────────────── L1 tile — NZ
// (ColMajor/RowMajor) layout required by the Cube engine
using TileL1 =
    Tile<TileType::Mat, half, TILE_SIZE, TILE_SIZE, BLayout::ColMajor,
         TILE_SIZE, TILE_SIZE, SLayout::RowMajor, 512, PadValue::Zero>;

// L0 tiles — one per independent Cube scratchpad
using TileL0A = TileLeft<half, TILE_SIZE, TILE_SIZE>;
using TileL0B = TileRight<half, TILE_SIZE, TILE_SIZE>;
using TileL0C = TileAcc<float, TILE_SIZE, TILE_SIZE>;  // fp32 accumulator

// UB Vec tile — row-major, HALF_TILE rows × TILE_SIZE cols, fp16
using TileVecUB =
    Tile<TileType::Vec, half, HALF_TILE, TILE_SIZE, BLayout::RowMajor,
         HALF_TILE, TILE_SIZE, SLayout::NoneBox, 512, PadValue::Null>;

// GlobalTensor aliases — contiguous 2D row-major in GM
using TileGlobal =
    GlobalTensor<half, TileShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, TILE_SIZE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

using HalfTileGlobal =
    GlobalTensor<half, TileShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 BaseShape2D<half, HALF_TILE, TILE_SIZE, Layout::ND>,
                 Layout::ND>;

// ── Cross-core sync helpers
// ──────────────────────────────────────────────────── SetCrossFlag: insert a
// signal into `Pipe`'s instruction stream.
//   mode = VEC_NUM means:
//     • When Cube signals FLAG_C2V: one call unblocks all VEC_NUM Vec
//     sub-blocks. • When Vec signals FLAG_V2C: Cube unblocks after receiving
//     VEC_NUM signals
//       (one per sub-block).
template <pipe_t Pipe>
AICORE inline void SetCrossFlag(int32_t flag) {
  ffts_cross_core_sync(Pipe, 1 | (VEC_NUM << 4) | (flag << 8));
}

AICORE inline void WaitCrossFlag(int32_t flag) { wait_flag_dev(flag); }

// ── Intra-pipe sync helpers
// ────────────────────────────────────────────────────
template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) {
  set_flag(Src, Dst, static_cast<event_t>(id));
}

template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) {
  wait_flag(Src, Dst, static_cast<event_t>(id));
}

// ── Kernel implementation
// ──────────────────────────────────────────────────────
AICORE void run_matmul_add_c2v(
    __gm__ half *A,  // [batch, TILE_SIZE]              input matrix
    __gm__ half *B,  // [TILE_SIZE, TILE_SIZE]          weight (constant)
    __gm__ half *C,  // [batch, TILE_SIZE]              output
    __gm__ half *D,  // [batch, TILE_SIZE]              bias
    __gm__ half *workspace,  // [num_cores*TILE_SIZE, TILE_SIZE] C2V buffer
    int64_t batch, uint64_t ffts_addr) {
  const int32_t cid = static_cast<int32_t>(get_block_idx());  // Cube core id
  const int32_t vid =
      static_cast<int32_t>(get_subblockid());  // Vec sub-block: 0 or 1
  const int32_t num_cores =
      static_cast<int32_t>(block_num);  // launched Cube cores

  set_ffts_base_addr(ffts_addr);

  // One wave processes (num_cores × TILE_SIZE) rows across all cores.
  const int32_t wave_rows = num_cores * TILE_SIZE;
  const int32_t num_rounds = static_cast<int32_t>(batch) / wave_rows;

  // ── Allocate on-chip buffers ───────────────────────────────────────────────
  TileL1 b_l1, a_l1;
  TASSIGN(b_l1, L1_B_OFFSET);
  TASSIGN(a_l1, L1_A_OFFSET);

  TileL0A a_l0;
  TileL0B b_l0;
  TileL0C c_l0;
  TASSIGN(a_l0, L0_OFFSET);
  TASSIGN(b_l0, L0_OFFSET);
  TASSIGN(c_l0, L0_OFFSET);

  TileVecUB c_ub, d_ub;
  TASSIGN(c_ub, UB_C_OFFSET);
  TASSIGN(d_ub, UB_D_OFFSET);

  // ── Cube core: GEMM ───────────────────────────────────────────────────────
#if defined(__DAV_C220_CUBE__)

  // Load the constant weight matrix B once — reused for every round.
  TileGlobal b_global(B);
  TLOAD(b_l1, b_global);
  SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
  WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(b_l0, b_l1);  // L1 → L0B (MTE1 pipe)
  SetFlag<PIPE_MTE1, PIPE_M>(0);
  WaitFlag<PIPE_MTE1, PIPE_M>(0);  // M pipe waits for b_l0 to be ready

  for (int32_t r = 0; r < num_rounds; ++r) {
    // Row offset in A for this core + round
    const int32_t row_c = r * wave_rows + cid * TILE_SIZE;

    // Load A tile: GM → L1 → L0A
    TileGlobal a_global(A + row_c * TILE_SIZE);
    TLOAD(a_l1, a_global);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(a_l0, a_l1);  // L1 → L0A (MTE1 pipe)
    SetFlag<PIPE_MTE1, PIPE_M>(0);
    WaitFlag<PIPE_MTE1, PIPE_M>(0);  // M pipe waits for a_l0 to be ready

    // GEMM: c_l0 = A @ B  (initialises c_l0 — no prior accumulation)
    TMATMUL(c_l0, a_l0, b_l0);
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);  // M→FIX: c_l0 ready for TSTORE

    // Wait for both Vec sub-blocks to finish reading the workspace slot
    // from the *previous* round before overwriting it. Skip on round 0.
    if (r > 0) {
      WaitCrossFlag(FLAG_V2C);
      pipe_barrier(PIPE_ALL);  // flush all pipes before issuing TSTORE
    }

    // Write GEMM result to workspace (fp32 → fp16 conversion via FIX pipe).
    TileGlobal ws_out(workspace + cid * TILE_SIZE * TILE_SIZE);
    TSTORE(ws_out, c_l0);
    pipe_barrier(PIPE_ALL);  // FIX: TSTORE complete before SetCrossFlag fires
    SetCrossFlag<PIPE_FIX>(FLAG_C2V);
  }

#endif  // __DAV_C220_CUBE__

  // ── Vec sub-block: add bias + store result ─────────────────────────────────
#if defined(__DAV_C220_VEC__)

  set_mask_norm();
  set_vector_mask(-1, -1);

  // This sub-block's workspace row offset (fixed across all rounds).
  const int32_t ws_row = cid * TILE_SIZE + vid * HALF_TILE;

  for (int32_t r = 0; r < num_rounds; ++r) {
    // Global output row this sub-block writes
    const int32_t row_v = r * wave_rows + cid * TILE_SIZE + vid * HALF_TILE;

    // Wait for Cube: workspace slot contains a fresh GEMM result.
    WaitCrossFlag(FLAG_C2V);

    // Load GEMM result and D from GM → UB (both loads can issue in parallel).
    HalfTileGlobal ws_in(workspace + ws_row * TILE_SIZE);
    TLOAD(c_ub, ws_in);

    HalfTileGlobal d_global(D + row_v * TILE_SIZE);
    TLOAD(d_ub, d_global);

    pipe_barrier(
        PIPE_ALL);  // MTE2→V+MTE3: both TLOADs done before signal and TADD

    // Signal Cube: workspace slot is consumed — safe to overwrite next round.
    SetCrossFlag<PIPE_MTE3>(FLAG_V2C);

    // C = GEMM_result + D  (Vec engine, element-wise)
    TADD(c_ub, c_ub, d_ub);
    pipe_barrier(PIPE_ALL);  // V→MTE3: TADD done before TSTORE

    // Store result to global memory C.
    HalfTileGlobal c_out(C + row_v * TILE_SIZE);
    TSTORE(c_out, c_ub);
    pipe_barrier(PIPE_ALL);  // MTE3: TSTORE complete before next round
  }

#endif  // __DAV_C220_VEC__
}

#endif  // __CCE_AICORE__

// ── Kernel entry point
// ─────────────────────────────────────────────────────────
extern "C" __global__ AICORE void matmul_add_c2v_kernel(
    __gm__ uint8_t *A, __gm__ uint8_t *B, __gm__ uint8_t *C, __gm__ uint8_t *D,
    __gm__ uint8_t *workspace, int64_t batch, uint64_t ffts_addr) {
  run_matmul_add_c2v(
      reinterpret_cast<__gm__ half *>(A), reinterpret_cast<__gm__ half *>(B),
      reinterpret_cast<__gm__ half *>(C), reinterpret_cast<__gm__ half *>(D),
      reinterpret_cast<__gm__ half *>(workspace), batch, ffts_addr);
}

// ── Host-side launcher (called from Python via ctypes)
// ─────────────────────────
extern "C" void call(uint32_t block_dim, void *stream, uint8_t *A, uint8_t *B,
                     uint8_t *C, uint8_t *D, uint8_t *workspace,
                     int64_t batch) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  matmul_add_c2v_kernel<<<block_dim, nullptr, stream>>>(A, B, C, D, workspace,
                                                        batch, ffts_addr);
}
