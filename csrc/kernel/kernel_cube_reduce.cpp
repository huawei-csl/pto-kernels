/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

// ============================================================================
// kernel_cube_reduce.cpp — Block reduction via Cube matmul + Vector column sum
//
// Algorithm (two-phase, adapted from AscendC KernelCubeReduce):
//
//   Phase 1 (AIC / Cube cores):
//     Each AIC block owns a contiguous slice of the input vector.  The slice
//     is viewed as consecutive S×S tiles.  Each tile A is multiplied by an
//     S×16 all-ones matrix B:
//
//         C = A @ B      (C ∈ ℝ^{S×16}, B ∈ ℝ^{S×16} all ones)
//
//     Because B is all-ones, every column j of C satisfies
//
//         C_{i,j} = Σ_k A_{i,k}  (row-sum of A)
//
//     so all 16 columns of C are identical, each containing the row-sums
//     of A.  Tiles are accumulated into a single running L0C accumulator,
//     so after the loop L0C = Σ_t A_t @ B, where the Σ is over all S×S
//     tiles owned by this AIC block.
//
//     The S×16 L0C result is then stored to workspace at offset
//     block_id * S * 16.
//
//   Cross-core barrier (kernel_utils::SyncAll<false>()):
//     Ensures all AIC writes to workspace are globally visible before any
//     AIV core reads workspace.
//
//   Phase 2 (AIV / Vector cores):
//     Each AIV block reads the S×16 workspace tile written by the
//     corresponding AIC block and sums column 0 (elements at indices
//     0, 16, 32, …, (S-1)*16).  Since all 16 columns are equal, this
//     gives the total sum of the input slice owned by that AIC block.
//
// Supported dtypes:  fp16 → float32
// Supported matmul sizes: 16, 32, 64, 128
// ============================================================================

#include "kernel_utils.h"

using namespace pto;

/**
 * @brief Phase 1 (AIC): reduce a slice of the input vector via Cube matmul.
 *
 * @tparam InputT  Input element type (half).
 * @tparam OutputT Accumulator type (float for half input).
 * @tparam S       Square matmul tile side length.
 *
 * @param vec_in     Input 1D vector in GM.
 * @param all_ones_b S×16 all-ones matrix in GM with dtype InputT.
 * @param workspace  Intermediate output in GM: block_num × S × 16 OutputT.
 * @param vec_len    Total number of input elements (must be divisible by S*S).
 * @param block_num  Number of AIC blocks launched.
 */
template <typename InputT, typename OutputT, uint32_t S>
AICORE void runKernelCubeReduce(__gm__ InputT* vec_in,
                                __gm__ InputT* all_ones_b,
                                __gm__ OutputT* workspace, uint32_t vec_len,
                                uint32_t block_num) {
  constexpr uint32_t MAT_DIM_16 = 16;
  constexpr uint32_t tile_len = S * S;
  constexpr uint32_t out_tile_len = S * MAT_DIM_16;

  const uint32_t id = get_block_idx();
  const uint32_t num_tiles = vec_len / tile_len;
  const uint32_t tiles_per_block = (num_tiles + block_num - 1) / block_num;
  const uint32_t start_tile = id * tiles_per_block;
  const uint32_t end_tile = min(start_tile + tiles_per_block, num_tiles);

  if (start_tile >= num_tiles) return;

  // ── Global memory tensor types ───────────────────────────────────────────
  using TensorShapeA = TileShape2D<InputT, S, S, Layout::ND>;
  using TensorStridesA = BaseShape2D<InputT, S, S, Layout::ND>;
  using GlobalDataA =
      GlobalTensor<InputT, TensorShapeA, TensorStridesA, Layout::ND>;

  using TensorShapeB = TileShape2D<InputT, S, MAT_DIM_16, Layout::ND>;
  using TensorStridesB = BaseShape2D<InputT, S, MAT_DIM_16, Layout::ND>;
  using GlobalDataB =
      GlobalTensor<InputT, TensorShapeB, TensorStridesB, Layout::ND>;

  using TensorShapeOut = TileShape2D<OutputT, S, MAT_DIM_16, Layout::ND>;
  using TensorStridesOut = BaseShape2D<OutputT, S, MAT_DIM_16, Layout::ND>;
  using GlobalDataOut =
      GlobalTensor<OutputT, TensorShapeOut, TensorStridesOut, Layout::ND>;

  // ── L1 tile types ─────────────────────────────────────────────────────────
  // A tile: S×S input matrix residing in L1
  using TileL1A = Tile<TileType::Mat, InputT, S, S, BLayout::ColMajor, S, S,
                       SLayout::RowMajor, 512>;
  // B tile: S×16 all-ones matrix residing in L1
  using TileL1B = Tile<TileType::Mat, InputT, S, MAT_DIM_16, BLayout::ColMajor,
                       S, MAT_DIM_16, SLayout::RowMajor, 512>;

  // ── L0 tile types ─────────────────────────────────────────────────────────
  using TileL0A = TileLeft<InputT, S, S>;            // M×K = S×S
  using TileL0B = TileRight<InputT, S, MAT_DIM_16>;  // K×N = S×16
  using TileL0C = TileAcc<OutputT, S, MAT_DIM_16>;   // M×N = S×16

  // ── L1 memory layout (A and B placed back-to-back) ───────────────────────
  constexpr uint32_t a_l1_bytes = S * S * sizeof(InputT);
  TileL1A a_l1;
  TileL1B b_l1;
  TASSIGN(a_l1, 0x0);
  TASSIGN(b_l1, 0x0 + a_l1_bytes);

  // ── L0A / L0B / L0C occupy independent scratchpads ───────────────────────
  TileL0A a_l0;
  TileL0B b_l0;
  TileL0C c_l0;
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(c_l0, 0x0);

  // ── Load all-ones B once: GM → L1 → L0B ─────────────────────────────────
  GlobalDataB b_global(all_ones_b);
  TLOAD(b_l1, b_global);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(b_l0, b_l1);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  // ── Tile loop: for each S×S input tile, multiply by B and accumulate ─────
  for (uint32_t t = start_tile; t < end_tile; t++) {
    // GM → L1A
    GlobalDataA a_global(vec_in + t * tile_len);
    TLOAD(a_l1, a_global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

    // L1A → L0A
    TMOV(a_l0, a_l1);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

    if (t == start_tile) {
      // First tile: fresh matmul, C = A @ B
      TMATMUL(c_l0, a_l0, b_l0);
    } else {
      // Subsequent tiles: accumulate, C += A @ B
      pipe_barrier(PIPE_M);
      TMATMUL_ACC(c_l0, c_l0, a_l0, b_l0);
    }
  }

  // ── Store accumulated L0C → workspace[block_id * S * 16] ─────────────────
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  GlobalDataOut out_global(workspace + id * out_tile_len);
  TSTORE(out_global, c_l0);
}

/**
 * @brief Phase 2 (AIV): sum column 0 of each block's S×16 intermediate result.
 *
 * After `runKernelCubeReduce`, workspace[b*S*16 .. (b+1)*S*16-1] holds the
 * S×16 reduction matrix for block b.  Because all 16 columns are equal,
 * column 0 (elements at index i*16 within the tile) already contains the
 * per-row sums; summing those S values gives the total input sum for block b.
 *
 * @tparam OutputT Workspace/output element type (float for fp16 input).
 * @tparam S       Matmul tile side length used in Phase 1.
 *
 * @param workspace  Intermediate results: block_num × S × 16 elements.
 * @param vec_out    Output: one scalar per block (length block_num).
 * @param block_num  Number of blocks.
 */
template <typename OutputT, uint32_t S>
AICORE void runKernelCompleteCubeReduce(__gm__ OutputT* workspace,
                                        __gm__ OutputT* vec_out,
                                        uint32_t block_num) {
#if defined(__DAV_VEC__)
  constexpr uint32_t MAT_DIM_16 = 16;

  const uint32_t id = get_block_idx();
  // Only the first AIV sub-block of each AI core does work; the second idles.
  if (get_subblockid() != 0 || id >= block_num) return;

  // Sum column 0: element i of column 0 lives at index i*MAT_DIM_16 in the
  // flat S*16 tile for this block.
  const uint32_t gm_offset = id * S * MAT_DIM_16;
  OutputT sum = static_cast<OutputT>(0);
  for (uint32_t i = 0; i < S; i++) {
    sum += workspace[gm_offset + i * MAT_DIM_16];
  }

  vec_out[id] = sum;
#endif
}

// ── Per-size dispatcher helpers
// ───────────────────────────────────────────────

template <typename InputT, typename OutputT, uint32_t S>
AICORE void run_cube_reduce_impl(__gm__ InputT* vec_in,
                                 __gm__ InputT* all_ones_b,
                                 __gm__ OutputT* workspace,
                                 __gm__ OutputT* vec_out, uint32_t vec_len,
                                 uint32_t block_num) {
#if defined(__DAV_CUBE__)
  runKernelCubeReduce<InputT, OutputT, S>(vec_in, all_ones_b, workspace,
                                          vec_len, block_num);
  // Barrier: ensure workspace writes are globally visible before AIV reads.
  kernel_utils::SyncAll<false>();
#elif defined(__DAV_VEC__)
  // Wait for AIC to finish writing to workspace.
  kernel_utils::SyncAll<false>();
  runKernelCompleteCubeReduce<OutputT, S>(workspace, vec_out, block_num);
#endif
}

template <typename InputT, typename OutputT>
AICORE void run_cube_reduce(__gm__ InputT* vec_in, __gm__ InputT* all_ones_b,
                            __gm__ OutputT* workspace, __gm__ OutputT* vec_out,
                            uint32_t vec_len, uint32_t block_num,
                            uint32_t matmul_size) {
  switch (matmul_size) {
    case 16:
      run_cube_reduce_impl<InputT, OutputT, 16>(vec_in, all_ones_b, workspace,
                                                vec_out, vec_len, block_num);
      break;
    case 32:
      run_cube_reduce_impl<InputT, OutputT, 32>(vec_in, all_ones_b, workspace,
                                                vec_out, vec_len, block_num);
      break;
    case 64:
      run_cube_reduce_impl<InputT, OutputT, 64>(vec_in, all_ones_b, workspace,
                                                vec_out, vec_len, block_num);
      break;
    case 128:
      run_cube_reduce_impl<InputT, OutputT, 128>(vec_in, all_ones_b, workspace,
                                                 vec_out, vec_len, block_num);
      break;
    default:
      break;
  }
}

// ── Kernel entry points
// ───────────────────────────────────────────────────────

extern "C" __global__ AICORE void cube_reduce_fp16(
    GM_ADDR vec_in, GM_ADDR all_ones_b, GM_ADDR workspace, GM_ADDR vec_out,
    uint32_t vec_len, uint32_t block_num, uint32_t matmul_size) {
  run_cube_reduce<half, float>((__gm__ half*)vec_in, (__gm__ half*)all_ones_b,
                               (__gm__ float*)workspace, (__gm__ float*)vec_out,
                               vec_len, block_num, matmul_size);
}
