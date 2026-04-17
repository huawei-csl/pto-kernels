/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include "kernel_utils.h"

using namespace pto;

/**
 * @brief: Prepares Identity and 2*Identity matrices.
 *
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 *
 * @param I_neg_l1_tile Tile containing the -I (negative identity) matrix.
 * @param I_l1_tile Tile to store the identity matrix.
 * @param two_I_l1_tile Tile to store the 2 * identity matrix.
 * @param a_l0_tile Tile in L0A for matmuls.
 * @param b_l0_tile Tile in L0B for matmuls.
 * @param c_l0_tile_0 Tile in L0C for matmuls (first buffer).
 * @param c_l0_tile_1 Tile in L0C for matmuls (second buffer).
 */
template <typename TileL1AB, typename TileL0A, typename TileL0B,
          typename TileL0C>
AICORE inline void PrepareAuxiliaryMatrices(
    TileL1AB I_neg_l1_tile, TileL1AB I_l1_tile, TileL1AB two_I_l1_tile,
    TileL0A a_l0_tile, TileL0B b_l0_tile, TileL0C c_l0_tile_0,
    TileL0C c_l0_tile_1) {
  // Precompute I and store to L1
  TMOV(a_l0_tile, I_neg_l1_tile);
  TMOV(b_l0_tile, I_neg_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(c_l0_tile_0, a_l0_tile, b_l0_tile);  // c_l0_0 = I
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMATMUL(c_l0_tile_1, a_l0_tile, b_l0_tile);      // c_l0_1 = I
  TMATMUL_ACC(c_l0_tile_1, a_l0_tile, b_l0_tile);  // c_l0_1 = 2*I
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMOV(I_l1_tile, c_l0_tile_0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TMOV(two_I_l1_tile, c_l0_tile_1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID1);

  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID1);
}

/**
 * @brief: Inverts a single matrix / tile of the global tensor.
 *
 * @tparam InputT The type of the input elements.
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam NumTilesPerCubeIter How many matrices to load and invert in a single
 * cube iteration.
 *
 * @param X_l1_tile Tile in L1 used for intermediate computations.
 * @param I_l1_tile Tile containing the identity matrix.
 * @param two_I_l1_tile Tile containing the 2 * identity matrix.
 * @param I_neg_l1_tile Tile containing the negative identity matrix.
 * @param I_scaled_l1_tile Tile containing the matrix I / (2 * n).
 * @param A_neg_l1_tile Tile containing the negative input matrix.
 * @param Y_l1_tile Tile in L1 used for intermediate computations.
 * @param a_l0_tile* Array of two tiles in L0A (for double-buffering).
 * @param b_l0_tile* Array of two tiles in L0B (for double-buffering).
 * @param c_l0_tile* Tile in L0C for matmuls.
 * @param tile_id Index of the current tile (used for sync).
 */
template <typename InputT, typename TileL1AB, typename TileL0A,
          typename TileL0B, typename TileL0C, uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter>
AICORE inline void InvertSingleTile(
    TileL1AB X_l1_tile, TileL1AB I_l1_tile, TileL1AB two_I_l1_tile,
    TileL1AB I_neg_l1_tile, TileL1AB I_scaled_l1_tile, TileL1AB A_neg_l1_tile,
    TileL1AB Y_l1_tile, TileL0A* a_l0_tile, TileL0B* b_l0_tile,
    TileL0C* c_l0_tile, const uint32_t num_iters, const uint32_t tile_id) {
  const event_t event_0 = static_cast<event_t>(tile_id);
  const event_t event_1 = static_cast<event_t>(tile_id + NumTilesPerCubeIter);

  // a_l0[0] will store X and then Y
  // a_l0[1] stores 2I (always)
  // b_l0[0] will store -A = -I-M
  // b_l0[1] will store X (where initially X0 = I / (2 * n))

  TMOV(a_l0_tile[0], I_neg_l1_tile);
  TMOV(b_l0_tile[0], A_neg_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, event_0);

  TMOV(b_l0_tile[1], I_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, event_1);

  wait_flag(PIPE_MTE1, PIPE_M, event_0);
  TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // c_l0[0] <- -M

  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0],
              b_l0_tile[1]);  // c_l0[0] <- -I-M = -A
  set_flag(PIPE_M, PIPE_FIX, event_0);
  set_flag(PIPE_M, PIPE_MTE1, event_0);

  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  TMOV(a_l0_tile[1], two_I_l1_tile);  // a_l0[1] <- 2 * I (will stay constant)
  TMOV(b_l0_tile[1], I_scaled_l1_tile);  // b_l0[1] <- X0 = I / (2 * n)
  TMOV(a_l0_tile[0], I_scaled_l1_tile);  // a_l0[0] <- X0 = I / (2 * n)
  set_flag(PIPE_MTE1, PIPE_M, event_0);
  wait_flag(PIPE_MTE1, PIPE_M, event_0);

  wait_flag(PIPE_M, PIPE_FIX, event_0);
  TMOV(A_neg_l1_tile, c_l0_tile[0]);
  set_flag(PIPE_FIX, PIPE_MTE1, event_0);
  wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
  TMOV(b_l0_tile[0], A_neg_l1_tile);  // b_l0[0] <- -A (will stay constant)
  set_flag(PIPE_MTE1, PIPE_M, event_1);
  wait_flag(PIPE_MTE1, PIPE_M, event_1);

  set_flag(PIPE_FIX, PIPE_M, event_1);
  for (uint32_t i = 0; i < num_iters; ++i) {
    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // c_l0[0] <- X @ (-A)
    set_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_0);

    wait_flag(PIPE_M, PIPE_FIX, event_0);
    TMOV(Y_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);

    wait_flag(PIPE_FIX, PIPE_M, event_1);
    TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]);  // c_l0[1] <- 2 * X

    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
    TMOV(a_l0_tile[0], Y_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL_ACC(c_l0_tile[1], c_l0_tile[1], a_l0_tile[0],
                b_l0_tile[1]);  // c_l0[1] <- Y @ X + 2 * X
    set_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);

    if (i < num_iters - 1) {
      set_flag(PIPE_M, PIPE_MTE1, event_1);
      TMOV(X_l1_tile, c_l0_tile[1]);        // X_l1 now contains X_new
      set_flag(PIPE_FIX, PIPE_M, event_1);  // for next iter
      set_flag(PIPE_FIX, PIPE_MTE1, event_0);

      wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
      wait_flag(PIPE_M, PIPE_MTE1, event_1);
      TMOV(b_l0_tile[1], X_l1_tile);
      TMOV(a_l0_tile[0], X_l1_tile);
      set_flag(PIPE_MTE1, PIPE_M, event_0);
      wait_flag(PIPE_MTE1, PIPE_M, event_0);
    }
  }
}

/**
 * @brief Triangular inverse using Newton–Schulz iterations.
 *
 * Implements the following algorithm:
 * A = I + M
 * X = I / (2 * MatrixSize)
 * for _ in range(num_iters):
 *     Y = X @ (-A)
 *     X = Y @ X + 2 * X
 * return X
 * @tparam InputT The type of the input elements.
 * @tparam OutputT The type of the output elements.
 * @tparam MatrixSize Size of the entire input/output matrices.
 *
 * @param M_inv pointer to the global memory to store the final inverse.
 * @param M Pointer to the global tensor matrix in global memory.
 * @param I_neg Pointer to global memory that contains the negative identity.
 * @param I_scaled Pointer to global memory containing the identity scaled by:
 * 1 / (2 * MatrixSize).
 * @param num_iters Number of Newton-Schulz iterations.
 * @param total_tiles The total number of matrices in the tensor to invert.
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter, bool IsBSND = false>
AICORE void runKernelTriInvNS(__gm__ OutputT* M_inv, __gm__ InputT* M,
                              __gm__ InputT* I_neg, __gm__ InputT* I_scaled,
                              uint32_t num_iters, uint32_t total_tiles) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))  // Cube compilation

  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  const uint32_t global_index = get_block_idx() * TileLen;
  constexpr uint32_t NumL0Buffers = 2;
  const uint32_t max_iters_per_aic = kernel_utils::CeilDiv(
      total_tiles, (uint32_t)(NumTilesPerCubeIter * get_block_num()));

  /* Global Memory / Tensors */
  using TensorShapeIn = TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using TensorStridesIn =
      BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTensorIn =
      GlobalTensor<InputT, TensorShapeIn, TensorStridesIn, Layout::ND>;

  using TensorShapeOut =
      TileShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>;
  using TensorStridesOut =
      BaseShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTensorOut =
      GlobalTensor<OutputT, TensorShapeOut, TensorStridesOut, Layout::ND>;

  /* L1 Memory */
  using TileL1AB =
      Tile<TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor,
           MatrixSize, MatrixSize, SLayout::RowMajor, 512>;

  /* L0 Memory */
  using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
  using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
  using TileL0C = TileAcc<OutputT, MatrixSize, MatrixSize>;

  GlobalTensorIn M_global_in(M + global_index);
  GlobalTensorIn I_neg_global_in(I_neg);
  GlobalTensorIn I_scaled_global_in(I_scaled);
  GlobalTensorOut M_inv_global_out(M_inv + global_index);

  TileL1AB A_neg_l1_tile[NumTilesPerCubeIter];
  TileL1AB X_l1_tile;
  TileL1AB Y_l1_tile;
  TileL1AB I_neg_l1_tile;
  TileL1AB I_l1_tile;
  TileL1AB two_I_l1_tile;
  TileL1AB I_scaled_l1_tile;

  TileL0A a_l0_tile[NumL0Buffers];
  TileL0B b_l0_tile[NumL0Buffers];
  TileL0C c_l0_tile[NumL0Buffers];

  TASSIGN(X_l1_tile, 0x0);
  TASSIGN(Y_l1_tile, 0x0 + TileLen * sizeof(InputT));
  TASSIGN(I_neg_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));
  TASSIGN(I_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
  TASSIGN(two_I_l1_tile, 0x0 + 4 * TileLen * sizeof(InputT));
  TASSIGN(I_scaled_l1_tile, 0x0 + 5 * TileLen * sizeof(InputT));
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    TASSIGN(A_neg_l1_tile[tile_id],
            0x0 + (6 + tile_id) * TileLen * sizeof(InputT));
  }

  for (uint32_t buffer_num = 0; buffer_num < NumL0Buffers; ++buffer_num) {
    TASSIGN(a_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(b_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(c_l0_tile[buffer_num],
            0x0 + buffer_num * TileLen * sizeof(OutputT));
  }

  // LOAD GM -> L1 (MTE2)
  TLOAD(I_neg_l1_tile, I_neg_global_in);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  TLOAD(I_scaled_l1_tile, I_scaled_global_in);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  PrepareAuxiliaryMatrices<TileL1AB, TileL0A, TileL0B, TileL0C>(
      I_neg_l1_tile, I_l1_tile, two_I_l1_tile, a_l0_tile[0], b_l0_tile[0],
      c_l0_tile[0], c_l0_tile[1]);

  uint32_t next_tile_id_that_waits_for_pipe_fix_pipe_m = 0;
  set_flag(PIPE_FIX, PIPE_M,
           static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
  }
  for (uint32_t cube_iter = 0; cube_iter < max_iters_per_aic; ++cube_iter) {
    const uint32_t global_index =
        (cube_iter * get_block_num() + get_block_idx()) * NumTilesPerCubeIter;
    if (global_index >= total_tiles) {
      break;
    }
    for (uint32_t tile_id = 0; (tile_id < NumTilesPerCubeIter) &&
                               (global_index + tile_id < total_tiles);
         ++tile_id) {
      if constexpr (IsBSND) {
        /* TODO */
      } else {
        GlobalTensorIn M_global_in(M + (global_index + tile_id) * TileLen);
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        TLOAD(A_neg_l1_tile[tile_id],
              M_global_in);  // Copies NumTilesPerCubeIter tiles at once
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
    }

    for (uint32_t tile_id = 0; (tile_id < NumTilesPerCubeIter) &&
                               (global_index + tile_id < total_tiles);
         ++tile_id) {
      // Wait for previous cube iter to write result
      wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(tile_id));
      // Wait for loading new matrices from GM
      wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
      InvertSingleTile<InputT, TileL1AB, TileL0A, TileL0B, TileL0C, MatrixSize,
                       NumTilesPerCubeIter>(
          X_l1_tile, I_l1_tile, two_I_l1_tile, I_neg_l1_tile, I_scaled_l1_tile,
          A_neg_l1_tile[tile_id], Y_l1_tile, a_l0_tile, b_l0_tile, c_l0_tile,
          num_iters, tile_id);

      // Allow next cube_iter to proceed for this tile_id
      set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
      if constexpr (IsBSND) {
        /* TODO */
      } else {
        GlobalTensorOut M_inv_global_out(M_inv +
                                         (global_index + tile_id) * TileLen);
        TSTORE(M_inv_global_out, c_l0_tile[1]);
      }
      next_tile_id_that_waits_for_pipe_fix_pipe_m =
          (tile_id + 1) % NumTilesPerCubeIter;
      set_flag(
          PIPE_FIX, PIPE_M,
          static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
    }
  }
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
  }
  wait_flag(PIPE_FIX, PIPE_M,
            static_cast<event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));

#else
// Nothing to do on AIV
#endif
}

template <typename InputT, uint32_t NumTilesPerCubeIter>
AICORE void run_tri_inv_ns(__gm__ float* tensor_out, __gm__ InputT* tensor_in,
                           __gm__ InputT* identity_neg_in,
                           __gm__ InputT* identity_over_n_in,
                           uint32_t matrix_size, uint32_t num_iters,
                           uint32_t num_matrices) {
  static_assert(std::is_same_v<InputT, half>, "tri_inv_ns supports only fp16.");
  switch (matrix_size) {
    case 16:
      runKernelTriInvNS<InputT, float, 16, NumTilesPerCubeIter>(
          tensor_out, tensor_in, identity_neg_in, identity_over_n_in, num_iters,
          num_matrices);
      break;
    case 32:
      runKernelTriInvNS<InputT, float, 32, NumTilesPerCubeIter>(
          tensor_out, tensor_in, identity_neg_in, identity_over_n_in, num_iters,
          num_matrices);
      break;
    case 64:
      runKernelTriInvNS<InputT, float, 64, NumTilesPerCubeIter>(
          tensor_out, tensor_in, identity_neg_in, identity_over_n_in, num_iters,
          num_matrices);
      break;
    case 96:
      runKernelTriInvNS<InputT, float, 96, NumTilesPerCubeIter>(
          tensor_out, tensor_in, identity_neg_in, identity_over_n_in, num_iters,
          num_matrices);
      break;
    case 128:
      runKernelTriInvNS<InputT, float, 128, NumTilesPerCubeIter>(
          tensor_out, tensor_in, identity_neg_in, identity_over_n_in, num_iters,
          num_matrices);
      break;
  }
}

extern "C" __global__ AICORE void tri_inv_ns_fp16(
    __gm__ void* tensor_out, __gm__ void* tensor_in,
    __gm__ void* identity_neg_in, __gm__ void* identity_over_n_in,
    uint32_t matrix_size, uint32_t num_iters, uint32_t num_matrices) {
  if (num_matrices <= get_block_num()) {
    run_tri_inv_ns<half, 1>((__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
                            (__gm__ half*)identity_neg_in,
                            (__gm__ half*)identity_over_n_in, matrix_size,
                            num_iters, num_matrices);
  } else if (num_matrices <= 2 * get_block_num()) {
    run_tri_inv_ns<half, 2>((__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
                            (__gm__ half*)identity_neg_in,
                            (__gm__ half*)identity_over_n_in, matrix_size,
                            num_iters, num_matrices);
  } else {
    run_tri_inv_ns<half, 4>((__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
                            (__gm__ half*)identity_neg_in,
                            (__gm__ half*)identity_over_n_in, matrix_size,
                            num_iters, num_matrices);
  }
}
