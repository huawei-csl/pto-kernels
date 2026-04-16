/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef MEMORY_BASE
#define MEMORY_BASE
#endif
#include <pto/pto-inst.hpp>

#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"

using namespace pto;

/**
 * @brief Triangular inverse using Newton–Schulz iterations.
 *
 * Implements the following algorithm:
 * A = I + M
 * X = I * scale
 * for _ in range(num_iters):
 *     Y = X @ (-A)
 *     X = Y @ X + 2 * X
 * return X
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize>
AICORE void runKernelTriInvNS(__gm__ OutputT* M_inv, __gm__ InputT* M,
                              __gm__ InputT* I_neg, __gm__ InputT* I_over_n,
                              uint32_t num_iters) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))  // Cube compilation

  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  const uint32_t global_index = get_block_idx() * TileLen;
  constexpr uint32_t NumL0Buffers = 2;

  /* Global Memory / Tensors */
  using TensorShapeInND =
      TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using TensorStridesInND =
      BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTensorIn =
      GlobalTensor<InputT, TensorShapeInND, TensorStridesInND, Layout::ND>;

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
  GlobalTensorIn I_over_n_global_in(I_over_n);
  GlobalTensorOut M_inv_global_out(M_inv + global_index);

  TileL1AB A_neg_l1_tile;
  TileL1AB X_l1_tile;
  TileL1AB Y_l1_tile;
  TileL1AB I_neg_l1_tile;
  TileL1AB I_l1_tile;
  TileL1AB two_I_l1_tile;

  TileL0A a_l0_tile[NumL0Buffers];
  TileL0B b_l0_tile[NumL0Buffers];
  TileL0C c_l0_tile[NumL0Buffers];

  TASSIGN(A_neg_l1_tile, 0x0);
  TASSIGN(X_l1_tile, 0x0 + TileLen * sizeof(InputT));
  TASSIGN(Y_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));
  TASSIGN(I_neg_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
  TASSIGN(I_l1_tile, 0x0 + 4 * TileLen * sizeof(InputT));
  TASSIGN(two_I_l1_tile, 0x0 + 5 * TileLen * sizeof(InputT));

  for (uint32_t buffer_num = 0; buffer_num < NumL0Buffers; ++buffer_num) {
    TASSIGN(a_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(b_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(c_l0_tile[buffer_num],
            0x0 + buffer_num * TileLen * sizeof(OutputT));
  }

  // LOAD GM -> L1 (MTE2)
  TLOAD(I_neg_l1_tile, I_neg_global_in);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  TLOAD(A_neg_l1_tile, M_global_in);
  TLOAD(X_l1_tile, I_over_n_global_in);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

  // Precompute I and store to L1
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(a_l0_tile[0], I_neg_l1_tile);
  TMOV(b_l0_tile[0], I_neg_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(a_l0_tile[1], A_neg_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(c_l0_tile[1], a_l0_tile[0], b_l0_tile[0]);  // c_l0[1] = I
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMOV(I_l1_tile, c_l0_tile[1]);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  TMATMUL(c_l0_tile[0], a_l0_tile[1], b_l0_tile[0]);  // c_l0[0] = -M
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  TMOV(a_l0_tile[1], I_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TMATMUL_ACC(c_l0_tile[1], c_l0_tile[1], a_l0_tile[0],
              b_l0_tile[0]);  // c_l0[1] <- 2I
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMOV(two_I_l1_tile, c_l0_tile[1]);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[1],
              b_l0_tile[0]);  // c_l0[0] = -M-I = -A
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);

  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TMOV(A_neg_l1_tile, c_l0_tile[0]);  // A_l1 = -A
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  TMOV(a_l0_tile[1], two_I_l1_tile);  // a_l0[1] <- 2I
  TMOV(b_l0_tile[1], A_neg_l1_tile);  // b_l0[1] <- -A
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

  for (uint32_t i = 0; i < num_iters; ++i) {
    TMOV(b_l0_tile[0], X_l1_tile);
    TMOV(a_l0_tile[0], X_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[1]);  // c_l0[0] <- -XA
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMOV(Y_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

    TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[0]);  // c_l0[1] <- 2X
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    TMOV(a_l0_tile[0], Y_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL_ACC(c_l0_tile[1], c_l0_tile[1], a_l0_tile[0], b_l0_tile[0]);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    if (i < num_iters - 1) {
      TMOV(X_l1_tile, c_l0_tile[1]);  // X_l1 now contains X_new
      set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    }
  }
  TSTORE(M_inv_global_out, c_l0_tile[1]);
#else
// Nothing to do on AIV
#endif
}

template <typename InputT>
AICORE void run_tri_inv_ns(__gm__ float* tensor_out, __gm__ InputT* tensor_in,
                           __gm__ InputT* identity_neg_in,
                           __gm__ InputT* identity_over_n_in,
                           uint32_t matrix_size, uint32_t num_iters) {
  static_assert(std::is_same_v<InputT, half>, "tri_inv_ns supports only fp16.");
  switch (matrix_size) {
    case 16:
      runKernelTriInvNS<InputT, float, 16>(tensor_out, tensor_in,
                                           identity_neg_in, identity_over_n_in,
                                           num_iters);
      break;
    case 32:
      runKernelTriInvNS<InputT, float, 32>(tensor_out, tensor_in,
                                           identity_neg_in, identity_over_n_in,
                                           num_iters);
      break;
    case 64:
      runKernelTriInvNS<InputT, float, 64>(tensor_out, tensor_in,
                                           identity_neg_in, identity_over_n_in,
                                           num_iters);
      break;
    case 96:
      runKernelTriInvNS<InputT, float, 96>(tensor_out, tensor_in,
                                           identity_neg_in, identity_over_n_in,
                                           num_iters);
      break;
    case 128:
      runKernelTriInvNS<InputT, float, 128>(tensor_out, tensor_in,
                                            identity_neg_in, identity_over_n_in,
                                            num_iters);
      break;
  }
}

extern "C" __global__ AICORE void tri_inv_ns_fp16(
    __gm__ void* tensor_out, __gm__ void* tensor_in,
    __gm__ void* identity_neg_in, __gm__ void* identity_over_n_in,
    uint32_t matrix_size, uint32_t num_iters) {
  run_tri_inv_ns<half>((__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
                       (__gm__ half*)identity_neg_in,
                       (__gm__ half*)identity_over_n_in, matrix_size,
                       num_iters);
}
