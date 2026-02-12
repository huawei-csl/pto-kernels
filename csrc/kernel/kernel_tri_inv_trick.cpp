/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#define MEMORY_BASE
#include <pto/pto-inst.hpp>

#include "kernel_operator.h"
using namespace pto;

/*
* @brief: Computes the inverse of the matrix M based on:
  X = I - M
  Y = M.copy()
  for _ in range(max_iters):
    Y = Y @ Y
    X = X + X @ Y
*/
template <typename InputT, typename OutputT, uint32_t MatrixSize>
AICORE void runKernelTriInvTrick(__gm__ OutputT* M_inv, __gm__ InputT* M,
                                 __gm__ InputT* I_neg,
                                 uint32_t max_block_size) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))  // Cube compilation

  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  const uint32_t global_index = get_block_idx() * TileLen;

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
  GlobalTensorOut M_inv_global_out(M_inv + global_index);
  TileL1AB X_l1_tile;
  TileL1AB Y_l1_tile;
  TileL1AB I_l1_tile;

  TileL0A a_l0_tile;
  TileL0B b_l0_tile;
  TileL0C c_l0_tile;

  TASSIGN(X_l1_tile, 0x0);
  TASSIGN(Y_l1_tile, 0x0 + TileLen * sizeof(InputT));
  TASSIGN(I_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));

  TASSIGN(a_l0_tile, 0x0);
  TASSIGN(b_l0_tile, 0x0);
  TASSIGN(c_l0_tile, 0x0);

  // LOAD GM -> L1 (MTE2)
  TLOAD(Y_l1_tile, M_global_in);
  TLOAD(X_l1_tile, I_neg_global_in);
  set_flag(PIPE_MTE2, PIPE_MTE1,
           EVENT_ID0);  // MTE2 pipe sets flag for MTE1 pipe
  wait_flag(PIPE_MTE2, PIPE_MTE1,
            EVENT_ID0);  // MTE1 pipe waits for MTE2 to set flag

  TMOV(a_l0_tile, Y_l1_tile);  // a_l0 contains M
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

  TMOV(b_l0_tile, Y_l1_tile);  // b_l0 also contains M
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains M^2
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  TMOV(Y_l1_tile, c_l0_tile);  // Y_l1 now contains M^2
  // set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  // wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);   // Why is this needed???
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);  // Why us this needed ???

  TMOV(b_l0_tile, X_l1_tile);  // b_l0 contains I_neg
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains M_neg
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

  TMOV(a_l0_tile, X_l1_tile);  // a_l0 contains I_neg
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile,
              b_l0_tile);  // c_l0 contains I - M
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  TMOV(X_l1_tile, c_l0_tile);  // X_l1 now contains I-M
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains I
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  TMOV(I_l1_tile, c_l0_tile);  // I_l1 now contains I
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

  for (uint32_t i = 1; i < max_block_size; i *= 2) {
    TMOV(a_l0_tile, X_l1_tile);
    TMOV(b_l0_tile, I_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains X
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);

    TMOV(b_l0_tile, Y_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile,
                b_l0_tile);  // c_l0 now contains X + X @ Y
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    if (i < max_block_size / 2) {  // Update Y except in last iteration
      TMOV(X_l1_tile, c_l0_tile);
      set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);

      TMOV(a_l0_tile, Y_l1_tile);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

      TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      TMOV(Y_l1_tile, c_l0_tile);
      set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    }
  }
  TSTORE(M_inv_global_out, c_l0_tile);
#else
// Nothing to do on AIV
#endif
}

template <typename InputT>
AICORE void run_tri_inv_trick(__gm__ float* tensor_out,
                              __gm__ InputT* tensor_in,
                              __gm__ InputT* identity_in, uint32_t matrix_size,
                              uint32_t max_block_size) {
  static_assert(std::is_same_v<InputT, half>,
                "tri_inv_trick supports only fp16.");
  switch (matrix_size) {
    case 16:
      runKernelTriInvTrick<InputT, float, 16>(tensor_out, tensor_in,
                                              identity_in, max_block_size);
      break;
    case 32:
      runKernelTriInvTrick<InputT, float, 32>(tensor_out, tensor_in,
                                              identity_in, max_block_size);
      break;
    case 64:
      runKernelTriInvTrick<InputT, float, 64>(tensor_out, tensor_in,
                                              identity_in, max_block_size);
      break;
    case 96:
      runKernelTriInvTrick<InputT, float, 96>(tensor_out, tensor_in,
                                              identity_in, max_block_size);
      break;
    case 128:
      runKernelTriInvTrick<InputT, float, 128>(tensor_out, tensor_in,
                                               identity_in, max_block_size);
      break;
  }
}

extern "C" __global__ AICORE void tri_inv_trick_fp16(__gm__ void* tensor_out,
                                                     __gm__ void* tensor_in,
                                                     __gm__ void* identity_in,
                                                     uint32_t matrix_size,
                                                     uint32_t max_block_size) {
  run_tri_inv_trick<half>((__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
                          (__gm__ half*)identity_in, matrix_size,
                          max_block_size);
}
