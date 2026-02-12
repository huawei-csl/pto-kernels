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

template <typename InputT, typename OutputT, uint32_t MatrixSize>
AICORE void runKernelBatchMatrixSquare(__gm__ OutputT* z, __gm__ InputT* x) {
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

  GlobalTensorIn x_global_in(x + global_index);
  GlobalTensorOut z_global_out(z + global_index);
  TileL1AB ab_l1_tile;
  TileL0A a_l0_tile;
  TileL0B b_l0_tile;
  TileL0C c_l0_tile;

  TASSIGN(ab_l1_tile, 0x0);

  TASSIGN(a_l0_tile, 0x0);
  TASSIGN(b_l0_tile, 0x0);
  TASSIGN(c_l0_tile, 0x0);

  // LOAD GM -> L1 (MTE2)
  TLOAD(ab_l1_tile, x_global_in);
  set_flag(PIPE_MTE2, PIPE_MTE1,
           EVENT_ID0);  // MTE2 pipe sets flag for MTE1 pipe
  wait_flag(PIPE_MTE2, PIPE_MTE1,
            EVENT_ID0);  // MTE1 pipe waits for MTE2 to set flag

  // LOAD L1 -> L0 (MTE1)
  TEXTRACT(a_l0_tile, ab_l1_tile, 0, 0);
  TEXTRACT(b_l0_tile, ab_l1_tile, 0, 0);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);  // MTE1 pipe sets flag for M pipe
  wait_flag(PIPE_MTE1, PIPE_M,
            EVENT_ID0);  // M pipe waits for MTE1 pipe to set flag

  // MATMUL (M)
  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);  // M pipe sets flag for FIX pipe
  wait_flag(PIPE_M, PIPE_FIX,
            EVENT_ID0);  // FIX pipe waits for M pipe to set flag
  TSTORE(z_global_out, c_l0_tile);
#else
// Nothing to do on AIV
#endif
}

template <typename InputT>
AICORE void run_batch_matrix_square(__gm__ float* z, __gm__ InputT* x,
                                    uint32_t matrix_size) {
  static_assert(std::is_same_v<InputT, half> or std::is_same_v<InputT, float>,
                "batch_matrix_square supports only fp16/fp32.");
  switch (matrix_size) {
    case 16:
      runKernelBatchMatrixSquare<InputT, float, 16>(z, x);
      break;
    case 32:
      runKernelBatchMatrixSquare<InputT, float, 32>(z, x);
      break;
    case 64:
      runKernelBatchMatrixSquare<InputT, float, 64>(z, x);
      break;
    case 96:
      runKernelBatchMatrixSquare<InputT, float, 96>(z, x);
      break;
    case 128:
      runKernelBatchMatrixSquare<InputT, float, 128>(z, x);
      break;
  }
}

extern "C" __global__ AICORE void batch_matrix_square_fp16(
    __gm__ void* z, __gm__ void* x, uint32_t matrix_size) {
  run_batch_matrix_square<half>((__gm__ float*)z, (__gm__ half*)x, matrix_size);
}

extern "C" __global__ AICORE void batch_matrix_square_fp32(
    __gm__ void* z, __gm__ void* x, uint32_t matrix_size) {
  run_batch_matrix_square<float>((__gm__ float*)z, (__gm__ float*)x,
                                 matrix_size);
}
