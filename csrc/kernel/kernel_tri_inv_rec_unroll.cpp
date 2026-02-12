/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#define MEMORY_BASE
#include <pto/pto-inst.hpp>

#include "kernel_operator.h"
using namespace pto;

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetWaitFlag(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

template <typename InputT, typename SrcL1TileT, typename DstL0TileT,
          uint32_t MatrixSize>
AICORE inline void CopyDiagonalFractalsL1ToL0(SrcL1TileT src) {
  constexpr uint32_t FractalSize = 16;
  constexpr uint32_t NumFractals = MatrixSize / FractalSize;
  DstL0TileT fractals[NumFractals];
  for (uint32_t i = 0; i < NumFractals; ++i) {
    TASSIGN(fractals[i], 0x0 + i * FractalSize * (MatrixSize + FractalSize) *
                                   sizeof(InputT));
    TEXTRACT(fractals[i], src, i * FractalSize, i * FractalSize);
  }
}

template <typename InputT, typename SrcL1TileT, typename FractalL0TileT,
          uint32_t MatrixSize>
AICORE inline void CopyOddOrEvenBlocksL1ToL0(SrcL1TileT src,
                                             uint32_t block_size,
                                             uint32_t starting_block) {
  constexpr uint32_t FractalSize = 16;
  const uint32_t num_blocks = MatrixSize / block_size;
  const uint32_t num_fractals_per_block = block_size / FractalSize;

  // might need less if block_size < FractalSize
  FractalL0TileT fractals[MatrixSize / FractalSize];
  for (uint32_t i = 0; i < num_fractals_per_block; ++i) {
    for (uint32_t j = 0; j < num_fractals_per_block; ++j) {
      for (uint32_t b = starting_block; b < num_blocks; b += 2) {
        const uint32_t offset =
            b * (MatrixSize + FractalSize) * block_size /* block_offset */ +
            i * MatrixSize * FractalSize /* col_fractal_offset */ +
            j * FractalSize * FractalSize /* row_fractal_offset */;
        TASSIGN(fractals[b], 0x0 + offset * sizeof(InputT));
        TEXTRACT(fractals[b], src, b * block_size + i * FractalSize,
                 b * block_size + j * FractalSize);
      }
    }
  }
}

/*
 * @brief: AIC kernel part
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize>
AICORE inline void RunAICKernel(__gm__ OutputT* M_inv, __gm__ InputT* M,
                                __gm__ InputT* I_neg) {
  /* Init tiles */
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  constexpr uint32_t FractalSize = 16;  // fractal size for half
  constexpr uint32_t NumFractalsRowWise = MatrixSize / FractalSize;
  const uint32_t global_index = get_block_idx() * TileLen;

  // Global Memory - Tensors
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

  // L1 Memory
  using TileL1AB =
      Tile<TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor,
           MatrixSize, MatrixSize, SLayout::RowMajor, 512>;

  // L0 Memory
  using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
  using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
  using TileL0C = TileAcc<OutputT, MatrixSize, MatrixSize>;

  GlobalTensorIn M_global_in(M + global_index);
  GlobalTensorIn I_neg_global_in(I_neg);
  GlobalTensorOut M_inv_global_out(M_inv + global_index);

  TileL1AB X_l1_tile;
  TileL1AB Y_l1_tile;
  TileL1AB I_l1_tile;
  TileL1AB M_neg_l1_tile;
  TileL1AB Zero_l1_tile;

  TileL0A a_l0_tile;
  TileL0B b_l0_tile;
  TileL0C c_l0_tile;

  TASSIGN(X_l1_tile, 0x0);
  TASSIGN(Y_l1_tile, 0x0 + TileLen * sizeof(InputT));
  TASSIGN(I_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));
  TASSIGN(M_neg_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
  TASSIGN(Zero_l1_tile, 0x0 + 4 * TileLen * sizeof(InputT));

  TASSIGN(a_l0_tile, 0x0);
  TASSIGN(b_l0_tile, 0x0);
  TASSIGN(c_l0_tile, 0x0);
  TLOAD(X_l1_tile, I_neg_global_in);
  TLOAD(Y_l1_tile, M_global_in);

  /* Initializations */
  SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);

  TMOV(a_l0_tile, X_l1_tile);  // a_l0 initialized with I_neg
  TMOV(b_l0_tile, Y_l1_tile);  // b_l0 contains M
  SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains I
  SetWaitFlag<PIPE_M, PIPE_FIX>(0);

  TMOV(M_neg_l1_tile, c_l0_tile);  // M_leg_l1 now contains I
  SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);

  TMOV(b_l0_tile, X_l1_tile);  // b_l0 initialized with I_neg
  SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains I
  SetWaitFlag<PIPE_M, PIPE_FIX>(0);

  TMOV(I_l1_tile, c_l0_tile);  // I_l1 now contains I
  SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);

  TMOV(b_l0_tile, I_l1_tile);  // b_l0 contains I
  SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile,
              b_l0_tile);  // c_l0 contains zeros
  SetWaitFlag<PIPE_M, PIPE_FIX>(0);

  TMOV(Zero_l1_tile, c_l0_tile);  // Zeros_l1 now contains zeros
  SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);

  CopyDiagonalFractalsL1ToL0<InputT, TileL1AB,
                             TileLeft<InputT, FractalSize, FractalSize>,
                             MatrixSize>(Y_l1_tile);  // a_l0 = diag_fractals(M)
  SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
  SetWaitFlag<PIPE_M, PIPE_MTE1>(0);
  CopyDiagonalFractalsL1ToL0<InputT, TileL1AB,
                             TileRight<InputT, FractalSize, FractalSize>,
                             MatrixSize>(Y_l1_tile);  // b_l0 = diag_fractals(M)
  SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
  SetWaitFlag<PIPE_M, PIPE_MTE1>(0);

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains M^2
  SetWaitFlag<PIPE_M, PIPE_FIX>(0);

  TMOV(Y_l1_tile, c_l0_tile);  // Y_l1 now contains M^2
  SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);

  TMOV(b_l0_tile, X_l1_tile);  // b_l0 contains I_neg
  SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 = diag_fractals(M_neg)
  SetWaitFlag<PIPE_M, PIPE_MTE1>(0);

  TMOV(a_l0_tile, X_l1_tile);  // a_l0 contains I_neg
  SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 has I-M
  SetWaitFlag<PIPE_M, PIPE_FIX>(0);

  TMOV(X_l1_tile, c_l0_tile);  // X_l1 now contains I-M
  SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);

  /* Inv Trick part */
  for (uint32_t i = 1; i < FractalSize / 2; i *= 2) {
    TMOV(a_l0_tile, X_l1_tile);
    TMOV(b_l0_tile, I_l1_tile);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

    TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains X
    SetWaitFlag<PIPE_M, PIPE_MTE1>(0);

    TMOV(b_l0_tile, Y_l1_tile);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

    TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile,
                b_l0_tile);  // c_l0 has X + X @ Y
    SetWaitFlag<PIPE_M, PIPE_FIX>(0);

    TMOV(X_l1_tile, c_l0_tile);
    SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);

    if (i < FractalSize / 4) {  // Update Y except in last iteration
      TMOV(a_l0_tile, Y_l1_tile);
      SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

      TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
      SetWaitFlag<PIPE_M, PIPE_FIX>(0);

      TMOV(Y_l1_tile, c_l0_tile);
      SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);
    }
  }
  /* Recursive part */
  for (uint32_t block_size = FractalSize; block_size < MatrixSize;
       block_size *= 2) {
    /* Load Identity In CO1 */
    TMOV(a_l0_tile, I_l1_tile);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
    SetWaitFlag<PIPE_M, PIPE_MTE1>(0);
    TMOV(b_l0_tile, I_l1_tile);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
    TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
    SetWaitFlag<PIPE_M, PIPE_MTE1>(0);

    /* Load Even Blocks Of X In L0A */
    TMOV(a_l0_tile, Zero_l1_tile);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
    SetWaitFlag<PIPE_M, PIPE_MTE1>(0);
    CopyOddOrEvenBlocksL1ToL0<InputT, TileL1AB,
                              TileLeft<InputT, FractalSize, FractalSize>,
                              MatrixSize>(X_l1_tile, block_size,
                                          0);  // a_l0_tile contains LX
    TMOV(b_l0_tile, M_neg_l1_tile);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

    TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile,
                b_l0_tile);  // c_l0_tile contains LX * M_neg + I
    SetWaitFlag<PIPE_M, PIPE_FIX>(0);

    TMOV(Y_l1_tile, c_l0_tile);  // Y_l1 contains LX * M_neg + I
    SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);

    TMOV(b_l0_tile, I_l1_tile);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
    TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0_tile contains LX
    SetWaitFlag<PIPE_M, PIPE_MTE1>(0);

    /* Load Odd Blocks Of X In L0B */
    TMOV(b_l0_tile, Zero_l1_tile);
    TMOV(a_l0_tile, Y_l1_tile);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
    SetWaitFlag<PIPE_M, PIPE_MTE1>(0);
    CopyOddOrEvenBlocksL1ToL0<InputT, TileL1AB,
                              TileRight<InputT, FractalSize, FractalSize>,
                              MatrixSize>(X_l1_tile, block_size, 1);
    SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

    TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile, b_l0_tile);
    SetWaitFlag<PIPE_M, PIPE_FIX>(0);

    if (block_size < MatrixSize / 2) {  // Update X_l1 except in last iteration
      TMOV(X_l1_tile, c_l0_tile);
      SetWaitFlag<PIPE_FIX, PIPE_MTE1>(0);
    }
  }

  /* Store result */
  TSTORE(M_inv_global_out, c_l0_tile);
}

/*
 * @brief: Computes the inverses of the blocks of tensor M
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize>
AICORE void runKernelTriInvRecUnroll(__gm__ OutputT* M_inv, __gm__ InputT* M,
                                     __gm__ InputT* I_neg) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))  // Cube compilation

  RunAICKernel<InputT, OutputT, MatrixSize>(M_inv, M, I_neg);
#else
// Nothing to do on AIV
#endif
}

template <typename InputT>
AICORE void run_tri_inv_rec_unroll(__gm__ float* tensor_out,
                                   __gm__ InputT* tensor_in,
                                   __gm__ InputT* identity_in,
                                   uint32_t matrix_size) {
  static_assert(std::is_same_v<InputT, half>,
                "tri_inv_rec_unroll supports only fp16.");
  switch (matrix_size) {
    case 16:
      runKernelTriInvRecUnroll<InputT, float, 16>(tensor_out, tensor_in,
                                                  identity_in);
      break;
    case 32:
      runKernelTriInvRecUnroll<InputT, float, 32>(tensor_out, tensor_in,
                                                  identity_in);
      break;
    case 64:
      runKernelTriInvRecUnroll<InputT, float, 64>(tensor_out, tensor_in,
                                                  identity_in);
      break;
    case 128:
      runKernelTriInvRecUnroll<InputT, float, 128>(tensor_out, tensor_in,
                                                   identity_in);
      break;
  }
}

extern "C" __global__ AICORE void tri_inv_rec_unroll_fp16(
    __gm__ void* tensor_out, __gm__ void* tensor_in, __gm__ void* identity_in,
    uint32_t matrix_size) {
  run_tri_inv_rec_unroll<half>((__gm__ float*)tensor_out,
                               (__gm__ half*)tensor_in,
                               (__gm__ half*)identity_in, matrix_size);
}
