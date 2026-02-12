/**
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it
 under the terms and conditions of CANN Open Software License Agreement
 Version 2.0 (the "License"). Please refer to the License for details. You may
 not use this file except in compliance with the License. THIS SOFTWARE IS
 PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
 repository for the full text of the License.
*/

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

#define MEMORY_BASE

#include <pto/pto-inst.hpp>

#include "kernel_operator.h"

using namespace pto;

/**
 * @brief Runs triangular matrix inverse on input buffer.
 *
 * @tparam T Input data type (fp16 or fp32).
 * @tparam S Matrix size. Supports 16, 32, 64, 128.

 * @param vec_in Pointer to input buffer in global memory.
 * @param vec_out Pointer to output buffer in global memory.
 * @param total_length Input tensor length, i.e., numel().
 */
template <typename T, unsigned S /* Matrix Size */>
AICORE void runTTriInv(__gm__ T* vec_in, __gm__ T* vec_out,
                       uint32_t total_length) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr uint32_t tile_len = S * S;
  const uint32_t matrix_in_size = tile_len * sizeof(T);
  const uint32_t b_size = S * sizeof(T);
  const uint32_t diff_size = S * sizeof(T);

  // UB zero address
  constexpr unsigned UB_ZERO_ADDR = 0x0;

  // define GlobalData on global memory with shape and stride
  using ShapeDim5 = pto::Shape<1, 1, 1, S, S>;
  using StrideDim5 = pto::Stride<1, 1, 1, S, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

  GlobalData global_in(vec_in);
  TASSIGN(global_in, vec_in + block_idx * tile_len);

  GlobalData global_out(vec_out);
  TASSIGN(global_out, vec_out + block_idx * tile_len);

  // define TileData on UB buffer with static shape and dynamic mask
  using TileData = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, -1, -1>;
  using TileVecData = Tile<TileType::Vec, T, 1, S, BLayout::RowMajor, -1, -1>;

  // Define all tiles
  TileData matrix_in(S, S);
  TASSIGN(matrix_in, UB_ZERO_ADDR);

  TileVecData b(1, S);
  TASSIGN(b, UB_ZERO_ADDR + matrix_in_size);

  TileData inv_matrix_out(S, S);
  const uint32_t out_start_ub_addr = matrix_in_size + b_size + diff_size;
  TASSIGN(inv_matrix_out, out_start_ub_addr);

  // Set output to all zeros.
  TEXPANDS(inv_matrix_out, static_cast<T>(0));

  // synchronization operations between hardware pipelines
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

  // load data from global memory to UB buffer
  TLOAD(matrix_in, global_in);

  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  TileVecData x(1, S);
  // TODO (anastasios) only first k elements must be updated
  TileVecData diff(1, S);
  TASSIGN(diff, UB_ZERO_ADDR + matrix_in_size + b_size);
  TileVecData A_k(1, S);

  // For every output column j-th
  for (int32_t j = 0; j < S; j++) {
    // Column sweep on each column.

    // `b` vector is  j-th standard vector (e_j).
    TEXPANDS(b, static_cast<T>(0));
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    b.SetValue(j, static_cast<T>(1));

    // Solve A x = e_j for vector x
    // Must be offset by UB address
    TASSIGN(x, out_start_ub_addr + j * S * sizeof(T));

    for (int32_t k = S - 1; k >= 0; k--) {
      TASSIGN(A_k, k * S * sizeof(T));

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // x[k] = b[k] / A[k, k]
      const T alpha = b.GetValue(k);
      x.SetValue(k, alpha);

      if (k > 0) {
        // b[:k] -= A[:k, k] * x[k]
        TEXPANDS(diff, static_cast<T>(0));
        TMULS(diff, A_k, alpha);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

        TSUB(b, b, diff);
      }
    }
  }

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(global_out, inv_matrix_out);
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp16(
    GM_ADDR x, GM_ADDR z, uint32_t in_length, uint32_t matrix_size) {
  if (matrix_size == 16) {
    runTTriInv<half, 16>((__gm__ half*)x, (__gm__ half*)z, in_length);
  } else if (matrix_size == 32) {
    runTTriInv<half, 32>((__gm__ half*)x, (__gm__ half*)z, in_length);
  } else if (matrix_size == 64) {
    runTTriInv<half, 64>((__gm__ half*)x, (__gm__ half*)z, in_length);
  } else if (matrix_size == 128) {
    runTTriInv<half, 128>((__gm__ half*)x, (__gm__ half*)z, in_length);
  }
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp32(
    GM_ADDR x, GM_ADDR z, uint32_t in_length, uint32_t matrix_size) {
  if (matrix_size == 16) {
    runTTriInv<float, 16>((__gm__ float*)x, (__gm__ float*)z, in_length);
  } else if (matrix_size == 32) {
    runTTriInv<float, 32>((__gm__ float*)x, (__gm__ float*)z, in_length);
  } else if (matrix_size == 64) {
    runTTriInv<float, 64>((__gm__ float*)x, (__gm__ float*)z, in_length);
  } else if (matrix_size == 128) {
    runTTriInv<float, 128>((__gm__ float*)x, (__gm__ float*)z, in_length);
  }
}

#endif
