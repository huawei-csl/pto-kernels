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
  const uint32_t num_aiv_cores = get_block_num() * get_subblockdim();
  const uint32_t aiv_core_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t num_tiles_per_aiv =
      kernel_utils::CeilDiv(total_length, (tile_len * num_aiv_cores));
  const uint32_t b_size = S * sizeof(T);
  const uint32_t diff_size = S * sizeof(T);

  // UB zero address
  constexpr unsigned UB_ZERO_ADDR = 0x0;

  // define GlobalData on global memory with shape and stride
  using ShapeDim5 = pto::Shape<1, 1, 1, S, S>;
  using StrideDim5 = pto::Stride<1, 1, 1, S, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

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

  TileVecData x(1, S);
  // TODO (anastasios) only first k elements must be updated
  TileVecData diff(1, S);
  TASSIGN(diff, UB_ZERO_ADDR + matrix_in_size + b_size);
  TileVecData A_k(1, S);

  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  const uint32_t global_tile_id = aiv_core_id * num_tiles_per_aiv;
  for (uint32_t tile_id = 0;
       (tile_id < num_tiles_per_aiv) &&
       (global_tile_id + tile_id < total_length / tile_len);
       ++tile_id) {
    // Set output to all zeros.
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(inv_matrix_out, static_cast<T>(0));
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    GlobalData global_in(vec_in);
    GlobalData global_out(vec_out);
    TASSIGN(global_in, vec_in + (global_tile_id + tile_id) * tile_len);
    TASSIGN(global_out, vec_out + (global_tile_id + tile_id) * tile_len);
    TLOAD(matrix_in, global_in);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Find inverse by doing back-sub for each basis vector b_j
    // Solve A x = e_j for vector x
    for (int32_t j = S - 1; j >= 0; j--) {
      TEXPANDS(b, static_cast<T>(0));
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      b.SetValue(j, static_cast<T>(1));

      TASSIGN(x, out_start_ub_addr + j * S * sizeof(T));

      if (j == 0) {
        x.SetValue(0, static_cast<T>(1));
        continue;
      }

      TASSIGN(A_k, j * S * sizeof(T));
      TMULS(diff, A_k, static_cast<T>(1));
      set_flag(PIPE_V, PIPE_S, EVENT_ID1);

      pipe_barrier(PIPE_V);
      TSUB(b, b, diff);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);

      wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
      // Scalar core only supports fp32 arithmetic, so use float rather than T
      float xk = -static_cast<float>(diff.GetValue(j - 1));
      float xkp1 = 1.0f;

      for (int32_t k = j - 1; k >= 1; --k) {
        TASSIGN(A_k, k * S * sizeof(T));

        // Keep one x lane buffered so the scalar write from the previous
        // step overlaps the current vector update.
        TMULS(diff, A_k, static_cast<T>(xk));
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);

        x.SetValue(k + 1, static_cast<T>(xkp1));
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        const float bkm1 = static_cast<float>(b.GetValue(k - 1));
        pipe_barrier(PIPE_V);
        TSUB(b, b, diff);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);

        xkp1 = xk;

        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        xk = bkm1 - static_cast<float>(diff.GetValue(k - 1));
      }

      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      x.SetValue(1, static_cast<T>(xkp1));
      x.SetValue(0, static_cast<T>(xk));
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(global_out, inv_matrix_out);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp16(
    GM_ADDR x, GM_ADDR z, uint32_t in_length, uint32_t matrix_size) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

  if (matrix_size == 16) {
    runTTriInv<half, 16>((__gm__ half*)x, (__gm__ half*)z, in_length);
  } else if (matrix_size == 32) {
    runTTriInv<half, 32>((__gm__ half*)x, (__gm__ half*)z, in_length);
  } else if (matrix_size == 64) {
    runTTriInv<half, 64>((__gm__ half*)x, (__gm__ half*)z, in_length);
  } else if (matrix_size == 128) {
    runTTriInv<half, 128>((__gm__ half*)x, (__gm__ half*)z, in_length);
  }
#endif
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp32(
    GM_ADDR x, GM_ADDR z, uint32_t in_length, uint32_t matrix_size) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

  if (matrix_size == 16) {
    runTTriInv<float, 16>((__gm__ float*)x, (__gm__ float*)z, in_length);
  } else if (matrix_size == 32) {
    runTTriInv<float, 32>((__gm__ float*)x, (__gm__ float*)z, in_length);
  } else if (matrix_size == 64) {
    runTTriInv<float, 64>((__gm__ float*)x, (__gm__ float*)z, in_length);
  } else if (matrix_size == 128) {
    runTTriInv<float, 128>((__gm__ float*)x, (__gm__ float*)z, in_length);
  }
#endif
}
