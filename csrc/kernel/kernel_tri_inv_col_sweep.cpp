/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include "kernel_utils.h"

using namespace pto;
using namespace kernel_utils;

/**
 * @brief Runs triangular matrix inverse on input buffer.
 *
 * @tparam T Input data type (fp16 or fp32).
 * @tparam S Matrix size. Supports 16, 32, 64, 128.
 * @tparam IsBSND Whether the tiles are strided inside a BSND tensor (true) or
 * stored one after the other (false).

 * @param vec_in Pointer to input buffer in global memory.
 * @param vec_out Pointer to output buffer in global memory.
 * @param total_tiles Number of matrices to invert.
 * @param num_bsnd_heads Number of heads N of the BSND tensor; unused unless
 * IsBSND.
 * @param cu_seqlens Cumulative sequence lengths for variable-length BSND
 * input, or nullptr when every chunk is full.
 */
template <typename T, unsigned S /* Matrix Size */, bool IsBSND>
AICORE void runTTriInv(__gm__ T* vec_in, __gm__ T* vec_out,
                       uint32_t total_tiles, uint32_t num_bsnd_heads,
                       __gm__ int32_t* cu_seqlens) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr uint32_t tile_len = S * S;
  const uint32_t matrix_in_size = tile_len * sizeof(T);
  const uint32_t num_aiv_cores = get_block_num() * get_subblockdim();
  const uint32_t aiv_core_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t num_tiles_per_aiv =
      kernel_utils::CeilDiv(total_tiles, num_aiv_cores);
  const uint32_t b_size = S * sizeof(T);

  // UB zero address
  constexpr unsigned UB_ZERO_ADDR = 0x0;

  // define GlobalData on global memory with shape and stride
  using ShapeDim5 = pto::Shape<1, 1, 1, S, S>;
  using StrideDim5 = pto::Stride<1, 1, 1, S, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;

  // BSND views: a full tile keeps the static S x S shape but takes the row
  // stride at runtime, a tail tile takes both the shape and the stride.
  using StrideDim5Dyn = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalDataStrided = pto::GlobalTensor<T, ShapeDim5, StrideDim5Dyn>;
  using ShapeDim5Dyn = pto::Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GlobalDataDyn = pto::GlobalTensor<T, ShapeDim5Dyn, StrideDim5Dyn>;

  // define TileData on UB buffer with static shape and dynamic mask
  using TileData = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, -1, -1>;
  using TileVecData = Tile<TileType::Vec, T, 1, S, BLayout::RowMajor, -1, -1>;
  // Tail tiles of a variable-length sequence: the valid block is smaller than
  // S in both directions, and TFILLPAD_INPLACE zeroes the rest of the buffer.
  using TailTile = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, DYNAMIC,
                        DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using PaddedTile = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, S, S,
                          SLayout::NoneBox, 512, PadValue::Zero>;

  // Define all tiles
  TileData matrix_in(S, S);
  TASSIGN(matrix_in, UB_ZERO_ADDR);

  TileVecData b(1, S);
  TASSIGN(b, UB_ZERO_ADDR + matrix_in_size);

  TileData inv_matrix_out(S, S);
  const uint32_t out_start_ub_addr = matrix_in_size + b_size;
  TASSIGN(inv_matrix_out, out_start_ub_addr);

  TileVecData x(1, S);
  TileVecData A_k(1, S);

  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  const uint32_t first_tile_id = aiv_core_id * num_tiles_per_aiv;
  for (uint32_t tile_id = 0;
       (tile_id < num_tiles_per_aiv) && (first_tile_id + tile_id < total_tiles);
       ++tile_id) {
    const uint32_t global_tile_id = first_tile_id + tile_id;

    // Where this tile starts in global memory and how much of it is valid.
    uint32_t tile_offset = global_tile_id * tile_len;
    uint32_t valid_size = S;
    int row_stride = static_cast<int>(S);
    if constexpr (IsBSND) {
      row_stride = static_cast<int>(S * num_bsnd_heads);
      if (cu_seqlens != nullptr) {
        const BSNDVarlenTileInfo tile_info =
            kernel_utils::GetBSNDVarlenTileInfoFromCuSeqlens(
                global_tile_id, num_bsnd_heads, S, cu_seqlens);
        tile_offset = tile_info.bsnd_offset;
        valid_size = tile_info.valid_size;
      } else {
        tile_offset = kernel_utils::GetBSNDFixedTileOffset(global_tile_id,
                                                           num_bsnd_heads, S);
      }
    }

    // Set output to all zeros.
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(inv_matrix_out, static_cast<T>(0));
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    GlobalData global_in(vec_in);
    GlobalData global_out(vec_out);
    if constexpr (IsBSND) {
      if (valid_size < S) {
        TailTile tail_in(valid_size, valid_size);
        TASSIGN(tail_in, UB_ZERO_ADDR);
        GlobalDataDyn global_in_dyn(vec_in + tile_offset,
                                    {1, 1, 1, static_cast<int>(valid_size),
                                     static_cast<int>(valid_size)},
                                    {1, 1, 1, row_stride, 1});
        TLOAD(tail_in, global_in_dyn);
      } else {
        GlobalDataStrided global_in_strided(vec_in + tile_offset, {},
                                            {row_stride});
        TLOAD(matrix_in, global_in_strided);
      }
    } else {
      TASSIGN(global_in, vec_in + tile_offset);
      TLOAD(matrix_in, global_in);
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (IsBSND) {
      // Zero everything outside the valid block. The sweep takes the diagonal
      // to be an implicit unit diagonal, so a zeroed pad row is an identity
      // row and the padded matrix is diag(A, I) -- the back substitution then
      // returns e_j for every padded column and leaves the valid block alone.
      if (valid_size < S) {
        TailTile tail_in(valid_size, valid_size);
        TASSIGN(tail_in, UB_ZERO_ADDR);
        PaddedTile padded_in;
        TASSIGN(padded_in, UB_ZERO_ADDR);
        TFILLPAD_INPLACE(padded_in, tail_in);
        PipeBarrierVec();
      }
    }

    // Find inverse by doing back-sub for each basis vector b_j
    // Solve A x = e_j for vector x
    for (int32_t j = S - 1; j >= 0; j--) {
      // Set b to the j-th basis vector e_j
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
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      TAXPY(b, A_k, static_cast<T>(-1));
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float xk = static_cast<float>(b.GetValue(j - 1));
      float xkp1 = 1.0f;
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

      for (int32_t k = j - 1; k >= 1; --k) {
        TASSIGN(A_k, k * S * sizeof(T));

        PipeBarrierVec();
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        TAXPY(b, A_k, static_cast<T>(-xk));

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        x.SetValue(k + 1, static_cast<T>(xkp1));
        xkp1 = xk;
        xk = static_cast<float>(b.GetValue(k - 1));
      }

      x.SetValue(1, static_cast<T>(xkp1));
      x.SetValue(0, static_cast<T>(xk));
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    if constexpr (IsBSND) {
      if (valid_size < S) {
        // Store the full width of the valid rows: the result is
        // diag(A^-1, I), so the columns past valid_size are exact zeros and
        // the output needs no zero-fill. The rows past valid_size belong to
        // the next sequence and are left alone.
        TailTile tail_out(valid_size, S);
        TASSIGN(tail_out, out_start_ub_addr);
        GlobalDataDyn global_out_dyn(
            vec_out + tile_offset,
            {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(S)},
            {1, 1, 1, row_stride, 1});
        TSTORE(global_out_dyn, tail_out);
      } else {
        GlobalDataStrided global_out_strided(vec_out + tile_offset, {},
                                             {row_stride});
        TSTORE(global_out_strided, inv_matrix_out);
      }
    } else {
      TASSIGN(global_out, vec_out + tile_offset);
      TSTORE(global_out, inv_matrix_out);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

// Picks the BSND specialisation. Named apart from the rank-one kernel's
// dispatchTriInv, which this file links alongside.
template <typename T, unsigned S>
AICORE void dispatchColSweep(__gm__ T* vec_in, __gm__ T* vec_out,
                             uint32_t total_tiles, uint32_t num_bsnd_heads,
                             __gm__ int32_t* cu_seqlens) {
  if (num_bsnd_heads == 0) {
    runTTriInv<T, S, false>(vec_in, vec_out, total_tiles, 0, nullptr);
  } else {
    runTTriInv<T, S, true>(vec_in, vec_out, total_tiles, num_bsnd_heads,
                           cu_seqlens);
  }
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp16(
    GM_ADDR x, GM_ADDR z, uint32_t total_tiles, uint32_t matrix_size,
    uint32_t num_bsnd_heads, GM_ADDR cu_seqlens) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)

  using fp16_t = __gm__ half*;
  __gm__ int32_t* _cu_seqlens = (__gm__ int32_t*)cu_seqlens;

  if (matrix_size == 16) {
    dispatchColSweep<half, 16>((fp16_t)x, (fp16_t)z, total_tiles,
                                num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 32) {
    dispatchColSweep<half, 32>((fp16_t)x, (fp16_t)z, total_tiles,
                                num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 64) {
    dispatchColSweep<half, 64>((fp16_t)x, (fp16_t)z, total_tiles,
                                num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 128) {
    dispatchColSweep<half, 128>((fp16_t)x, (fp16_t)z, total_tiles,
                                 num_bsnd_heads, _cu_seqlens);
  }
#endif
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp32(
    GM_ADDR x, GM_ADDR z, uint32_t total_tiles, uint32_t matrix_size,
    uint32_t num_bsnd_heads, GM_ADDR cu_seqlens) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)

  using fp32_t = __gm__ float*;
  __gm__ int32_t* _cu_seqlens = (__gm__ int32_t*)cu_seqlens;

  if (matrix_size == 16) {
    dispatchColSweep<float, 16>((fp32_t)x, (fp32_t)z, total_tiles,
                                num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 32) {
    dispatchColSweep<float, 32>((fp32_t)x, (fp32_t)z, total_tiles,
                                num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 64) {
    dispatchColSweep<float, 64>((fp32_t)x, (fp32_t)z, total_tiles,
                                num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 128) {
    dispatchColSweep<float, 128>((fp32_t)x, (fp32_t)z, total_tiles,
                                 num_bsnd_heads, _cu_seqlens);
  }
#endif
}

// Host-callable launch shims: the `<<<>>>` syntax is only
// understood by the kernel compiler, so the launch lives here
// rather than in the host wrappers under csrc/host/.
extern "C" void pto_launch_triv_inv_col_sweep_fp16(
    uint32_t blockDim, void* stream, void* x, void* z, uint32_t total_tiles,
    uint32_t matrix_size, uint32_t num_bsnd_heads, void* cu_seqlens) {
  triv_inv_col_sweep_fp16<<<blockDim, nullptr, stream>>>(
      (GM_ADDR)x, (GM_ADDR)z, total_tiles, matrix_size, num_bsnd_heads,
      (GM_ADDR)cu_seqlens);
}

extern "C" void pto_launch_triv_inv_col_sweep_fp32(
    uint32_t blockDim, void* stream, void* x, void* z, uint32_t total_tiles,
    uint32_t matrix_size, uint32_t num_bsnd_heads, void* cu_seqlens) {
  triv_inv_col_sweep_fp32<<<blockDim, nullptr, stream>>>(
      (GM_ADDR)x, (GM_ADDR)z, total_tiles, matrix_size, num_bsnd_heads,
      (GM_ADDR)cu_seqlens);
}
