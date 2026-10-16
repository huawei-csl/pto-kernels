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
 * @brief Transposes the S x S tile at `src_addr` into `dst_addr` in UB.
 *
 * When the scratch buffer only holds S / 2 rows (fp32 at S = 128), the
 * transpose is done in two halves: half the columns of the source transpose
 * into half the rows of the destination.
 */
template <typename T, unsigned S, uint32_t ScratchRows>
AICORE inline void TransposeTile(uint32_t dst_addr, uint32_t src_addr,
                                 uint32_t scratch_addr) {
  if constexpr (ScratchRows == S) {
    using MatTile = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, S, S>;
    MatTile src;
    TASSIGN(src, src_addr);
    MatTile dst;
    TASSIGN(dst, dst_addr);
    MatTile tmp;
    TASSIGN(tmp, scratch_addr);
    TTRANS(dst, src, tmp);
  } else {
    static_assert(2 * ScratchRows == S, "scratch must hold S or S / 2 rows");
    using HalfSrc = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, S, S / 2>;
    using HalfDst = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, S / 2, S>;
    using HalfTmp =
        Tile<TileType::Vec, T, S / 2, S, BLayout::RowMajor, S / 2, S>;
    for (uint32_t half = 0; half < 2; ++half) {
      HalfSrc src_half;
      TASSIGN(src_half, src_addr + half * (S / 2) * sizeof(T));
      HalfDst dst_half;
      TASSIGN(dst_half, dst_addr + half * (S / 2) * S * sizeof(T));
      HalfTmp half_tmp;
      TASSIGN(half_tmp, scratch_addr);
      TTRANS(dst_half, src_half, half_tmp);
      PipeBarrierVec();
    }
  }
  PipeBarrierVec();
}

/**
 * @brief Sets the S x S tile at `addr` in UB to the identity.
 *
 * Leaves an S -> V flag set; the caller waits on it before the next vector op
 * that reads the tile.
 */
template <typename T, unsigned S>
AICORE inline void InitIdentity(uint32_t addr) {
  using MatTile = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, S, S>;
  using RowTile = Tile<TileType::Vec, T, 1, S, BLayout::RowMajor, 1, S>;
  MatTile mat;
  TASSIGN(mat, addr);
  TEXPANDS(mat, static_cast<T>(0));
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  for (uint32_t i = 0; i < S; ++i) {
    RowTile diag;
    TASSIGN(diag, addr + i * S * sizeof(T));
    diag.SetValue(i, static_cast<T>(1));
  }
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

/**
 * @brief Inverts unit lower or upper triangular matrices with rank-1 updates.
 *
 *
 * The UB tile holds L (unit lower triangular, row-major) and the kernel has
 * to store X = L^-1. Writing Y = X^T, back substitution over all right-hand
 * sides at once is a sequence of rank-1 updates:
 *
 *     Y = I
 *     for m = S-1 .. 1:
 *         Y[0:m][:] -= col(L[m][0:m]) (x) row(Y[m][:])
 *
 * Y is built transposed on purpose: in that orientation both broadcast
 * sources are contiguous in UB (row m of L for the per-row factor, row m of
 * Y for the per-column factor), so no operand needs a strided gather. The
 * result is transposed back into the -- by then dead -- input buffer before
 * the store.
 *
 * Per step this is three vector instructions on the live block and, unlike
 * the per-column sweep, no scalar reads of UB and no S <-> V round trips at
 * all: ~3S vector instructions per matrix instead of ~S^2/2 AXPYs each
 * wrapped in two pipe serializations.
 *
 * Upper triangular input U is reduced to the lower case: U is loaded into the
 * Y buffer and transposed into the L buffer, L = U^T. The sweep then yields
 * Y = (L^-1)^T = U^-1, which is stored as-is, so the upper case also costs a
 * single transpose.
 *
 * BSND input (IsBSND) is the same inversion over tiles that are not
 * contiguous in global memory: the tensor is [total_tokens, N, S] and tile
 * (chunk, head) is the S x S block at row chunk * S of head `head`, so
 * consecutive rows of a tile sit N * S elements apart. Only the global
 * addressing changes; the UB-side sweep is untouched. With `cu_seqlens` the
 * sequences are variable length, so the last chunk of a sequence holds a
 * valid_size x valid_size matrix with valid_size < S. Such a tile is loaded
 * into the top-left corner of the S x S buffer and the rest is zero-filled,
 * which makes the buffer the block matrix diag(A, 0). The sweep never reads
 * the diagonal -- it assumes a unit one -- so it inverts diag(A, I) and the
 * top-left block of the result is A^-1. The valid rows are stored back at
 * full width, so the padding columns of the output come out as zeros.
 *
 * @tparam T Input data type (fp16 or fp32).
 * @tparam S Matrix size. Supports 16, 32, 64, 128.
 * @tparam IsLower Whether the input matrices are lower (true) or upper
 * (false) triangular.
 * @tparam IsBSND Whether the tiles are strided inside a BSND tensor (true) or
 * stored one after the other (false).
 *
 * @param vec_in Pointer to input buffer in global memory.
 * @param vec_out Pointer to output buffer in global memory.
 * @param total_tiles Number of matrices to invert.
 * @param num_bsnd_heads Number of heads N of the BSND tensor; unused unless
 * IsBSND.
 * @param cu_seqlens Cumulative sequence lengths for variable-length BSND
 * input, or nullptr when every chunk is full.
 */
template <typename T, unsigned S /* Matrix Size */, bool IsLower, bool IsBSND>
AICORE void runTTriInvRankOne(__gm__ T* vec_in, __gm__ T* vec_out,
                              uint32_t total_tiles, uint32_t num_bsnd_heads,
                              __gm__ int32_t* cu_seqlens) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  constexpr uint32_t tile_len = S * S;
  constexpr uint32_t tile_bytes = tile_len * sizeof(T);
  constexpr uint32_t row_bytes = S * sizeof(T);
  const uint32_t num_aiv_cores = get_block_num() * get_subblockdim();
  const uint32_t aiv_core_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t num_tiles_per_aiv =
      kernel_utils::CeilDiv(total_tiles, num_aiv_cores);

  using ShapeDim5 = pto::Shape<1, 1, 1, S, S>;
  using StrideDim5 = pto::Stride<1, 1, 1, S, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StrideDim5>;
  // BSND views: a full tile keeps the static S x S shape but takes the row
  // stride at runtime, a tail tile takes both the shape and the stride.
  using StrideDim5Dyn = pto::Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalDataStrided = pto::GlobalTensor<T, ShapeDim5, StrideDim5Dyn>;
  using ShapeDim5Dyn = pto::Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GlobalDataDyn = pto::GlobalTensor<T, ShapeDim5Dyn, StrideDim5Dyn>;

  // Row-major (ND) tiles for the blocks, column-major (DN) for the per-row
  // broadcast source that TROWEXPAND expects.
  //
  // The column count stays static on purpose: with Cols == ValidCol the
  // expand ops issue a single instruction with repeat = valid rows, while a
  // dynamic column count drops them onto a path that loops per row. Only the
  // row count varies per step.
  using MatTile = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, -1, S>;
  using RowTile = Tile<TileType::Vec, T, 1, S, BLayout::RowMajor, 1, S>;
  using ColTile = Tile<TileType::Vec, T, S, 1, BLayout::ColMajor, -1, 1>;
  // Tail tiles of a variable-length sequence: the valid block is smaller than
  // S in both directions, and TFILLPAD_INPLACE zeroes the rest of the buffer.
  using TailTile = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, DYNAMIC,
                        DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using PaddedTile = Tile<TileType::Vec, T, S, S, BLayout::RowMajor, S, S,
                          SLayout::NoneBox, 512, PadValue::Zero>;

  // A kernel may only use the first 184 KB of UB: pto parks an 8 KB scratch
  // at TMP_UB_OFFSET (184 KB) and TROWEXPANDMUL writes it whenever the
  // per-row source is a column-major tile, as it is here. Three full S x S
  // buffers only exceed that for fp32 at S = 128, which halves its scratch
  // and pays for it with a second chunk on the widest sweep steps.
  constexpr uint32_t UsableUbBytes = 184 * 1024;
  constexpr uint32_t OuterRows = (3 * tile_bytes <= UsableUbBytes) ? S : S / 2;
  constexpr uint32_t outer_bytes = OuterRows * S * sizeof(T);
  static_assert(2 * tile_bytes + outer_bytes <= UsableUbBytes,
                "tri_inv col sweep: UB budget exceeded");

  constexpr uint32_t L_UB_ADDR = 0x0;
  constexpr uint32_t Y_UB_ADDR = L_UB_ADDR + tile_bytes;
  constexpr uint32_t SCRATCH_UB_ADDR = Y_UB_ADDR + tile_bytes;
  // Lower input loads straight into the L buffer. Upper input loads into the
  // Y buffer, which is free until it gets transposed into the L buffer.
  constexpr uint32_t IN_UB_ADDR = IsLower ? L_UB_ADDR : Y_UB_ADDR;
  // Lower: the input is dead once the sweep ends, so the transposed result
  // lands on top of it rather than in a fourth buffer. Upper: Y is the result.
  constexpr uint32_t OUT_UB_ADDR = IsLower ? L_UB_ADDR : Y_UB_ADDR;

  MatTile in_tile(S);
  TASSIGN(in_tile, IN_UB_ADDR);
  MatTile out_tile(S);
  TASSIGN(out_tile, OUT_UB_ADDR);

  GlobalData global_in(vec_in);
  GlobalData global_out(vec_out);

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

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if constexpr (IsLower) {
      InitIdentity<T, S>(Y_UB_ADDR);
    }

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    if constexpr (IsBSND) {
      if (valid_size < S) {
        TailTile tail_in(valid_size, valid_size);
        TASSIGN(tail_in, IN_UB_ADDR);
        GlobalDataDyn global_in_dyn(vec_in + tile_offset,
                                    {1, 1, 1, static_cast<int>(valid_size),
                                     static_cast<int>(valid_size)},
                                    {1, 1, 1, row_stride, 1});
        TLOAD(tail_in, global_in_dyn);
      } else {
        GlobalDataStrided global_in_strided(vec_in + tile_offset, {},
                                            {row_stride});
        TLOAD(in_tile, global_in_strided);
      }
    } else {
      TASSIGN(global_in, vec_in + tile_offset);
      TLOAD(in_tile, global_in);
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (IsBSND) {
      // Zero everything outside the valid block, so that the sweep sees the
      // unit triangular diag(A, I) and leaves the padding alone.
      if (valid_size < S) {
        TailTile tail_in(valid_size, valid_size);
        TASSIGN(tail_in, IN_UB_ADDR);
        PaddedTile padded_in;
        TASSIGN(padded_in, IN_UB_ADDR);
        TFILLPAD_INPLACE(padded_in, tail_in);
        PipeBarrierVec();
      }
    }
    if constexpr (!IsLower) {
      // L = U^T, then Y can take over the input's buffer.
      TransposeTile<T, S, OuterRows>(L_UB_ADDR, Y_UB_ADDR, SCRATCH_UB_ADDR);
      InitIdentity<T, S>(Y_UB_ADDR);
    }
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

    // Rank-1 sweep. Row m of Y is final when step m starts, so the pivot row
    // can be read straight out of Y. No S <-> V round trips are needed here;
    // the barriers only order the back-to-back vector ops on `outer`.
    for (int32_t m = S - 1; m >= 1; --m) {
      const uint32_t rows = static_cast<uint32_t>(m);

      RowTile y_pivot;
      TASSIGN(y_pivot, Y_UB_ADDR + rows * row_bytes);

      for (uint32_t r0 = 0; r0 < rows; r0 += OuterRows) {
        const uint32_t chunk =
            (rows - r0 < OuterRows) ? (rows - r0) : OuterRows;
        ColTile l_col(chunk);
        TASSIGN(l_col, L_UB_ADDR + (rows * S + r0) * sizeof(T));
        MatTile outer(chunk);
        TASSIGN(outer, SCRATCH_UB_ADDR);
        MatTile y_blk(chunk);
        TASSIGN(y_blk, Y_UB_ADDR + r0 * row_bytes);

        // Broadcast the pivot row down the block first, then scale each row by
        // L[m][i]. Doing the row direction with the 3-argument TROWEXPANDMUL
        // matters enormously: the 2-argument TROWEXPAND lowers to one
        // vector_dup per row, which measured ~89% of the whole sweep, while
        // this form goes through the vbrcb path and issues one instruction.
        TCOLEXPAND(outer, y_pivot);  // outer[i][j]  = Y[m][j]
        PipeBarrierVec();
        TROWEXPANDMUL(outer, outer, l_col);  // outer[i][j] *= L[m][i]
        PipeBarrierVec();
        TSUB(y_blk, y_blk, outer);  // Y[i][j] -= L[m][i] * Y[m][j]
        PipeBarrierVec();
      }
    }

    if constexpr (IsLower) {
      // X = Y^T
      TransposeTile<T, S, OuterRows>(L_UB_ADDR, Y_UB_ADDR, SCRATCH_UB_ADDR);
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
        TASSIGN(tail_out, OUT_UB_ADDR);
        GlobalDataDyn global_out_dyn(
            vec_out + tile_offset,
            {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(S)},
            {1, 1, 1, row_stride, 1});
        TSTORE(global_out_dyn, tail_out);
      } else {
        GlobalDataStrided global_out_strided(vec_out + tile_offset, {},
                                             {row_stride});
        TSTORE(global_out_strided, out_tile);
      }
    } else {
      TASSIGN(global_out, vec_out + tile_offset);
      TSTORE(global_out, out_tile);
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

template <typename T, unsigned S>
AICORE void dispatchTriInv(__gm__ T* vec_in, __gm__ T* vec_out,
                           uint32_t total_tiles, uint32_t is_lower,
                           uint32_t num_bsnd_heads,
                           __gm__ int32_t* cu_seqlens) {
  if (num_bsnd_heads == 0) {
    if (is_lower) {
      runTTriInvRankOne<T, S, true, false>(vec_in, vec_out, total_tiles, 0,
                                           nullptr);
    } else {
      runTTriInvRankOne<T, S, false, false>(vec_in, vec_out, total_tiles, 0,
                                            nullptr);
    }
  } else {
    if (is_lower) {
      runTTriInvRankOne<T, S, true, true>(vec_in, vec_out, total_tiles,
                                          num_bsnd_heads, cu_seqlens);
    } else {
      runTTriInvRankOne<T, S, false, true>(vec_in, vec_out, total_tiles,
                                           num_bsnd_heads, cu_seqlens);
    }
  }
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp16(
    GM_ADDR x, GM_ADDR z, uint32_t total_tiles, uint32_t matrix_size,
    uint32_t is_lower, uint32_t num_bsnd_heads, GM_ADDR cu_seqlens) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)

  // Input/output dtype
  using fp16_t = __gm__ half*;
  __gm__ int32_t* _cu_seqlens = (__gm__ int32_t*)cu_seqlens;

  if (matrix_size == 16) {
    dispatchTriInv<half, 16>((fp16_t)x, (fp16_t)z, total_tiles, is_lower,
                             num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 32) {
    dispatchTriInv<half, 32>((fp16_t)x, (fp16_t)z, total_tiles, is_lower,
                             num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 64) {
    dispatchTriInv<half, 64>((fp16_t)x, (fp16_t)z, total_tiles, is_lower,
                             num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 128) {
    dispatchTriInv<half, 128>((fp16_t)x, (fp16_t)z, total_tiles, is_lower,
                              num_bsnd_heads, _cu_seqlens);
  }
#endif
}

extern "C" __global__ AICORE void triv_inv_col_sweep_fp32(
    GM_ADDR x, GM_ADDR z, uint32_t total_tiles, uint32_t matrix_size,
    uint32_t is_lower, uint32_t num_bsnd_heads, GM_ADDR cu_seqlens) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)

  // Input/output dtype
  using fp32_t = __gm__ float*;
  __gm__ int32_t* _cu_seqlens = (__gm__ int32_t*)cu_seqlens;

  if (matrix_size == 16) {
    dispatchTriInv<float, 16>((fp32_t)x, (fp32_t)z, total_tiles, is_lower,
                              num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 32) {
    dispatchTriInv<float, 32>((fp32_t)x, (fp32_t)z, total_tiles, is_lower,
                              num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 64) {
    dispatchTriInv<float, 64>((fp32_t)x, (fp32_t)z, total_tiles, is_lower,
                              num_bsnd_heads, _cu_seqlens);
  } else if (matrix_size == 128) {
    dispatchTriInv<float, 128>((fp32_t)x, (fp32_t)z, total_tiles, is_lower,
                               num_bsnd_heads, _cu_seqlens);
  }
#endif
}

extern "C" void pto_launch_triv_inv_col_sweep_fp16(
    uint32_t blockDim, void* stream, void* x, void* z, uint32_t total_tiles,
    uint32_t matrix_size, uint32_t is_lower, uint32_t num_bsnd_heads,
    void* cu_seqlens) {
  triv_inv_col_sweep_fp16<<<blockDim, nullptr, stream>>>(
      (GM_ADDR)x, (GM_ADDR)z, total_tiles, matrix_size, is_lower,
      num_bsnd_heads, (GM_ADDR)cu_seqlens);
}

extern "C" void pto_launch_triv_inv_col_sweep_fp32(
    uint32_t blockDim, void* stream, void* x, void* z, uint32_t total_tiles,
    uint32_t matrix_size, uint32_t is_lower, uint32_t num_bsnd_heads,
    void* cu_seqlens) {
  triv_inv_col_sweep_fp32<<<blockDim, nullptr, stream>>>(
      (GM_ADDR)x, (GM_ADDR)z, total_tiles, matrix_size, is_lower,
      num_bsnd_heads, (GM_ADDR)cu_seqlens);
}
