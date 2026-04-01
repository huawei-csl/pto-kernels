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

#include "kernel_utils.h"

#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"
using namespace pto;
using namespace kernel_utils;

#define BSND_OFFSET(tile_id, N, S, D) \
  (((tile_id) / (N)) * (S) * (N) * (D) + ((tile_id) % (N)) * (D))

/*
 * For aligned BSND, tile_id enumerates chunk-major then head-major and maps to
 * a fixed-stride address inside the dense BSND tensor.
 */
AICORE inline uint32_t GetBSNDFixedTileOffset(uint32_t tile_id,
                                              uint32_t num_bsnd_heads,
                                              uint32_t matrix_size) {
  return BSND_OFFSET(tile_id, num_bsnd_heads, matrix_size, matrix_size);
}

struct BSNDVarlenTileInfo {
  uint32_t bsnd_offset;
  uint32_t valid_size;
};

/*
 * For cu_seqlens-based varlen BSND, tile_id still enumerates chunk-major then
 * head-major. We recover the owning sequence by scanning cu_seqlens and
 * counting chunks per sequence.
 */
AICORE inline BSNDVarlenTileInfo GetBSNDVarlenTileInfoFromCuSeqlens(
    uint32_t tile_id, uint32_t num_bsnd_heads, uint32_t matrix_size,
    __gm__ int32_t* cu_seqlens) {
  const uint32_t head_idx = tile_id % num_bsnd_heads;
  const uint32_t chunk_idx = tile_id / num_bsnd_heads;

  uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[0]);
  uint32_t accumulated_chunks = 0;
  for (uint32_t seq_idx = 0;; ++seq_idx) {
    const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
    const uint32_t seq_len = seq_end - seq_start;
    const uint32_t seq_num_chunks = CeilDiv(seq_len, matrix_size);
    if (chunk_idx < accumulated_chunks + seq_num_chunks) {
      const uint32_t local_chunk_idx = chunk_idx - accumulated_chunks;
      const uint32_t row_start = seq_start + local_chunk_idx * matrix_size;
      const uint32_t valid_size =
          min(static_cast<uint32_t>(seq_end - row_start), matrix_size);
      return {row_start * num_bsnd_heads * matrix_size + head_idx * matrix_size,
              valid_size};
    }
    accumulated_chunks += seq_num_chunks;
    seq_start = seq_end;
  }
}

AICORE inline BSNDVarlenTileInfo GetBSNDVarlenTileInfoFromChunkMetadata(
    uint32_t tile_id, uint32_t num_bsnd_heads, uint32_t matrix_size,
    __gm__ int32_t* chunk_indices, __gm__ int32_t* chunk_valid_sizes) {
  const uint32_t head_idx = tile_id % num_bsnd_heads;
  const uint32_t chunk_idx = tile_id / num_bsnd_heads;
  const uint32_t row_start = static_cast<uint32_t>(chunk_indices[chunk_idx]);
  const uint32_t valid_size =
      static_cast<uint32_t>(chunk_valid_sizes[chunk_idx]);
  return {row_start * num_bsnd_heads * matrix_size + head_idx * matrix_size,
          valid_size};
}

AICORE inline BSNDVarlenTileInfo GetBSNDVarlenTileInfoFromChunkPrefix(
    uint32_t tile_id, uint32_t num_bsnd_heads, uint32_t matrix_size,
    __gm__ int32_t* cu_seqlens, __gm__ int32_t* chunk_sequence_prefix) {
  const uint32_t head_idx = tile_id % num_bsnd_heads;
  const uint32_t chunk_idx = tile_id / num_bsnd_heads;
  const uint32_t num_sequences =
      static_cast<uint32_t>(chunk_sequence_prefix[0]);

  uint32_t left = 0;
  uint32_t right = num_sequences;
  while (left < right) {
    const uint32_t mid = (left + right) / 2;
    const uint32_t chunk_end =
        static_cast<uint32_t>(chunk_sequence_prefix[mid + 2]);
    if (chunk_idx < chunk_end) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }

  const uint32_t seq_idx = left;
  const uint32_t chunk_base =
      static_cast<uint32_t>(chunk_sequence_prefix[seq_idx + 1]);
  const uint32_t local_chunk_idx = chunk_idx - chunk_base;
  const uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[seq_idx]);
  const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
  const uint32_t row_start = seq_start + local_chunk_idx * matrix_size;
  const uint32_t valid_size =
      min(static_cast<uint32_t>(seq_end - row_start), matrix_size);
  return {row_start * num_bsnd_heads * matrix_size + head_idx * matrix_size,
          valid_size};
}

/*
 * @brief: Takes as input two matrices of size MatrixSize * MatrixSize each.
 * The src matrix lies in L1, while the dst matrix lies either in L0A or L0B.
 * This kernel copies only the diagonal blocks (fractals) of size FractalSize *
 * FractalSize from the src matrix to the dst matrix.
 *
 * @tparam InputT Input data type (fp16).
 * @tparam FractalSize Size of each fractal matrix (diagonal block).
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam SrcL1TileT The actual tile type of the src matrix.
 * @tparam DstL0TileT The actual tile type of the dst matrix.
 *
 * @param src Tile in L1 memory.
 * @param dst Tile in L0A or L0B memory.
 */
template <typename InputT, uint32_t FractalSize, uint32_t MatrixSize,
          typename SrcL1TileT, typename DstL0TileT>
AICORE inline void CopyDiagonalFractalsL1ToL0(SrcL1TileT src, DstL0TileT dst) {
  constexpr uint32_t NumFractals = MatrixSize / FractalSize;
  constexpr bool is_left =
      std::is_same_v<DstL0TileT, TileLeft<InputT, MatrixSize, MatrixSize>>;
  constexpr TileType LeftOrRight = is_left ? TileType::Left : TileType::Right;
  constexpr SLayout InnerLayout =
      is_left ? SLayout::RowMajor : SLayout::ColMajor;

  Tile<LeftOrRight, InputT, FractalSize, FractalSize, BLayout::RowMajor,
       FractalSize, FractalSize, InnerLayout, TileConfig::fractalABSize>
      fractals[NumFractals];
  const std::uintptr_t starting_address =
      reinterpret_cast<std::uintptr_t>(dst.data());
  for (uint32_t i = 0; i < NumFractals; ++i) {
    TASSIGN(fractals[i], starting_address + i * FractalSize *
                                                (MatrixSize + FractalSize) *
                                                sizeof(InputT));
    TEXTRACT(fractals[i], src, i * FractalSize, i * FractalSize);
  }
}

/*
 * @brief: Takes as input two matrices of size MatrixSize * MatrixSize each,
 * and an integer block_size. The src matrix lies in L1, while the dst matrix
 * either in L0A or L0B. This method copies some of the diagonal blocks from the
 * input to the output as follows:
 * - If dst is in L0A (left): copy even diagonal blocks 0, 2, 4, ...
 * - If dst is in L0B (right): copy odd blocks 1, 3, 5, ...
 * Important note: the dst matrix should be initialized to all-zeros before
 * calling this method
 *
 * @tparam InputT Input data type (fp16).
 * @tparam FractalSize Size of each fractal matrix (diagonal block).
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam SrcL1TileT The actual tile type of the src matrix.
 * @tparam DstL0TileT The actual tile type of the dst matrix.
 *
 * @param src Tile in L1 memory.
 * @param dst Tile in L0A or L0B memory.
 * @param block_size Size of diagonal blocks. Needs: block_size >= FractalSize.
 */
template <typename InputT, uint32_t FractalSize, uint32_t MatrixSize,
          typename SrcL1TileT, typename DstL0TileT>
AICORE inline void CopyOddOrEvenBlocksL1ToL0(SrcL1TileT src, DstL0TileT dst,
                                             uint32_t block_size) {
  constexpr bool is_left =
      std::is_same_v<DstL0TileT, TileLeft<InputT, MatrixSize, MatrixSize>>;
  constexpr TileType LeftOrRight = is_left ? TileType::Left : TileType::Right;
  constexpr SLayout InnerLayout =
      is_left ? SLayout::RowMajor : SLayout::ColMajor;

  const uint32_t starting_block_index = is_left ? 0 : 1;

  const uint32_t num_blocks = MatrixSize / block_size;
  const uint32_t num_fractals_per_block = block_size / FractalSize;

  Tile<LeftOrRight, InputT, FractalSize, FractalSize, BLayout::RowMajor,
       FractalSize, FractalSize, InnerLayout, TileConfig::fractalABSize>
      fractals[MatrixSize / FractalSize];

  const std::uintptr_t starting_address =
      reinterpret_cast<std::uintptr_t>(dst.data());
  for (uint32_t i = 0; i < num_fractals_per_block; ++i) {
    for (uint32_t j = 0; j < num_fractals_per_block; ++j) {
      for (uint32_t b = starting_block_index; b < num_blocks; b += 2) {
        const uint32_t offset =
            b * (MatrixSize + FractalSize) * block_size +
            i * MatrixSize * FractalSize +
            j * FractalSize * FractalSize;
        TASSIGN(fractals[b], starting_address + offset * sizeof(InputT));
        TEXTRACT(fractals[b], src, b * block_size + i * FractalSize,
                 b * block_size + j * FractalSize);
      }
    }
  }
}

/*
 * @brief: Prepares Identity and Zeros matrix.
 *
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 *
 * @param I_neg_l1_tile Tile containing the -I (negative identity) matrix.
 * @param Zero_l1_tile Tile to store the all-zero matrix.
 * @param I_l1_tile Tile to store the identity matrix.
 * @param a_l0_tile Tile in L0A for matmuls.
 * @param b_l0_tile Tile in L0B for matmuls.
 * @param c_l0_tile Tile in L0C for matmuls.
 */
template <typename TileL1AB, typename TileL0A, typename TileL0B,
          typename TileL0C>
AICORE inline void PrepareAuxiliaryMatrices(
    TileL1AB I_neg_l1_tile, TileL1AB Zero_l1_tile, TileL1AB I_l1_tile,
    TileL0A a_l0_tile, TileL0B b_l0_tile, TileL0C c_l0_tile) {
  TMOV(a_l0_tile, I_neg_l1_tile);
  TMOV(b_l0_tile, I_neg_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
  wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));

  TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
  set_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  wait_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));

  TMOV(I_l1_tile, c_l0_tile);
  set_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));

  TMOV(b_l0_tile, I_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
  wait_flag(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));

  TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile, b_l0_tile);
  set_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  wait_flag(PIPE_M, PIPE_FIX, static_cast<event_t>(0));

  TMOV(Zero_l1_tile, c_l0_tile);
  set_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<event_t>(0));
}

/*
 * @brief: Inverts a single matrix / tile of the global tensor.
 */
template <typename InputT, typename TileL1AB, typename TileL0A,
          typename TileL0B, typename TileL0C, uint32_t MatrixSize,
          uint32_t FractalSize, uint32_t NumTilesPerCubeIter>
AICORE inline void InvertSingleTile(TileL1AB X_l1_tile, TileL1AB I_l1_tile,
                                    TileL1AB I_neg_l1_tile,
                                    TileL1AB M_neg_l1_tile,
                                    TileL1AB Zero_l1_tile, TileL1AB Y_l1_tile,
                                    TileL0A* a_l0_tile, TileL0B* b_l0_tile,
                                    TileL0C* c_l0_tile,
                                    const uint32_t tile_id) {
  const event_t event_0 = static_cast<event_t>(tile_id);
  const event_t event_1 = static_cast<event_t>(tile_id + NumTilesPerCubeIter);

  TMOV(b_l0_tile[0], Y_l1_tile);
  TMOV(a_l0_tile[0], I_neg_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, event_0);
  TMOV(a_l0_tile[1], Zero_l1_tile);
  TMOV(b_l0_tile[1], Zero_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, event_1);
  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  set_flag(PIPE_M, PIPE_MTE1, event_1);
  wait_flag(PIPE_M, PIPE_MTE1, event_1);
  CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile,
                                                              a_l0_tile[1]);
  CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile,
                                                              b_l0_tile[1]);
  set_flag(PIPE_MTE1, PIPE_M, event_1);

  wait_flag(PIPE_MTE1, PIPE_M, event_0);
  TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);
  set_flag(PIPE_M, PIPE_FIX, event_0);
  set_flag(PIPE_M, PIPE_MTE1, event_0);

  wait_flag(PIPE_M, PIPE_FIX, event_0);
  TMOV(M_neg_l1_tile, c_l0_tile[0]);
  set_flag(PIPE_FIX, PIPE_M, event_0);

  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  set_flag(PIPE_MTE1, PIPE_M, event_1);
  TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]);
  set_flag(PIPE_M, PIPE_FIX, event_1);
  wait_flag(PIPE_M, PIPE_FIX, event_1);
  TMOV(Y_l1_tile, c_l0_tile[1]);
  set_flag(PIPE_FIX, PIPE_M, event_1);
  wait_flag(PIPE_FIX, PIPE_M, event_1);

  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  TMOV(b_l0_tile[0], I_neg_l1_tile);
  TMOV(a_l0_tile[0], I_neg_l1_tile);
  set_flag(PIPE_MTE1, PIPE_M, event_0);

  wait_flag(PIPE_MTE1, PIPE_M, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);
  wait_flag(PIPE_MTE1, PIPE_M, event_1);
  TMATMUL(c_l0_tile[0], a_l0_tile[1], b_l0_tile[0]);
  set_flag(PIPE_M, PIPE_FIX, event_0);
  wait_flag(PIPE_M, PIPE_FIX, event_0);
  set_flag(PIPE_FIX, PIPE_M, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);

  TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);
  set_flag(PIPE_M, PIPE_FIX, event_1);
  wait_flag(PIPE_M, PIPE_FIX, event_1);
  TMOV(X_l1_tile, c_l0_tile[0]);

  set_flag(PIPE_FIX, PIPE_M, event_0);
  set_flag(PIPE_M, PIPE_MTE1, event_0);
  set_flag(PIPE_FIX, PIPE_MTE1, event_0);
  set_flag(PIPE_FIX, PIPE_M, event_1);
  set_flag(PIPE_M, PIPE_MTE1, event_1);
  set_flag(PIPE_FIX, PIPE_MTE1, event_1);
  for (uint32_t block_size = 1; block_size < FractalSize / 2; block_size *= 2) {
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    TMOV(b_l0_tile[0], I_l1_tile);
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
    TMOV(a_l0_tile[0], X_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
    TMOV(b_l0_tile[1], Y_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_1);

    wait_flag(PIPE_FIX, PIPE_M, event_0);
    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);
    set_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);

    if (block_size < FractalSize / 4) {
      wait_flag(PIPE_M, PIPE_MTE1, event_1);
      TMOV(a_l0_tile[1], Y_l1_tile);
      wait_flag(PIPE_MTE1, PIPE_M, event_1);
      set_flag(PIPE_MTE1, PIPE_M, event_1);

      wait_flag(PIPE_MTE1, PIPE_M, event_1);
      wait_flag(PIPE_FIX, PIPE_M, event_1);
      TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]);
      set_flag(PIPE_M, PIPE_MTE1, event_1);
      set_flag(PIPE_M, PIPE_FIX, event_1);
      set_flag(PIPE_MTE1, PIPE_M, event_1);

      wait_flag(PIPE_M, PIPE_FIX, event_1);
      TMOV(Y_l1_tile, c_l0_tile[1]);
      set_flag(PIPE_FIX, PIPE_M, event_1);
    }
    set_flag(PIPE_FIX, PIPE_MTE1, event_1);

    wait_flag(PIPE_MTE1, PIPE_M, event_1);
    TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0], b_l0_tile[1]);
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    set_flag(PIPE_M, PIPE_FIX, event_0);

    wait_flag(PIPE_M, PIPE_FIX, event_0);
    TMOV(X_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_FIX, PIPE_M, event_0);
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);
  }
  wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
  wait_flag(PIPE_M, PIPE_MTE1, event_1);
  wait_flag(PIPE_FIX, PIPE_M, event_1);
  wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  wait_flag(PIPE_FIX, PIPE_M, event_0);

  TMOV(b_l0_tile[1], M_neg_l1_tile);
  TMOV(a_l0_tile[0], I_l1_tile);

  if constexpr (MatrixSize > FractalSize) {
    set_flag(PIPE_FIX, PIPE_M, event_1);
  }
  set_flag(PIPE_M, PIPE_MTE1, event_1);
  set_flag(PIPE_M, PIPE_MTE1, event_0);
  set_flag(PIPE_FIX, PIPE_MTE1, event_1);
  set_flag(PIPE_FIX, PIPE_M, event_0);
  for (uint32_t block_size = FractalSize; block_size < MatrixSize;
       block_size *= 2) {
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    TMOV(a_l0_tile[1], Zero_l1_tile);

    wait_flag(PIPE_M, PIPE_MTE1, event_1);
    TMOV(b_l0_tile[0], I_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
    CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(X_l1_tile,
                                                               a_l0_tile[1],
                                                               block_size);
    set_flag(PIPE_MTE1, PIPE_M, event_1);

    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);
    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);

    wait_flag(PIPE_MTE1, PIPE_M, event_1);
    wait_flag(PIPE_FIX, PIPE_M, event_1);
    TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[0]);
    set_flag(PIPE_M, PIPE_MTE1, event_1);

    TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[1], b_l0_tile[1]);
    set_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_0);

    wait_flag(PIPE_M, PIPE_FIX, event_0);
    TMOV(Y_l1_tile, c_l0_tile[0]);
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_0);

    wait_flag(PIPE_M, PIPE_MTE1, event_1);
    TMOV(b_l0_tile[0], Zero_l1_tile);
    CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(X_l1_tile,
                                                               b_l0_tile[0],
                                                               block_size);

    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
    TMOV(a_l0_tile[1], Y_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL_ACC(c_l0_tile[1], c_l0_tile[1], a_l0_tile[1], b_l0_tile[0]);
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_1);
    set_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);

    if (block_size < MatrixSize / 2) {
      TMOV(X_l1_tile, c_l0_tile[1]);
      set_flag(PIPE_FIX, PIPE_M, event_1);
    }
    set_flag(PIPE_FIX, PIPE_MTE1, event_1);
  }
  wait_flag(PIPE_M, PIPE_MTE1, event_0);
  wait_flag(PIPE_M, PIPE_MTE1, event_1);
  wait_flag(PIPE_FIX, PIPE_M, event_0);
  wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
}

/*
 * @brief: Runs the main kernel (inverts all matrices in the tensor)
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE inline void TriInvRecUnrollKernel(__gm__ OutputT* M_inv,
                                         __gm__ InputT* M, __gm__ InputT* I_neg,
                                         uint32_t total_tiles,
                                         uint32_t num_bsnd_heads = 0,
                                         __gm__ int32_t* cu_seqlens = nullptr,
                                         __gm__ int32_t* chunk_sequence_prefix =
                                             nullptr,
                                         __gm__ int32_t* chunk_indices = nullptr,
                                         __gm__ int32_t* chunk_valid_sizes =
                                             nullptr) {
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  constexpr uint32_t FractalSize = 16;
  constexpr uint32_t NumL0Buffers = 2;

  if (get_block_idx() * NumTilesPerCubeIter >= total_tiles) {
    return;
  }

  using GlobalTileShapeIn =
      TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileStridesIn = typename std::conditional<
      !IsBSND, BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>,
      Stride<1, 1, 1, -1, 1>>::type;
  using GlobalTileIn =
      GlobalTensor<InputT, GlobalTileShapeIn, GlobalTileStridesIn, Layout::ND>;
  using GlobalTileDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GlobalTileDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalTileInDyn =
      GlobalTensor<InputT, GlobalTileDynShape, GlobalTileDynStride, Layout::ND>;

  using GlobalTileStridesINeg =
      BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileINeg = GlobalTensor<InputT, GlobalTileShapeIn,
                                      GlobalTileStridesINeg, Layout::ND>;

  using GlobalTileShapeOut =
      TileShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileStridesOut = typename std::conditional<
      !IsBSND, BaseShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>,
      Stride<1, 1, 1, -1, 1>>::type;
  using GlobalTileOut = GlobalTensor<OutputT, GlobalTileShapeOut,
                                     GlobalTileStridesOut, Layout::ND>;
  using GlobalTileOutDyn =
      GlobalTensor<OutputT, GlobalTileDynShape, GlobalTileDynStride, Layout::ND>;

  using TileL1AB =
      Tile<TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor,
           MatrixSize, MatrixSize, SLayout::RowMajor, 512>;
  using TileL1ABDyn = Tile<TileType::Mat, InputT, MatrixSize, MatrixSize,
                           BLayout::ColMajor, DYNAMIC, DYNAMIC,
                           SLayout::RowMajor, 512, PadValue::Zero>;
  using TileL0CDyn = TileAcc<OutputT, MatrixSize, MatrixSize, DYNAMIC, DYNAMIC>;

  using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
  using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
  using TileL0C = TileAcc<OutputT, MatrixSize, MatrixSize>;

  GlobalTileINeg I_neg_global_in(I_neg);

  TileL1AB X_l1_tile;
  TileL1AB I_l1_tile;
  TileL1AB I_neg_l1_tile;
  TileL1AB M_neg_l1_tile;
  TileL1AB Zero_l1_tile;
  TileL1AB Y_l1_tile[NumTilesPerCubeIter];

  TileL0A a_l0_tile[NumL0Buffers];
  TileL0B b_l0_tile[NumL0Buffers];
  TileL0C c_l0_tile[NumL0Buffers];

  TASSIGN(I_l1_tile, 0x0);
  TASSIGN(I_neg_l1_tile, 0x0 + TileLen * sizeof(InputT));
  TASSIGN(Zero_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));
  TASSIGN(M_neg_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
  TASSIGN(X_l1_tile, 0x0 + 4 * TileLen * sizeof(InputT));
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    TASSIGN(Y_l1_tile[tile_id], 0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
  }

  for (uint32_t buffer_num = 0; buffer_num < NumL0Buffers; ++buffer_num) {
    TASSIGN(a_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(b_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(c_l0_tile[buffer_num],
            0x0 + buffer_num * TileLen * sizeof(OutputT));
  }
  TLOAD(I_neg_l1_tile, I_neg_global_in);
  set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));

  PrepareAuxiliaryMatrices<TileL1AB, TileL0A, TileL0B, TileL0C>(
      I_neg_l1_tile, Zero_l1_tile, I_l1_tile, a_l0_tile[0], b_l0_tile[0],
      c_l0_tile[0]);

  const uint32_t max_iters_per_aic =
      CeilDiv(total_tiles, (uint32_t)(NumTilesPerCubeIter * get_block_num()));

  uint32_t bsnd_tile_offsets[NumTilesPerCubeIter] = {0};
  uint32_t bsnd_tile_valid_sizes[NumTilesPerCubeIter] = {0};

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
        const uint32_t global_tile_id = global_index + tile_id;
        if (chunk_indices != nullptr && chunk_valid_sizes != nullptr) {
          const BSNDVarlenTileInfo tile_info =
              GetBSNDVarlenTileInfoFromChunkMetadata(
                  global_tile_id, num_bsnd_heads, MatrixSize, chunk_indices,
                  chunk_valid_sizes);
          bsnd_tile_offsets[tile_id] = tile_info.bsnd_offset;
          bsnd_tile_valid_sizes[tile_id] = tile_info.valid_size;
        } else if (chunk_sequence_prefix != nullptr && cu_seqlens != nullptr) {
          const BSNDVarlenTileInfo tile_info =
              GetBSNDVarlenTileInfoFromChunkPrefix(
                  global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens,
                  chunk_sequence_prefix);
          bsnd_tile_offsets[tile_id] = tile_info.bsnd_offset;
          bsnd_tile_valid_sizes[tile_id] = tile_info.valid_size;
        } else if (cu_seqlens != nullptr) {
          const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
              global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
          bsnd_tile_offsets[tile_id] = tile_info.bsnd_offset;
          bsnd_tile_valid_sizes[tile_id] = tile_info.valid_size;
        } else {
          bsnd_tile_offsets[tile_id] =
              GetBSNDFixedTileOffset(global_tile_id, num_bsnd_heads, MatrixSize);
          bsnd_tile_valid_sizes[tile_id] = MatrixSize;
        }
        const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
        const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
        const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        if (valid_size < MatrixSize) {
          TileL1ABDyn Y_dyn_l1_tile(valid_size, valid_size);
          TASSIGN(Y_dyn_l1_tile,
                  0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
          GlobalTileInDyn M_global_in_dyn(
              M + bsnd_offset,
              {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(valid_size)},
              {1, 1, 1, row_stride, 1});
          TLOAD(Y_dyn_l1_tile, M_global_in_dyn);
          set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
          wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
          TFILLPAD(Y_dyn_l1_tile, Y_dyn_l1_tile);
        } else {
          GlobalTileIn M_global_in(M + bsnd_offset, {}, {row_stride});
          TLOAD(Y_l1_tile[tile_id], M_global_in);
        }
      } else {
        GlobalTileIn M_global_in(M + (global_index + tile_id) * TileLen);
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));
        TLOAD(Y_l1_tile[tile_id], M_global_in);
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
    }

    constexpr uint32_t final_c_buffer_index = MatrixSize > FractalSize ? 1 : 0;
    for (uint32_t tile_id = 0; (tile_id < NumTilesPerCubeIter) &&
                               (global_index + tile_id < total_tiles);
         ++tile_id) {
      wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(tile_id));
      wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));

      InvertSingleTile<InputT, TileL1AB, TileL0A, TileL0B, TileL0C, MatrixSize,
                       FractalSize, NumTilesPerCubeIter>(
          X_l1_tile, I_l1_tile, I_neg_l1_tile, M_neg_l1_tile, Zero_l1_tile,
          Y_l1_tile[tile_id], a_l0_tile, b_l0_tile, c_l0_tile, tile_id);

      set_flag(PIPE_M, PIPE_MTE2, static_cast<event_t>(tile_id));

      if constexpr (IsBSND) {
        const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
        const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
        const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
        if (valid_size < MatrixSize) {
          const event_t event_0 = static_cast<event_t>(tile_id);
          const event_t event_1 =
              static_cast<event_t>(tile_id + NumTilesPerCubeIter);
          TileL0CDyn c_l0_tail_tile(valid_size, valid_size);
          TASSIGN(c_l0_tail_tile,
                  0x0 + final_c_buffer_index * TileLen * sizeof(OutputT));
          if constexpr (final_c_buffer_index == 1) {
            set_flag(PIPE_M, PIPE_FIX, event_1);
            wait_flag(PIPE_M, PIPE_FIX, event_1);
          } else {
            set_flag(PIPE_M, PIPE_FIX, event_0);
            wait_flag(PIPE_M, PIPE_FIX, event_0);
          }
          set_flag(PIPE_FIX, PIPE_MTE3, static_cast<event_t>(tile_id));
          wait_flag(PIPE_FIX, PIPE_MTE3, static_cast<event_t>(tile_id));
          GlobalTileOutDyn M_inv_global_out_dyn(
              M_inv + bsnd_offset,
              {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(valid_size)},
              {1, 1, 1, row_stride, 1});
          TSTORE(M_inv_global_out_dyn, c_l0_tail_tile);
        } else {
          GlobalTileOut M_inv_global_out(M_inv + bsnd_offset, {},
                                         {row_stride});
          TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
        }
      } else {
        GlobalTileOut M_inv_global_out(M_inv +
                                       (global_index + tile_id) * TileLen);
        TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
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
}

/*
 * @brief: Varlen BSND kernel.
 *
 * The input/output tensors stay unpadded. For tail chunks with size
 * `actual_size < MatrixSize`, the kernel:
 * 1. derives the chunk row-start and runtime size from `cu_seqlens`
 * 2. loads only the valid `actual_size x actual_size` prefix via dynamic TLOAD
 * 3. zero-fills the remaining rows/cols in-place via TFILLPAD_INPLACE
 * 4. runs the original dense recursive inverse on the materialized full tile
 * 5. stores only the valid `actual_size x actual_size` prefix back to GM
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter>
AICORE inline void TriInvRecUnrollKernelBSNDVarlen(
    __gm__ OutputT* M_inv, __gm__ InputT* M, __gm__ InputT* I_neg,
    uint32_t total_tiles, uint32_t num_bsnd_heads, __gm__ int32_t* cu_seqlens) {
  constexpr uint32_t TileLen = MatrixSize * MatrixSize;
  constexpr uint32_t FractalSize = 16;
  constexpr uint32_t NumL0Buffers = 2;

  if (get_block_idx() * NumTilesPerCubeIter >= total_tiles) {
    return;
  }

  using GlobalTileShapeIn =
      TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileStridesIn = Stride<1, 1, 1, -1, 1>;
  using GlobalTileIn =
      GlobalTensor<InputT, GlobalTileShapeIn, GlobalTileStridesIn, Layout::ND>;

  using GlobalTileDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GlobalTileDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using GlobalTileInDyn =
      GlobalTensor<InputT, GlobalTileDynShape, GlobalTileDynStride, Layout::ND>;
  using GlobalTileOutDyn =
      GlobalTensor<OutputT, GlobalTileDynShape, GlobalTileDynStride, Layout::ND>;

  using GlobalTileStridesINeg =
      BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileINeg = GlobalTensor<InputT, GlobalTileShapeIn,
                                      GlobalTileStridesINeg, Layout::ND>;

  using GlobalTileShapeOut =
      TileShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>;
  using GlobalTileStridesOut = Stride<1, 1, 1, -1, 1>;
  using GlobalTileOut = GlobalTensor<OutputT, GlobalTileShapeOut,
                                     GlobalTileStridesOut, Layout::ND>;

  using TileL1AB =
      Tile<TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor,
           MatrixSize, MatrixSize, SLayout::RowMajor, 512>;
  using TileL1ABDyn = Tile<TileType::Mat, InputT, MatrixSize, MatrixSize,
                           BLayout::ColMajor, DYNAMIC, DYNAMIC,
                           SLayout::RowMajor, 512, PadValue::Zero>;

  using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
  using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
  using TileL0C = TileAcc<OutputT, MatrixSize, MatrixSize>;
  using TileL0CDyn = TileAcc<OutputT, MatrixSize, MatrixSize, DYNAMIC, DYNAMIC>;

  GlobalTileINeg I_neg_global_in(I_neg);

  TileL1AB X_l1_tile;
  TileL1AB I_l1_tile;
  TileL1AB I_neg_l1_tile;
  TileL1AB M_neg_l1_tile;
  TileL1AB Zero_l1_tile;
  TileL1AB Y_l1_tile[NumTilesPerCubeIter];

  TileL0A a_l0_tile[NumL0Buffers];
  TileL0B b_l0_tile[NumL0Buffers];
  TileL0C c_l0_tile[NumL0Buffers];

  TASSIGN(I_l1_tile, 0x0);
  TASSIGN(I_neg_l1_tile, 0x0 + TileLen * sizeof(InputT));
  TASSIGN(Zero_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));
  TASSIGN(M_neg_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
  TASSIGN(X_l1_tile, 0x0 + 4 * TileLen * sizeof(InputT));
  for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
    TASSIGN(Y_l1_tile[tile_id], 0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
  }

  for (uint32_t buffer_num = 0; buffer_num < NumL0Buffers; ++buffer_num) {
    TASSIGN(a_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(b_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
    TASSIGN(c_l0_tile[buffer_num],
            0x0 + buffer_num * TileLen * sizeof(OutputT));
  }
  TLOAD(I_neg_l1_tile, I_neg_global_in);
  set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));

  PrepareAuxiliaryMatrices<TileL1AB, TileL0A, TileL0B, TileL0C>(
      I_neg_l1_tile, Zero_l1_tile, I_l1_tile, a_l0_tile[0], b_l0_tile[0],
      c_l0_tile[0]);

  const uint32_t max_iters_per_aic =
      CeilDiv(total_tiles, (uint32_t)(NumTilesPerCubeIter * get_block_num()));
  constexpr uint32_t final_c_buffer_index = MatrixSize > FractalSize ? 1 : 0;

  for (uint32_t cube_iter = 0; cube_iter < max_iters_per_aic; ++cube_iter) {
    const uint32_t global_index =
        (cube_iter * get_block_num() + get_block_idx()) * NumTilesPerCubeIter;
    if (global_index >= total_tiles) {
      break;
    }

    for (uint32_t tile_id = 0; (tile_id < NumTilesPerCubeIter) &&
                               (global_index + tile_id < total_tiles);
         ++tile_id) {
      const uint32_t global_tile_id = global_index + tile_id;
      const BSNDVarlenTileInfo tile_info = GetBSNDVarlenTileInfoFromCuSeqlens(
          global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens);
      const uint32_t valid_size = tile_info.valid_size;
      const uint32_t bsnd_offset = tile_info.bsnd_offset;
      const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);

      if (valid_size == MatrixSize) {
        GlobalTileIn M_global_in(M + bsnd_offset, {}, {row_stride});
        TLOAD(Y_l1_tile[tile_id], M_global_in);
        set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
      } else {
        TileL1ABDyn Y_dyn_l1_tile(valid_size, valid_size);
        TASSIGN(Y_dyn_l1_tile,
                0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
        GlobalTileInDyn M_global_in_dyn(M + bsnd_offset,
                                        {1, 1, 1, valid_size, valid_size},
                                        {1, 1, 1, row_stride, 1});
        TLOAD(Y_dyn_l1_tile, M_global_in_dyn);
        set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
        wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(tile_id));
        TFILLPAD(Y_dyn_l1_tile, Y_dyn_l1_tile);
      }

      InvertSingleTile<InputT, TileL1AB, TileL0A, TileL0B, TileL0C, MatrixSize,
                       FractalSize, NumTilesPerCubeIter>(
          X_l1_tile, I_l1_tile, I_neg_l1_tile, M_neg_l1_tile, Zero_l1_tile,
          Y_l1_tile[tile_id], a_l0_tile, b_l0_tile, c_l0_tile, tile_id);

      if (valid_size == MatrixSize) {
        GlobalTileOut M_inv_global_out(M_inv + bsnd_offset, {}, {row_stride});
        TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
      } else {
        const event_t event_0 = static_cast<event_t>(tile_id);
        const event_t event_1 = static_cast<event_t>(tile_id + NumTilesPerCubeIter);
        TileL0CDyn c_l0_tail_tile(valid_size, valid_size);
        TASSIGN(c_l0_tail_tile,
                0x0 + final_c_buffer_index * TileLen * sizeof(OutputT));
        if constexpr (final_c_buffer_index == 1) {
          set_flag(PIPE_M, PIPE_FIX, event_1);
          wait_flag(PIPE_M, PIPE_FIX, event_1);
        } else {
          set_flag(PIPE_M, PIPE_FIX, event_0);
          wait_flag(PIPE_M, PIPE_FIX, event_0);
        }
        set_flag(PIPE_FIX, PIPE_MTE3, static_cast<event_t>(tile_id));
        wait_flag(PIPE_FIX, PIPE_MTE3, static_cast<event_t>(tile_id));
        GlobalTileOutDyn M_inv_global_out_dyn(
            M_inv + bsnd_offset, {1, 1, 1, valid_size, valid_size},
            {1, 1, 1, row_stride, 1});
        TSTORE(M_inv_global_out_dyn, c_l0_tail_tile);
      }
    }
  }
}

template <typename InputT, typename OutputT, uint32_t MatrixSize,
          uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE void runKernelTriInvRecUnroll(__gm__ OutputT* M_inv, __gm__ InputT* M,
                                     __gm__ InputT* I_neg, uint32_t total_tiles,
                                     uint32_t num_bsnd_heads = 0,
                                     __gm__ int32_t* cu_seqlens = nullptr,
                                     __gm__ int32_t* chunk_sequence_prefix =
                                         nullptr,
                                     __gm__ int32_t* chunk_indices = nullptr,
                                     __gm__ int32_t* chunk_valid_sizes =
                                         nullptr) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  TriInvRecUnrollKernel<InputT, OutputT, MatrixSize, NumTilesPerCubeIter,
                        IsBSND>(M_inv, M, I_neg, total_tiles, num_bsnd_heads,
                                cu_seqlens, chunk_sequence_prefix,
                                chunk_indices,
                                chunk_valid_sizes);
#else
// Nothing to do on AIV
#endif
}

template <typename InputT, uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE void run_tri_inv_rec_unroll(__gm__ float* tensor_out,
                                   __gm__ InputT* tensor_in,
                                   __gm__ InputT* minus_identity_in,
                                   uint32_t matrix_size, uint32_t num_matrices,
                                   uint32_t num_bsnd_heads,
                                   __gm__ int32_t* cu_seqlens,
                                   __gm__ int32_t* chunk_sequence_prefix,
                                   __gm__ int32_t* chunk_indices,
                                   __gm__ int32_t* chunk_valid_sizes) {
  static_assert(std::is_same_v<InputT, half>,
                "tri_inv_rec_unroll supports only fp16.");
  switch (matrix_size) {
    case 16:
      runKernelTriInvRecUnroll<InputT, float, 16, NumTilesPerCubeIter, IsBSND>(
          tensor_out, tensor_in, minus_identity_in, num_matrices, num_bsnd_heads,
          cu_seqlens, chunk_sequence_prefix, chunk_indices,
          chunk_valid_sizes);
      break;
    case 32:
      runKernelTriInvRecUnroll<InputT, float, 32, NumTilesPerCubeIter, IsBSND>(
          tensor_out, tensor_in, minus_identity_in, num_matrices, num_bsnd_heads,
          cu_seqlens, chunk_sequence_prefix, chunk_indices,
          chunk_valid_sizes);
      break;
    case 64:
      runKernelTriInvRecUnroll<InputT, float, 64, NumTilesPerCubeIter, IsBSND>(
          tensor_out, tensor_in, minus_identity_in, num_matrices, num_bsnd_heads,
          cu_seqlens, chunk_sequence_prefix, chunk_indices,
          chunk_valid_sizes);
      break;
    case 128:
      runKernelTriInvRecUnroll<InputT, float, 128, NumTilesPerCubeIter, IsBSND>(
          tensor_out, tensor_in, minus_identity_in, num_matrices, num_bsnd_heads,
          cu_seqlens, chunk_sequence_prefix, chunk_indices,
          chunk_valid_sizes);
      break;
  }
}

extern "C" __global__ AICORE void tri_inv_rec_unroll_fp16(
    __gm__ void* tensor_out, __gm__ void* tensor_in,
    __gm__ void* minus_identity_in, uint32_t matrix_size, uint32_t num_matrices,
    uint32_t num_bsnd_heads, __gm__ void* cu_seqlens,
    __gm__ void* chunk_sequence_prefix,
    __gm__ void* chunk_indices, __gm__ void* chunk_valid_sizes) {
  if (num_bsnd_heads == 0) {
    if (num_matrices <= get_block_num()) {
      run_tri_inv_rec_unroll<half, 1, false>(
          (__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
          (__gm__ half*)minus_identity_in, matrix_size, num_matrices,
          num_bsnd_heads, (__gm__ int32_t*)cu_seqlens,
          (__gm__ int32_t*)chunk_sequence_prefix,
          (__gm__ int32_t*)chunk_indices, (__gm__ int32_t*)chunk_valid_sizes);
    } else if (num_matrices <= 2 * get_block_num()) {
      run_tri_inv_rec_unroll<half, 2, false>(
          (__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
          (__gm__ half*)minus_identity_in, matrix_size, num_matrices,
          num_bsnd_heads, (__gm__ int32_t*)cu_seqlens,
          (__gm__ int32_t*)chunk_sequence_prefix,
          (__gm__ int32_t*)chunk_indices, (__gm__ int32_t*)chunk_valid_sizes);
    } else {
      run_tri_inv_rec_unroll<half, 4, false>(
          (__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
          (__gm__ half*)minus_identity_in, matrix_size, num_matrices,
          num_bsnd_heads, (__gm__ int32_t*)cu_seqlens,
          (__gm__ int32_t*)chunk_sequence_prefix,
          (__gm__ int32_t*)chunk_indices, (__gm__ int32_t*)chunk_valid_sizes);
    }
  } else {
    if (num_matrices <= get_block_num()) {
      run_tri_inv_rec_unroll<half, 1, true>(
          (__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
          (__gm__ half*)minus_identity_in, matrix_size, num_matrices,
          num_bsnd_heads, (__gm__ int32_t*)cu_seqlens,
          (__gm__ int32_t*)chunk_sequence_prefix,
          (__gm__ int32_t*)chunk_indices, (__gm__ int32_t*)chunk_valid_sizes);
    } else if (num_matrices <= 2 * get_block_num()) {
      run_tri_inv_rec_unroll<half, 2, true>(
          (__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
          (__gm__ half*)minus_identity_in, matrix_size, num_matrices,
          num_bsnd_heads, (__gm__ int32_t*)cu_seqlens,
          (__gm__ int32_t*)chunk_sequence_prefix,
          (__gm__ int32_t*)chunk_indices, (__gm__ int32_t*)chunk_valid_sizes);
    } else {
      run_tri_inv_rec_unroll<half, 4, true>(
          (__gm__ float*)tensor_out, (__gm__ half*)tensor_in,
          (__gm__ half*)minus_identity_in, matrix_size, num_matrices,
          num_bsnd_heads, (__gm__ int32_t*)cu_seqlens,
          (__gm__ int32_t*)chunk_sequence_prefix,
          (__gm__ int32_t*)chunk_indices, (__gm__ int32_t*)chunk_valid_sizes);
    }
  }
}
