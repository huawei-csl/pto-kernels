/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

#define MEMORY_BASE

#include <pto/pto-inst.hpp>
#include <limits>
#include "kernel_utils.h"

#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"

using namespace pto;

/**
 * runTLocalHistogram - Phase 1: Local histogram calculation per core.
 */
template <typename T, unsigned TILE_LEN>
AICORE void runTLocalHistogram(__gm__ T* x, __gm__ int32_t* z_local, const uint32_t total_length,
                              const int32_t num_bins, const float min_val, const float max_val)
{
  set_mask_norm();
  set_vector_mask(-1, -1);

  // --- Define Global Tensors ---
  constexpr uint32_t TILE_SIZE = TILE_LEN * TILE_LEN;
  using InputShape = pto::Shape<1, 1, 1, TILE_LEN, TILE_LEN>;
  using InputStride = pto::Stride<1, 1, 1, TILE_LEN, 1>;
  using InputGlobalData = pto::GlobalTensor<T, InputShape, InputStride>;
  
  // Align num_bins for vector processing and GM 32-byte alignment.
  const uint32_t num_bins_aligned = kernel_utils::CeilDiv(num_bins, 8) * 8;
  using HistGlobalData = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;

  // --- Define UB Tiles and Memory Layout ---
  constexpr uint32_t MASK_COLS = TILE_LEN / 8;
  constexpr uint32_t MASK_CAPACITY_COLS = 32;

  uint32_t addr = 0;
  const uint32_t UB_X_ADDR = addr; addr += TILE_SIZE * sizeof(T);
  const uint32_t UB_CUR_MASK_ADDR = addr; addr += TILE_LEN * MASK_CAPACITY_COLS * sizeof(uint8_t);
  const uint32_t UB_CUR_F32_ADDR = addr; addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_PREV_F32_ADDR = addr; addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_BIN_F32_ADDR = addr; addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_ONE_ADDR = addr; addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_ZERO_ADDR = addr; addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_REDUCE_TMP_ADDR = addr; addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_RSUM_ADDR = addr; addr += TILE_LEN * 8 * sizeof(float); // 64x8 for alignment
  const uint32_t UB_COUNT_ADDR = addr; addr += 8 * sizeof(float);
  const uint32_t UB_LOCAL_HIST_ADDR = addr; 

  // Input tile
  using InputTileData = Tile<TileType::Vec, T, TILE_LEN, TILE_LEN, BLayout::RowMajor, -1, -1>;
  InputTileData xTiles(TILE_LEN, TILE_LEN);
  TASSIGN(xTiles, UB_X_ADDR);

  // Mask tile (packed bits)
  using MaskTileData = Tile<TileType::Vec, uint8_t, TILE_LEN, MASK_CAPACITY_COLS, BLayout::RowMajor, -1, -1>;
  MaskTileData current_mask(TILE_LEN, MASK_COLS);
  TASSIGN(current_mask, UB_CUR_MASK_ADDR);

  // Float conversion tiles
  using F32TileData = Tile<TileType::Vec, float, TILE_LEN, TILE_LEN, BLayout::RowMajor, -1, -1>;
  F32TileData cur_f32(TILE_LEN, TILE_LEN);
  TASSIGN(cur_f32, UB_CUR_F32_ADDR);
  F32TileData prev_f32(TILE_LEN, TILE_LEN);
  TASSIGN(prev_f32, UB_PREV_F32_ADDR);
  F32TileData bin_mask_f32(TILE_LEN, TILE_LEN);
  TASSIGN(bin_mask_f32, UB_BIN_F32_ADDR);
  
  F32TileData one_tile(TILE_LEN, TILE_LEN);
  TASSIGN(one_tile, UB_ONE_ADDR);
  F32TileData zero_tile(TILE_LEN, TILE_LEN);
  TASSIGN(zero_tile, UB_ZERO_ADDR);
  TEXPANDS(one_tile, 1.0f);
  TEXPANDS(zero_tile, 0.0f);

  F32TileData reduce_tmp(TILE_LEN, TILE_LEN);
  TASSIGN(reduce_tmp, UB_REDUCE_TMP_ADDR);

  // Reduction result tiles.
  // For RowMajor and NoneBox, Cols must be a multiple of 8 (32 bytes for float).
  using FloatRowSumTile = Tile<TileType::Vec, float, TILE_LEN, 8, BLayout::RowMajor, TILE_LEN, DYNAMIC>;
  FloatRowSumTile row_sum_f32_tile(1); // Set ValidCol to 1
  TASSIGN(row_sum_f32_tile, UB_RSUM_ADDR);

  using FloatCountTile = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8>;
  FloatCountTile count_f32_tile;
  TASSIGN(count_f32_tile, UB_COUNT_ADDR);

  // Local histogram tile
  constexpr uint32_t MAX_BINS = 8192;
  using HistTile = Tile<TileType::Vec, int32_t, 1, MAX_BINS, BLayout::RowMajor, 1, DYNAMIC>;
  HistTile localHist(static_cast<size_t>(num_bins_aligned));
  TASSIGN(localHist, UB_LOCAL_HIST_ADDR);
  TEXPANDS(localHist, (int32_t)0);

  // --- Phase 1: Local histogram calculation ---
  const uint32_t num_tiles_total = kernel_utils::CeilDiv(total_length, TILE_SIZE);
  const uint32_t num_tiles_per_core = kernel_utils::CeilDiv(num_tiles_total, get_block_num());
  const uint32_t start_tile_idx = get_block_idx() * num_tiles_per_core;
  const uint32_t end_tile_idx = (start_tile_idx + num_tiles_per_core > num_tiles_total) ? num_tiles_total : (start_tile_idx + num_tiles_per_core);

  const float bin_width = (max_val - min_val) / num_bins;

  for (uint32_t tile_idx = start_tile_idx; tile_idx < end_tile_idx; ++tile_idx) {
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    const uint32_t offset = tile_idx * TILE_SIZE;
    InputGlobalData xGlobal(x + offset);
    TLOAD(xTiles, xGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TCMPS(current_mask, xTiles, static_cast<T>(min_val), CmpMode::LT);
    TSEL(prev_f32, current_mask, one_tile, zero_tile);

    for (int32_t j = 0; j < num_bins; ++j) {
      float bin_upper_bound = min_val + (j + 1) * bin_width;
      CmpMode mode = CmpMode::LT;
      if (j == num_bins - 1) {
        bin_upper_bound = max_val;
        mode = CmpMode::LE;
      }

      TCMPS(current_mask, xTiles, static_cast<T>(bin_upper_bound), mode);
      TSEL(cur_f32, current_mask, one_tile, zero_tile);

      TSUB(bin_mask_f32, cur_f32, prev_f32);

      TROWSUM(row_sum_f32_tile, bin_mask_f32, reduce_tmp);
      TCOLSUM(count_f32_tile, row_sum_f32_tile, row_sum_f32_tile, true);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

      float f_count = count_f32_tile.GetValue(0);
      int32_t count = static_cast<int32_t>(f_count + 0.5f);
      if (count > 0) {
        localHist.SetValue(j, localHist.GetValue(j) + count);
      }

      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

      TMOV(prev_f32, cur_f32);
    }
  }

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  const uint32_t local_hist_offset = get_block_idx() * num_bins_aligned;
  HistGlobalData zLocalGlobal(z_local + local_hist_offset, {static_cast<int>(num_bins_aligned)});
  TSTORE(zLocalGlobal, localHist);

  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

/**
 * runTReduceHistogram - Phase 2: Reduction of local histograms from all cores.
 */
AICORE void runTReduceHistogram(__gm__ int32_t* z, __gm__ int32_t* z_local,
                                const int32_t num_bins, const uint32_t num_cores) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_bins_aligned = kernel_utils::CeilDiv(num_bins, 8) * 8;
  using HistGlobalData = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;

  constexpr uint32_t UB_MAIN_HIST_ADDR = 0;
  const uint32_t UB_OTHER_HIST_ADDR = UB_MAIN_HIST_ADDR + num_bins_aligned * sizeof(int32_t);

  constexpr uint32_t MAX_BINS = 8192;
  using HistTile = Tile<TileType::Vec, int32_t, 1, MAX_BINS, BLayout::RowMajor, 1, DYNAMIC>;

  if (get_block_idx() == 0) {
    HistTile mainHist(static_cast<size_t>(num_bins_aligned));
    TASSIGN(mainHist, UB_MAIN_HIST_ADDR);

    HistGlobalData mainHistGlobal(z_local, {static_cast<int>(num_bins_aligned)});
    TLOAD(mainHist, mainHistGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    HistTile otherHist(static_cast<size_t>(num_bins_aligned));
    TASSIGN(otherHist, UB_OTHER_HIST_ADDR);

    for (uint32_t i = 1; i < num_cores; ++i) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

      const uint32_t other_hist_offset = i * num_bins_aligned;
      HistGlobalData otherHistGlobal(z_local + other_hist_offset, {static_cast<int>(num_bins_aligned)});
      TLOAD(otherHist, otherHistGlobal);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      TADD(mainHist, mainHist, otherHist);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    // Create a new tile object sharing the same address to set ValidCol to num_bins for final store
    HistTile finalHist(static_cast<size_t>(num_bins));
    TASSIGN(finalHist, UB_MAIN_HIST_ADDR);
    HistGlobalData zGlobal(z, {static_cast<int>(num_bins)});
    TSTORE(zGlobal, finalHist);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}

extern "C" __global__ AICORE void vhistogram_local_fp16(GM_ADDR x, GM_ADDR z_local,
                                                  const uint32_t in_length,
                                                  const int32_t num_bins,
                                                  const float min_val, const float max_val) {
  constexpr unsigned TILE_LEN = 64;
  runTLocalHistogram<half, TILE_LEN>((__gm__ half*)x, (__gm__ int32_t*)z_local, in_length,
                                   num_bins, min_val, max_val);
}

extern "C" __global__ AICORE void vhistogram_local_fp32(GM_ADDR x, GM_ADDR z_local,
                                                  const uint32_t in_length,
                                                  const int32_t num_bins,
                                                  const float min_val, const float max_val) {
  constexpr unsigned TILE_LEN = 64;
  runTLocalHistogram<float, TILE_LEN>((__gm__ float*)x, (__gm__ int32_t*)z_local, in_length,
                                   num_bins, min_val, max_val);
}

extern "C" __global__ AICORE void vhistogram_reduce_fp16(__gm__ int32_t* z, __gm__ int32_t* z_local,
                                                        const int32_t num_bins, const uint32_t num_cores) {
  runTReduceHistogram(z, z_local, num_bins, num_cores);
}

extern "C" __global__ AICORE void vhistogram_reduce_fp32(__gm__ int32_t* z, __gm__ int32_t* z_local,
                                                        const int32_t num_bins, const uint32_t num_cores) {
  runTReduceHistogram(z, z_local, num_bins, num_cores);
}
#endif