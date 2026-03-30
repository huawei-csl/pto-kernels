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

template <typename T, unsigned TILE_LEN>
AICORE void runTHistogram(__gm__ T* x, __gm__ int32_t* z, __gm__ int32_t* z_local, const uint32_t total_length,
                          const int32_t num_bins, const float min_val, const float max_val) {

  set_mask_norm();
  set_vector_mask(-1, -1);

  // --- Define Global Tensors ---
  const uint32_t tile_num_elems = TILE_LEN * TILE_LEN;
  using InputShape = pto::Shape<1, 1, 1, TILE_LEN, TILE_LEN>;
  using InputStride = pto::Stride<1, 1, 1, TILE_LEN, 1>;
  using InputGlobalData = pto::GlobalTensor<T, InputShape, InputStride>;
  using HistGlobalData = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;

  // --- Define UB Tiles ---
  // Align num_bins for vector processing. Each int32_t is 4 bytes. 32-byte alignment.
  const uint32_t num_bins_aligned = kernel_utils::CeilDiv(num_bins, (uint32_t)(32 / sizeof(int32_t))) * (32 / sizeof(int32_t));

  // UB Memory Layout
  constexpr uint32_t UB_X_TILES_ADDR = 0;
  const uint32_t UB_CUR_MASK_ADDR = UB_X_TILES_ADDR + tile_num_elems * sizeof(T);
  const uint32_t UB_CUR_MASK_I32_ADDR = UB_CUR_MASK_ADDR + tile_num_elems * sizeof(uint8_t);
  const uint32_t UB_PREV_MASK_I32_ADDR = UB_CUR_MASK_I32_ADDR + tile_num_elems * sizeof(int32_t);
  const uint32_t UB_BIN_MASK_I32_ADDR = UB_PREV_MASK_I32_ADDR + tile_num_elems * sizeof(int32_t);
  const uint32_t UB_BIN_MASK_F32_ADDR = UB_BIN_MASK_I32_ADDR + tile_num_elems * sizeof(int32_t);
  const uint32_t UB_ROW_SUM_ADDR = UB_BIN_MASK_F32_ADDR + tile_num_elems * sizeof(float);
  const uint32_t UB_COUNT_ADDR = UB_ROW_SUM_ADDR + TILE_LEN * 8 * sizeof(float);
  const uint32_t UB_LOCAL_HIST_ADDR = UB_COUNT_ADDR + 8 * sizeof(float);

  // Input tile
  using InputTileData = Tile<TileType::Vec, T, TILE_LEN, TILE_LEN, BLayout::RowMajor, -1, -1>;
  InputTileData xTiles(TILE_LEN, TILE_LEN);
  TASSIGN(xTiles, UB_X_TILES_ADDR);

  // Mask tiles for binning
  using MaskTileData = Tile<TileType::Vec, uint8_t, TILE_LEN, TILE_LEN, BLayout::RowMajor, -1, -1>;
  MaskTileData current_mask(TILE_LEN, TILE_LEN);
  TASSIGN(current_mask, UB_CUR_MASK_ADDR);

  // Tiles for reduction (counting)
  using I32TileData = Tile<TileType::Vec, int32_t, TILE_LEN, TILE_LEN, BLayout::RowMajor, -1, -1>;
  I32TileData current_mask_i32(TILE_LEN, TILE_LEN);
  TASSIGN(current_mask_i32, UB_CUR_MASK_I32_ADDR);
  I32TileData prev_mask_i32(TILE_LEN, TILE_LEN);
  TASSIGN(prev_mask_i32, UB_PREV_MASK_I32_ADDR);
  I32TileData bin_mask_i32(TILE_LEN, TILE_LEN);
  TASSIGN(bin_mask_i32, UB_BIN_MASK_I32_ADDR);

  // Tiles for reduction (counting) - float versions for reduction ops
  using F32TileData = Tile<TileType::Vec, float, TILE_LEN, TILE_LEN, BLayout::RowMajor, -1, -1>;
  F32TileData bin_mask_f32(TILE_LEN, TILE_LEN);
  TASSIGN(bin_mask_f32, UB_BIN_MASK_F32_ADDR);

  using FloatRowSumTile = Tile<TileType::Vec, float, TILE_LEN, 8, BLayout::RowMajor, -1, -1>;
  FloatRowSumTile row_sum_f32_tile(TILE_LEN, 8);
  TASSIGN(row_sum_f32_tile, UB_ROW_SUM_ADDR);

  using FloatCountTile = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1>;
  FloatCountTile count_f32_tile(1, 8);
  TASSIGN(count_f32_tile, UB_COUNT_ADDR);

  // Local histogram tile
  constexpr uint32_t MAX_BINS = 8192;
  using HistTile = Tile<TileType::Vec, int32_t, 1, MAX_BINS, BLayout::RowMajor, 1, DYNAMIC>;
  HistTile localHist(num_bins_aligned);
  TASSIGN(localHist, UB_LOCAL_HIST_ADDR);
  TEXPANDS(localHist, (int32_t)0);

  // --- Phase 1: Local histogram calculation ---
  const uint32_t num_tiles_total = total_length / tile_num_elems;
  const uint32_t num_tiles_per_core = num_tiles_total / get_block_num();
  const uint32_t start_tile_idx = get_block_idx() * num_tiles_per_core;
  const uint32_t end_tile_idx = start_tile_idx + num_tiles_per_core;

  const float bin_width = (max_val - min_val) / num_bins;

  for (uint32_t tile_idx = start_tile_idx; tile_idx < end_tile_idx; ++tile_idx) {
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    // Load input tile from GM to UB
    const uint32_t offset = tile_idx * tile_num_elems;
    InputGlobalData xGlobal(x + offset);
    TLOAD(xTiles, xGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Initialize prev_mask to all zeros for the first bin boundary check
    TEXPANDS(prev_mask_i32, (int32_t)0);

    for (int32_t j = 0; j < num_bins; ++j) {
      float bin_upper_bound = min_val + (j + 1) * bin_width;
      CmpMode mode = CmpMode::LT;

      // The last bin is inclusive on the upper bound
      if (j == num_bins - 1) {
        bin_upper_bound = max_val;
        mode = CmpMode::LE;
      }

      // Create a mask for elements less than (or equal to) the bin's upper bound.
      // The result of TCMPS is a tile where elements are 0 or 1.
      TCMPS(current_mask, xTiles, static_cast<T>(bin_upper_bound), mode);

      // Convert the uint8_t mask to int32_t.
      TCVT(current_mask_i32, current_mask, RoundMode::CAST_NONE);

      // The elements in the current bin are those in the current mask but not the previous one.
      // Should have been done with TXOR but that fails. Since prev_mask is a subset of current_mask,
      // using TSUB is the same.
      TSUB(bin_mask_i32, current_mask_i32, prev_mask_i32);

      // TROWSUM/TCOLSUM do not support int32_t, so convert to float for reduction.
      TCVT(bin_mask_f32, bin_mask_i32, RoundMode::CAST_NONE);

      // Reduce the 2D tile to a single scalar value.
      // This requires a temporary tile for the intermediate row sums.
      TROWSUM(row_sum_f32_tile, bin_mask_f32, row_sum_f32_tile); // In-place temporary for some targets
      TCOLSUM(count_f32_tile, row_sum_f32_tile, row_sum_f32_tile, true);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

      // Add the count to the local histogram for the current bin.
      // This part is scalar as we update one bin at a time.
      float f_count = count_f32_tile.GetValue(0);
      int32_t count = static_cast<int32_t>(f_count + 0.5f); // Round to nearest int
      if (count > 0) {
        int32_t current_bin_count = localHist.GetValue(j);
        localHist.SetValue(j, current_bin_count + count);
      }

      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

      // The current mask becomes the previous mask for the next iteration.
      TMOV(prev_mask_i32, current_mask_i32);
    }
  }

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  // Store local histogram to GM
  const uint32_t local_hist_offset = get_block_idx() * num_bins;
  HistGlobalData zLocalGlobal(z_local + local_hist_offset, {num_bins});
  TSTORE(zLocalGlobal, localHist);

  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  // Barrier to ensure all local histograms are in GM before reduction phase
  pipe_barrier(PIPE_ALL);

  // --- Phase 2: Reduction of local histograms ---
  if (get_block_idx() == 0) {
    // Block 0's local histogram is already in its UB.
    // Now, add histograms from other blocks.
    HistTile otherHist(num_bins_aligned);
    TASSIGN(otherHist, UB_X_TILES_ADDR); // Reuse UB space from the beginning

    for (uint32_t i = 1; i < get_block_num(); ++i) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

      // Load other block's histogram
      const uint32_t other_hist_offset = i * num_bins;
      HistGlobalData otherHistGlobal(z_local + other_hist_offset, {num_bins});
      TLOAD(otherHist, otherHistGlobal);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // Add to the main histogram
      TADD(localHist, localHist, otherHist);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    // Store final histogram to z
    HistGlobalData zGlobal(z, {num_bins});
    TSTORE(zGlobal, localHist);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}

extern "C" __global__ AICORE void vhistogram_fp16(GM_ADDR x, GM_ADDR z, GM_ADDR z_local,
                                                  const uint32_t in_length,
                                                  const int32_t num_bins,
                                                  const float min_val, const float max_val) {
  constexpr unsigned TILE_LEN = 64;
  runTHistogram<half, TILE_LEN>((__gm__ half*)x, (__gm__ int32_t*)z, (__gm__ int32_t*)z_local, in_length,
                                   num_bins, min_val, max_val);
}

extern "C" __global__ AICORE void vhistogram_fp32(GM_ADDR x, GM_ADDR z, GM_ADDR z_local,
                                                  const uint32_t in_length,
                                                  const int32_t num_bins,
                                                  const float min_val, const float max_val) {
  constexpr unsigned TILE_LEN = 64;
  runTHistogram<float, TILE_LEN>((__gm__ float*)x, (__gm__ int32_t*)z, (__gm__ int32_t*)z_local, in_length,
                                   num_bins, min_val, max_val);
}
#endif
