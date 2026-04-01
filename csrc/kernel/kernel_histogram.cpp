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
#include "kernel_utils.h"

#define GM_ADDR __gm__ uint8_t*

using namespace pto;

/**
 * runTHistogram - Local histogram calculation with Atomic Addition to Global Memory.
 */
template <typename T, unsigned TILE_LEN>
AICORE void runTHistogram(__gm__ T* x, __gm__ int32_t* z, const uint32_t total_length,
                                    const int32_t num_bins, const float min_val, const float max_val,
                                    const float bin_width)
{
  set_mask_norm();
  set_vector_mask(-1, -1);

  // --- Define Global Tensors ---
  constexpr uint32_t TILE_SIZE = TILE_LEN * TILE_LEN;
  using InputShape = pto::Shape<1, 1, 1, TILE_LEN, TILE_LEN>;
  using InputStride = pto::Stride<1, 1, 1, TILE_LEN, 1>;
  using InputGlobalData = pto::GlobalTensor<T, InputShape, InputStride>;
  
  // Align num_bins for vector processing (multiple of 8 for 32-byte alignment)
  const uint32_t num_bins_aligned = kernel_utils::CeilDiv(num_bins, 8) * 8;
  using HistGlobalData = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;

  // --- Work Distribution ---
  const uint32_t num_tiles_total = kernel_utils::CeilDiv(total_length, TILE_SIZE);
  const uint32_t num_tiles_per_core = kernel_utils::CeilDiv(num_tiles_total, get_block_num());
  const uint32_t start_tile_idx = get_block_idx() * num_tiles_per_core;
  const uint32_t end_tile_idx = (start_tile_idx + num_tiles_per_core > num_tiles_total) ? num_tiles_total : (start_tile_idx + num_tiles_per_core);

  // Only cores that have actual tiles to process perform the Atomic operation.
  if (start_tile_idx < end_tile_idx) {
    // --- Define UB Tiles and Memory Layout ---
    constexpr uint32_t MASK_COLS = TILE_LEN / 8;
    constexpr uint32_t MASK_CAPACITY_COLS = MASK_COLS > 32 ? MASK_COLS : 32;

    uint32_t addr = 0;
    const uint32_t UB_X_ADDR = addr;          addr += TILE_SIZE * sizeof(T);
    const uint32_t UB_CUR_MASK_ADDR = addr;   addr += TILE_LEN * MASK_CAPACITY_COLS * sizeof(uint8_t);
    const uint32_t UB_CUR_F32_ADDR = addr;    addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_PREV_F32_ADDR = addr;   addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_BIN_F32_ADDR = addr;    addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_ONE_ADDR = addr;        addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_ZERO_ADDR = addr;       addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_REDUCE_TMP_ADDR = addr; addr += TILE_SIZE * sizeof(float);
    const uint32_t UB_RSUM_ADDR = addr;       addr += TILE_LEN * 8 * sizeof(float);
    const uint32_t UB_COUNT_ADDR = addr;      addr += 8 * sizeof(float);
    const uint32_t UB_LOCAL_HIST_ADDR = addr;

    using InputTileData = Tile<TileType::Vec, T, TILE_LEN, TILE_LEN>;
    InputTileData xTiles;
    TASSIGN(xTiles, UB_X_ADDR);

    // Mask tile
    //using MaskTileData = Tile<TileType::Vec, uint8_t, TILE_LEN, MASK_CAPACITY_COLS, BLayout::RowMajor, -1, -1>;
    //MaskTileData current_mask(TILE_LEN, MASK_COLS);
    Tile<TileType::Vec, uint8_t, TILE_LEN, MASK_CAPACITY_COLS> current_mask;
    TASSIGN(current_mask, UB_CUR_MASK_ADDR);

    // Float conversion tiles
    using F32TileData = Tile<TileType::Vec, float, TILE_LEN, TILE_LEN>;
    F32TileData cur_f32;
    TASSIGN(cur_f32, UB_CUR_F32_ADDR);
    F32TileData prev_f32;
    TASSIGN(prev_f32, UB_PREV_F32_ADDR);
    F32TileData bin_mask_f32;
    TASSIGN(bin_mask_f32, UB_BIN_F32_ADDR);
    
    F32TileData one_tile;
    TASSIGN(one_tile, UB_ONE_ADDR);
    TEXPANDS(one_tile, 1.0f);
    F32TileData zero_tile;
    TASSIGN(zero_tile, UB_ZERO_ADDR);
    TEXPANDS(zero_tile, 0.0f);

    F32TileData reduce_tmp;
    TASSIGN(reduce_tmp, UB_REDUCE_TMP_ADDR);

    using FloatRowSumTile = Tile<TileType::Vec, float, TILE_LEN, 8, BLayout::RowMajor, TILE_LEN, DYNAMIC>;
    FloatRowSumTile row_sum_f32_tile(1);
    TASSIGN(row_sum_f32_tile, UB_RSUM_ADDR);

    using FloatCountTile = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, DYNAMIC>;
    FloatCountTile count_f32_tile(1);
    TASSIGN(count_f32_tile, UB_COUNT_ADDR);

    // Local histogram tile in UB
    constexpr uint32_t MAX_BINS_LIMIT = 1024;
    using HistTile = Tile<TileType::Vec, int32_t, 1, MAX_BINS_LIMIT, BLayout::RowMajor, 1, DYNAMIC>;
    HistTile localHist(static_cast<size_t>(num_bins_aligned));
    TASSIGN(localHist, UB_LOCAL_HIST_ADDR);
    TEXPANDS(localHist, (int32_t)0);

    // Initial barrier to ensure UB layout and constants are set.
    pipe_barrier(PIPE_ALL);

    // --- Main Calculation Loop ---
    for (uint32_t tile_idx = start_tile_idx; tile_idx < end_tile_idx; ++tile_idx) {
      const uint32_t offset = tile_idx * TILE_SIZE;
      InputGlobalData xGlobal(x + offset);
      
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(xTiles, xGlobal);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      TCMPS(current_mask, xTiles, static_cast<T>(min_val), CmpMode::LT);
      TSEL(prev_f32, current_mask, one_tile, zero_tile);
      //TEXPANDS(prev_f32, 0.0f);

      for (int32_t j = 0; j < num_bins; ++j) {
        float bin_upper_bound = min_val + (j + 1) * bin_width;
        CmpMode mode = (j == num_bins - 1) ? CmpMode::LE : CmpMode::LT;

        TCMPS(current_mask, xTiles, static_cast<T>(bin_upper_bound), mode);
        TSEL(cur_f32, current_mask, one_tile, zero_tile);
        TSUB(bin_mask_f32, cur_f32, prev_f32);

        TEXPANDS(row_sum_f32_tile, 0.0f);
        TEXPANDS(reduce_tmp, 0.0f);
        TROWSUM(row_sum_f32_tile, bin_mask_f32, reduce_tmp);
        
        TEXPANDS(count_f32_tile, 0.0f);
        TEXPANDS(reduce_tmp, 0.0f);
        TCOLSUM(count_f32_tile, row_sum_f32_tile, reduce_tmp, true);

        // Scalar move to update UB local histogram
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
        pipe_barrier(PIPE_ALL);
      }
    }
    pipe_barrier(PIPE_ALL);

    // --- Final Atomic Store to Global Memory ---
    HistTile finalHist(static_cast<size_t>(num_bins));
    TASSIGN(finalHist, UB_LOCAL_HIST_ADDR);
    HistGlobalData zGlobal(z, {1, 1, 1, 1, num_bins});

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    // Perform Atomic Addition directly into the final result tensor
    TSTORE<HistTile, HistGlobalData, AtomicType::AtomicAdd>(zGlobal, finalHist);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}

extern "C" __global__ AICORE void vhistogram_fp16(GM_ADDR x, GM_ADDR z,
                                                   const uint32_t in_length,
                                                   const int32_t num_bins,
                                                   const float min_val, const float max_val,
                                                   const float bin_width) {
  constexpr unsigned TILE_LEN = 64;
  runTHistogram<half, TILE_LEN>((__gm__ half*)x, (__gm__ int32_t*)z, in_length,
                                         num_bins, min_val, max_val, bin_width);
}

extern "C" __global__ AICORE void vhistogram_fp32(GM_ADDR x, GM_ADDR z,
                                                   const uint32_t in_length,
                                                   const int32_t num_bins,
                                                   const float min_val, const float max_val,
                                                   const float bin_width) {
  constexpr unsigned TILE_LEN = 64;
  runTHistogram<float, TILE_LEN>((__gm__ float*)x, (__gm__ int32_t*)z, in_length,
                                         num_bins, min_val, max_val, bin_width);
}

#endif
