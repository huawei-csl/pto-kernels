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

constexpr uint32_t DEFAULT_TILE_SIZE = 512;
constexpr uint32_t MAX_BINS = 256;
constexpr uint32_t MAX_BLOCKS = 64;

/**
 * runTLocalHistogram - Local, per-core histogram calculation
 */
template <typename T, unsigned TILE_SIZE>
AICORE void runTLocalHistogram(__gm__ T* x, __gm__ float* z_local,
                               const uint32_t total_length,
                               const int32_t num_bins, const float min_val,
                               const float max_val) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  // --- Define Global Tensors ---
  using InputGlobalData = pto::GlobalTensor<T, pto::Shape<1, 1, 1, 1, DYNAMIC>,
                                            pto::Stride<1, 1, 1, 1, 1>>;
  using HistGlobalData =
      pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, DYNAMIC>,
                        pto::Stride<1, 1, 1, 1, 1>>;

  // --- Work Distribution ---
  const uint32_t block_idx = get_block_idx();
  const uint32_t block_num = get_block_num();
  const uint32_t num_tiles_total =
      kernel_utils::CeilDiv(total_length, TILE_SIZE);
  const uint32_t num_tiles_per_core =
      kernel_utils::CeilDiv(num_tiles_total, block_num);
  const uint32_t start_idx = block_idx * num_tiles_per_core;
  const uint32_t end_idx = (start_idx + num_tiles_per_core > num_tiles_total)
                               ? num_tiles_total
                               : (start_idx + num_tiles_per_core);

  // --- Define UB Tiles and Memory Layout ---
  uint32_t addr = 0;
  const uint32_t UB_X_ADDR = addr;
  addr += TILE_SIZE * sizeof(T);
  const uint32_t UB_CUR_MASK_ADDR = addr;
  addr += TILE_SIZE * sizeof(uint8_t);
  const uint32_t UB_CUR_F32_ADDR = addr;
  addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_PREV_F32_ADDR = addr;
  addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_BIN_F32_ADDR = addr;
  addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_ONE_ADDR = addr;
  addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_ZERO_ADDR = addr;
  addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_REDUCE_TMP_ADDR = addr;
  addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_COUNT_ADDR = addr;
  addr += 8 * sizeof(float);
  const uint32_t UB_LOCAL_HIST_ADDR = addr;

  InputGlobalData x_gm(x, {static_cast<int32_t>(total_length)});

  using InputTileData = Tile<TileType::Vec, T, 1, TILE_SIZE>;
  InputTileData x_tile;
  TASSIGN(x_tile, UB_X_ADDR);

  using MaskTileData = Tile<TileType::Vec, uint8_t, 1, TILE_SIZE>;
  MaskTileData current_mask;
  TASSIGN(current_mask, UB_CUR_MASK_ADDR);

  // Float conversion tiles
  using F32TileData = Tile<TileType::Vec, float, 1, TILE_SIZE>;
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

  using F32CountTile =
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;
  F32CountTile count_f32_tile;
  TASSIGN(count_f32_tile, UB_COUNT_ADDR);

  // Local histogram tile in UB
  using HistTile =
      Tile<TileType::Vec, float, 1, MAX_BINS, BLayout::RowMajor, 1, DYNAMIC>;
  HistTile local_hist(num_bins);
  TASSIGN(local_hist, UB_LOCAL_HIST_ADDR);
  TEXPANDS(local_hist, 0.0f);

  const float bin_width = (max_val - min_val) / static_cast<float>(num_bins);

  // --- Main Calculation Loop ---
  for (uint32_t tile_idx = start_idx; tile_idx < end_idx; ++tile_idx) {
    int offset = tile_idx * TILE_SIZE;
    TASSIGN(x_gm, x + offset);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(x_tile, x_gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Generate packed bit-mask
    TCMPS(current_mask, x_tile, static_cast<T>(min_val), CmpMode::LT);
    // Select 1.0f or 0.0f based on the packed bit-mask
    TSEL(prev_f32, current_mask, one_tile, zero_tile);

    for (int32_t j = 0; j < num_bins; ++j) {
      float bin_upper_bound = min_val + (j + 1) * bin_width;
      CmpMode mode = (j == num_bins - 1) ? CmpMode::LE : CmpMode::LT;

      TCMPS(current_mask, x_tile, static_cast<T>(bin_upper_bound), mode);
      TSEL(cur_f32, current_mask, one_tile, zero_tile);
      TSUB(bin_mask_f32, cur_f32, prev_f32);

      // Reduce the selected tile to get the count of elements less than pivot
      // in this tile
      TEXPANDS(count_f32_tile, 0.0f);
      TEXPANDS(reduce_tmp, 0.0f);
      TROWSUM(count_f32_tile, bin_mask_f32, reduce_tmp);

      // Scalar move to update UB local histogram
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float f_count = count_f32_tile.GetValue(0);
      if (f_count > 0.0f) {
        local_hist.SetValue(j, local_hist.GetValue(j) + f_count);
      }
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

      TMOV(prev_f32, cur_f32);
    }
  }

  // --- Final Store to Global Memory ---
  HistGlobalData z_gm(z_local + block_idx * num_bins,
                      {static_cast<int32_t>(block_num) * num_bins});

  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(z_gm, local_hist);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

// Template parameter to avoid "no function" kernel launch error
template <unsigned UNUSED>
AICORE void runTHistogramFinal(__gm__ float* z_local, __gm__ int32_t* z,
                               const int32_t num_bins,
                               const int32_t num_blocks) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (get_block_idx() == 0) {
    // --- Define Global Tensors ---
    using InGlobalData =
        pto::GlobalTensor<float, pto::Shape<1, 1, 1, 1, DYNAMIC>,
                          pto::Stride<1, 1, 1, 1, 1>>;
    using OutGlobalData =
        pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, DYNAMIC>,
                          pto::Stride<1, 1, 1, 1, 1>>;

    uint32_t addr = 0;
    const uint32_t UB_IN_ADDR = addr;
    addr += MAX_BLOCKS * MAX_BINS * sizeof(float);
    const uint32_t UB_REDUCE_TMP_ADDR = addr;
    addr += MAX_BINS * sizeof(float);
    const uint32_t UB_FLOAT_OUT_ADDR = addr;
    addr += MAX_BINS * sizeof(float);
    const uint32_t UB_OUT_ADDR = addr;
    addr += MAX_BINS * sizeof(int32_t);

    using InTile = Tile<TileType::Vec, float, MAX_BLOCKS, MAX_BINS,
                        BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    InTile in_tile(
        {static_cast<size_t>(num_blocks), static_cast<size_t>(num_bins)});
    TASSIGN(in_tile, UB_IN_ADDR);

    using ReduceTmpTile =
        Tile<TileType::Vec, float, 1, MAX_BINS, BLayout::RowMajor, 1, DYNAMIC>;
    ReduceTmpTile reduce_tmp_tile(num_bins);
    TASSIGN(reduce_tmp_tile, UB_REDUCE_TMP_ADDR);
    TEXPANDS(reduce_tmp_tile, 0.0f);

    using FloatOutTile =
        Tile<TileType::Vec, float, 1, MAX_BINS, BLayout::RowMajor, 1, DYNAMIC>;
    FloatOutTile float_out_tile(num_bins);
    TASSIGN(float_out_tile, UB_FLOAT_OUT_ADDR);
    TEXPANDS(float_out_tile, 0.0f);

    using OutTile = Tile<TileType::Vec, int32_t, 1, MAX_BINS, BLayout::RowMajor,
                         1, DYNAMIC>;
    OutTile out_tile(num_bins);
    TASSIGN(out_tile, UB_OUT_ADDR);
    TEXPANDS(out_tile, static_cast<int32_t>(0));

    // Load all block counts into UB row by row to match 2D tile padding
    using InRowTile =
        Tile<TileType::Vec, float, 1, MAX_BINS, BLayout::RowMajor, 1, DYNAMIC>;
    InRowTile row_tile(static_cast<size_t>(num_bins));
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int32_t b = 0; b < num_blocks; ++b) {
      InGlobalData z_local_gm(z_local + b * num_bins, {num_bins});
      TASSIGN(row_tile, UB_IN_ADDR + b * MAX_BINS * sizeof(float));
      TLOAD(row_tile, z_local_gm);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // TCOLSUM reduces along the row dimension (num_blocks)
    TCOLSUM(float_out_tile, in_tile, reduce_tmp_tile, true);

    TCVT(out_tile, float_out_tile, RoundMode::CAST_RINT);

    // --- Final Store to Global Memory ---
    OutGlobalData z_gm(z, {num_bins});

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(z_gm, out_tile);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}

extern "C" __global__ AICORE void histogram_fp16(GM_ADDR x, GM_ADDR z_local,
                                                 const uint32_t in_length,
                                                 const int32_t num_bins,
                                                 const float min_val,
                                                 const float max_val) {
  runTLocalHistogram<half, DEFAULT_TILE_SIZE>((__gm__ half*)x,
                                              (__gm__ float*)z_local, in_length,
                                              num_bins, min_val, max_val);
}

extern "C" __global__ AICORE void histogram_fp32(GM_ADDR x, GM_ADDR z_local,
                                                 const uint32_t in_length,
                                                 const int32_t num_bins,
                                                 const float min_val,
                                                 const float max_val) {
  runTLocalHistogram<float, DEFAULT_TILE_SIZE>(
      (__gm__ float*)x, (__gm__ float*)z_local, in_length, num_bins, min_val,
      max_val);
}

extern "C" __global__ AICORE void histogram_final(GM_ADDR z_local, GM_ADDR z,
                                                  const int32_t num_bins,
                                                  const int32_t num_blocks) {
  runTHistogramFinal<0>((__gm__ float*)z_local, (__gm__ int32_t*)z, num_bins,
                        num_blocks);
}

#endif
