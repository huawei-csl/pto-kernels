/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include <pto/pto-inst.hpp>

#include "../kernel_utils.h"

using namespace pto;

#ifndef HIST_TILE_SIZE
#define HIST_TILE_SIZE 1024
#endif
constexpr uint32_t MAX_BINS = 256;
constexpr uint32_t MAX_BLOCKS = 64;

/**
 * runTLocalHistogram - Local histogram using MSCATTER with AtomicAdd
 */
template <typename T, unsigned TILE_SIZE>
AICORE void runTLocalHistogram(__gm__ T *x, __gm__ float *z_local,
                               const uint32_t total_length,
                               const int32_t num_bins, const float min_val,
                               const float max_val) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
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
  const uint32_t UB_X_PING = addr;
  addr += TILE_SIZE * sizeof(T);
  const uint32_t UB_X_PONG = addr;
  addr += TILE_SIZE * sizeof(T);
  const uint32_t UB_F32_ADDR = addr;
  addr += TILE_SIZE * sizeof(float);
  const uint32_t UB_IDX_ADDR = addr;
  addr += TILE_SIZE * sizeof(int32_t);
  const uint32_t UB_ONES_ADDR = addr;
  addr += TILE_SIZE * sizeof(float);

  InputGlobalData x_gm(x, {static_cast<int32_t>(total_length)});
  HistGlobalData z_gm(z_local + block_idx * num_bins, {num_bins});

  using InputTileData = Tile<TileType::Vec, T, 1, TILE_SIZE>;
  using F32TileData = Tile<TileType::Vec, float, 1, TILE_SIZE>;
  using IdxTileData = Tile<TileType::Vec, int32_t, 1, TILE_SIZE>;

  F32TileData f32_tile;
  TASSIGN(f32_tile, UB_F32_ADDR);

  IdxTileData idx_tile;
  TASSIGN(idx_tile, UB_IDX_ADDR);

  F32TileData ones_tile;
  TASSIGN(ones_tile, UB_ONES_ADDR);
  TEXPANDS(ones_tile, 1.0f);

  const float inv_bin_width =
      static_cast<float>(num_bins) / (max_val - min_val);

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  // --- Main Calculation Loop ---
  for (uint32_t tile_idx = start_idx, ping = 1; tile_idx < end_idx;
       ++tile_idx) {
    int offset = tile_idx * TILE_SIZE;
    TASSIGN(x_gm, x + offset);

    const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned x_base = ping ? UB_X_PING : UB_X_PONG;

    InputTileData x_tile;
    TASSIGN(x_tile, x_base);

    wait_flag(PIPE_V, PIPE_MTE2, ev);
    TLOAD(x_tile, x_gm);
    set_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE2, PIPE_V, ev);

    // Calculate bin index: idx = (val - min_val) * inv_bin_width
    TADDS(f32_tile, x_tile, -min_val);
    TMULS(f32_tile, f32_tile, inv_bin_width);

    // Convert to int32 index using Floor rounding
    TCVT(idx_tile, f32_tile, RoundMode::CAST_FLOOR);

    // Clamp indices to [0, num_bins - 1] to handle edge cases and outliers
    TMAXS(idx_tile, idx_tile, static_cast<int32_t>(0));
    TMINS(idx_tile, idx_tile, static_cast<int32_t>(num_bins - 1));

    // Atomic Scatter-Add to global memory histogram
    MSCATTER<AtomicType::AtomicAdd>(z_gm, ones_tile, idx_tile);

    set_flag(PIPE_V, PIPE_MTE2, ev);
    ping = 1 - ping;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

#endif
}

// Template parameter to avoid "no function" kernel launch error
template <unsigned UNUSED>
AICORE void runTHistogramFinal(__gm__ float *z_local, __gm__ int32_t *z,
                               const int32_t num_bins,
                               const int32_t num_blocks) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
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

#endif
}

__global__ AICORE void histogram_local_fp32(
    __gm__ void *x, __gm__ void *z_local, const uint32_t in_length,
    const int32_t num_bins, const float min_val, const float max_val) {
  runTLocalHistogram<float, HIST_TILE_SIZE>((__gm__ float *)x,
                                            (__gm__ float *)z_local, in_length,
                                            num_bins, min_val, max_val);
}

__global__ AICORE void histogram_final(__gm__ void *z_local, __gm__ void *z,
                                       const int32_t num_bins,
                                       const int32_t num_blocks) {
  runTHistogramFinal<0>((__gm__ float *)z_local, (__gm__ int32_t *)z, num_bins,
                        num_blocks);
}

extern "C" void histogram_fp32(uint32_t num_blocks, void *stream, void *x,
                               void *z_local, void *z, const uint32_t in_length,
                               const int32_t num_bins, const float min_val,
                               const float max_val) {
  histogram_local_fp32<<<num_blocks, nullptr, stream>>>(
      x, z_local, in_length, num_bins, min_val, max_val);
  histogram_final<<<1, nullptr, stream>>>(z_local, z, num_bins, num_blocks);
}
