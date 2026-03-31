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

#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"

using namespace pto;

/// Maximum number of histogram bins supported by the kernel.
constexpr uint32_t HISTOGRAM_MAX_N_BINS = 1024;

/// Number of elements processed per tile load.
constexpr uint32_t HISTOGRAM_TILE_LEN = 64;

/**
 * @brief Computes a partial histogram for a slice of the input tensor.
 *
 * Each block processes its slice of the input and accumulates element counts
 * into a local histogram in the Unified Buffer.  The per-block partial
 * histograms are written to consecutive segments of `partial_hist` in global
 * memory.  The host is responsible for summing them.
 *
 * Values that fall below `range_min` are placed in bin 0; values that fall at
 * or above `range_max` are placed in bin `n_bins - 1`.  This matches the
 * clamping convention used by `torch.histogram` for out-of-range values.
 *
 * @tparam T  Input data type.  Supports `half` (fp16) or `float` (fp32).
 *
 * @param [in]  input        Pointer to the 1-D input array in global memory.
 * @param [out] partial_hist Pointer to the output array in global memory with
 *                           shape [get_block_num(), HISTOGRAM_MAX_N_BINS].
 * @param [in]  n_elements   Total number of elements in `input`.  Must be a
 *                           multiple of (get_block_num() * HISTOGRAM_TILE_LEN).
 * @param [in]  n_bins       Number of histogram bins.  Must satisfy
 *                           1 <= n_bins <= HISTOGRAM_MAX_N_BINS.
 * @param [in]  range_min    Lower bound of the histogram range (inclusive).
 * @param [in]  range_max    Upper bound of the histogram range (exclusive for
 *                           interior bins, inclusive for the last bin edge).
 */
template <typename T>
AICORE void runHistogram(__gm__ T* input, __gm__ float* partial_hist,
                         uint32_t n_elements, uint32_t n_bins, float range_min,
                         float range_max) {
  // ---- UB memory layout ----
  // [0, DATA_TILE_BYTES)                              : input data tile
  // [DATA_TILE_BYTES, DATA_TILE_BYTES + HIST_BYTES)   : local histogram tile
  constexpr uint32_t DATA_TILE_UB_ADDR = 0;
  constexpr uint32_t DATA_TILE_BYTES =
      HISTOGRAM_TILE_LEN * sizeof(T);  // 32-byte aligned for T=fp16/fp32
  constexpr uint32_t HIST_UB_ADDR = DATA_TILE_BYTES;

  // ---- Global tensor descriptors ----
  using GlobalInputShape = pto::Shape<1, 1, 1, 1, HISTOGRAM_TILE_LEN>;
  using GlobalInputStride = pto::Stride<1, 1, 1, HISTOGRAM_TILE_LEN, 1>;
  using GlobalInputData =
      pto::GlobalTensor<T, GlobalInputShape, GlobalInputStride>;

  using GlobalHistShape = pto::Shape<1, 1, 1, 1, HISTOGRAM_MAX_N_BINS>;
  using GlobalHistStride = pto::Stride<1, 1, 1, HISTOGRAM_MAX_N_BINS, 1>;
  using GlobalHistData =
      pto::GlobalTensor<float, GlobalHistShape, GlobalHistStride>;

  // ---- UB tile types ----
  using InputTile =
      Tile<TileType::Vec, T, 1, HISTOGRAM_TILE_LEN, BLayout::RowMajor, -1, -1>;
  using HistTile = Tile<TileType::Vec, float, 1, HISTOGRAM_MAX_N_BINS,
                        BLayout::RowMajor, -1, -1>;

  InputTile input_tile(1, HISTOGRAM_TILE_LEN);
  TASSIGN(input_tile, DATA_TILE_UB_ADDR);

  HistTile local_hist(1, HISTOGRAM_MAX_N_BINS);
  TASSIGN(local_hist, HIST_UB_ADDR);

  // Initialise local histogram to zero
  set_mask_norm();
  set_vector_mask(-1, -1);
  TEXPANDS(local_hist, 0.0f);

  // ---- Work distribution ----
  const uint32_t num_tiles = n_elements / HISTOGRAM_TILE_LEN;
  const uint32_t tiles_per_block = num_tiles / get_block_num();
  const uint32_t block_tile_start = block_idx * tiles_per_block;

  // Pre-compute bin lookup constants
  const float bin_width = (range_max - range_min) / static_cast<float>(n_bins);
  const float inv_bin_width = (bin_width > 0.0f) ? (1.0f / bin_width) : 0.0f;
  const int32_t n_bins_i = static_cast<int32_t>(n_bins);

  // ---- Pipeline initialisation ----
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

  // ---- Main loop: process tiles ----
  for (uint32_t tile_id = 0; tile_id < tiles_per_block; tile_id++) {
    const uint32_t offset = (block_tile_start + tile_id) * HISTOGRAM_TILE_LEN;

    GlobalInputData global_in(input + offset);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(input_tile, global_in);
    pipe_barrier(PIPE_ALL);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Switch to scalar pipeline for element-wise bin assignment
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    for (uint32_t i = 0; i < HISTOGRAM_TILE_LEN; i++) {
      const float val = static_cast<float>(input_tile.GetValue(i));
      int32_t bin = static_cast<int32_t>((val - range_min) * inv_bin_width);
      // Clamp to valid bin range
      if (bin < 0) {
        bin = 0;
      } else if (bin >= n_bins_i) {
        bin = n_bins_i - 1;
      }
      local_hist.SetValue(bin, local_hist.GetValue(bin) + 1.0f);
    }

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

  // ---- Write partial histogram to global memory ----
  GlobalHistData global_hist_out(partial_hist +
                                 block_idx * HISTOGRAM_MAX_N_BINS);

  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  TSTORE(global_hist_out, local_hist);
  pipe_barrier(PIPE_ALL);

  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

extern "C" __global__ AICORE void histogram_fp16(
    GM_ADDR input, GM_ADDR partial_hist, uint32_t n_elements, uint32_t n_bins,
    float range_min, float range_max) {
  runHistogram<half>((__gm__ half*)input, (__gm__ float*)partial_hist,
                     n_elements, n_bins, range_min, range_max);
}

extern "C" __global__ AICORE void histogram_fp32(
    GM_ADDR input, GM_ADDR partial_hist, uint32_t n_elements, uint32_t n_bins,
    float range_min, float range_max) {
  runHistogram<float>((__gm__ float*)input, (__gm__ float*)partial_hist,
                      n_elements, n_bins, range_min, range_max);
}

#endif
