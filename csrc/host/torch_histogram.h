/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "aclrtlaunch_histogram_fp16.h"
#include "aclrtlaunch_histogram_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/// Number of elements per tile load – must match HISTOGRAM_TILE_LEN in the
/// kernel source.
static constexpr uint32_t kHistogramTileLen = 64;

/// Maximum supported number of bins – must match HISTOGRAM_MAX_N_BINS in the
/// kernel source.
static constexpr int64_t kHistogramMaxNBins = 1024;

/**
 * @brief Computes a histogram matching the semantics of `torch.histogram`.
 *
 * Runs on Ascend NPU using the `histogram_fp16` / `histogram_fp32` PTO-ISA
 * kernel.  Each AIV core accumulates a partial histogram over its slice of
 * the input; the partial results are summed on-device and the final counts
 * together with the bin-edge tensor are returned.
 *
 * Limitations imposed by the tile-based PTO-ISA kernel:
 *   - Input dtype must be `torch.float16` or `torch.float32`.
 *   - `input.numel()` must be a positive multiple of
 *     `kHistogramTileLen` (64).
 *   - `1 <= bins <= kHistogramMaxNBins` (1024).
 *   - `range_min < range_max`.
 *
 * @param [in] input      Input tensor (fp16 or fp32).  May have any shape;
 *                        it is flattened internally.
 * @param [in] bins       Number of equal-width histogram bins.
 * @param [in] range_min  Inclusive lower bound of the histogram range.
 * @param [in] range_max  Exclusive upper bound for interior bins; the
 *                        rightmost bin edge equals `range_max`.
 *
 * @return A pair `(hist, bin_edges)` where
 *   - `hist` is a 1-D float32 tensor of length `bins` on the same device as
 *     `input`, containing the element counts for each bin.
 *   - `bin_edges` is a 1-D float32 tensor of length `bins + 1` on the same
 *     device as `input`, containing the monotonically increasing bin edges.
 */
std::tuple<at::Tensor, at::Tensor> run_histogram(const at::Tensor& input,
                                                 int64_t bins, double range_min,
                                                 double range_max) {
  const auto dtype = input.options().dtype();

  if (dtype != at::kHalf && dtype != at::kFloat) {
    throw std::runtime_error(
        "pto_histogram supports only fp16 and fp32 input tensors.");
  }

  if (bins < 1 || bins > kHistogramMaxNBins) {
    throw std::runtime_error(
        "pto_histogram: bins must be between 1 and 1024 (inclusive).");
  }

  if (range_min >= range_max) {
    throw std::runtime_error(
        "pto_histogram: range_min must be strictly less than range_max.");
  }

  const at::Tensor flat_input = input.flatten();
  const uint32_t total_len = static_cast<uint32_t>(flat_input.numel());

  if (total_len == 0) {
    throw std::runtime_error("pto_histogram: input tensor must be non-empty.");
  }

  if (total_len % kHistogramTileLen != 0) {
    throw std::runtime_error(
        "pto_histogram: input length must be a multiple of 64 (tile length).");
  }

  // One block per tile so that all tiles are processed in parallel.
  const uint32_t block_dim = total_len / kHistogramTileLen;

  // Allocate partial-histogram buffer: [block_dim, kHistogramMaxNBins].
  // Initialised to zero so that unused bin slots are already correct.
  at::Tensor partial_hist =
      at::zeros({static_cast<int64_t>(block_dim), kHistogramMaxNBins},
                input.options().dtype(at::kFloat));

  const float f_range_min = static_cast<float>(range_min);
  const float f_range_max = static_cast<float>(range_max);
  const uint32_t u_bins = static_cast<uint32_t>(bins);

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(histogram_fp16, block_dim, flat_input, partial_hist,
                    total_len, u_bins, f_range_min, f_range_max);
  } else {
    EXEC_KERNEL_CMD(histogram_fp32, block_dim, flat_input, partial_hist,
                    total_len, u_bins, f_range_min, f_range_max);
  }

  // Sum partial histograms across all blocks and slice to the requested number
  // of bins.  Shape: [block_dim, kHistogramMaxNBins] -> [bins].
  at::Tensor hist = partial_hist.narrow(1, 0, bins).sum(0);

  // Compute bin edges: linearly spaced from range_min to range_max.
  at::Tensor bin_edges = at::linspace(range_min, range_max, bins + 1,
                                      input.options().dtype(at::kFloat));

  return std::make_tuple(hist, bin_edges);
}

}  // namespace pto_isa_ops
