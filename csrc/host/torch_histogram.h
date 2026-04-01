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

#include "aclrtlaunch_vhistogram_fp16.h"
#include "aclrtlaunch_vhistogram_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Computes the histogram of a tensor.
 *
 * @param [in] x Input tensor of dtype fp16 or fp32.
 * @param [in] bins Number of histogram bins.
 * @param [in] min_val Lower bound of the range.
 * @param [in] max_val Upper bound of the range.
 * @return at::Tensor Computed histogram tensor.
 */
at::Tensor run_histogram(const at::Tensor& x, int64_t bins = 100,
                         double min_val = 0.0, double max_val = 0.0) {
  const uint32_t total_len = x.numel();
  constexpr uint32_t TILE_LEN = 64;
  constexpr uint32_t TILE_SIZE = TILE_LEN * TILE_LEN;
  // const uint32_t block_dim = (total_len + TILE_SIZE - 1) / TILE_SIZE;
  // const uint32_t block_dim = GetNumVectorCores();;
  const uint32_t block_dim = 1;

  TORCH_CHECK(total_len / TILE_SIZE != 0,
              "total number of elements must be divisible by 64 * 64");
  TORCH_CHECK(bins <= 1024, "bins must be <= 1024");

  const auto dtype = x.options().dtype();
  const auto device = x.options().device();
  auto z_opts =
      at::TensorOptions()
          .dtype(at::kInt)  // Set data type to int32 for histogram counts
          .device(device);
  // Allocate a 1D tensor sized `[bins]` for the histogram.
  at::Tensor z = at::zeros({bins}, z_opts);

  const auto num_bins = static_cast<int32_t>(bins);

  if (min_val == 0.0 && max_val == 0.0) {
    const double tensor_min = x.min().item<double>();
    const double tensor_max = x.max().item<double>();
    min_val = tensor_min;
    max_val = tensor_max;
  }

  const auto f_min_val = static_cast<float>(min_val);
  const auto f_max_val = static_cast<float>(max_val);
  const float f_bin_width = (f_max_val - f_min_val) / (float)num_bins;

  at::Tensor x_contig = x.contiguous();  // Just in case
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(vhistogram_fp16, block_dim, x_contig, z, total_len,
                    num_bins, f_min_val, f_max_val, f_bin_width);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(vhistogram_fp32, block_dim, x_contig, z, total_len,
                    num_bins, f_min_val, f_max_val, f_bin_width);
  } else {
    throw std::runtime_error("Unsupported dtype for `pto_histogram` kernel");
  }

  return z;
}
}  // namespace pto_isa_ops
