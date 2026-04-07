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

#include "aclrtlaunch_histogram_final.h"
#include "aclrtlaunch_histogram_fp16.h"
#include "aclrtlaunch_histogram_fp32.h"
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
  constexpr uint32_t TILE_SIZE = 512;
  const uint32_t block_dim = GetNumVectorCores();

  TORCH_CHECK(total_len % TILE_SIZE == 0,
              "total number of elements must be divisible by TILE_SIZE");
  TORCH_CHECK(bins <= 256, "bins must be <= 256");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

  const auto dtype = x.options().dtype();
  const auto device = x.options().device();

  // Allocate a 1D tensor sized `[block_dim * bins]` for the local histogram
  // counts.
  auto z_local_opts =
      at::TensorOptions()
          .dtype(
              at::kFloat)  // Local (per-core) histogram counts will be floats
          .device(device);
  at::Tensor z_local = at::zeros({block_dim * bins}, z_local_opts);

  // Allocate a 1D tensor sized `[bins]` for the histogram.
  auto z_opts = at::TensorOptions()
                    .dtype(at::kInt)  // The final result will be int32 counts
                    .device(device);
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

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(histogram_fp16, block_dim, x, z_local, total_len, num_bins,
                    f_min_val, f_max_val);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(histogram_fp32, block_dim, x, z_local, total_len, num_bins,
                    f_min_val, f_max_val);
  } else {
    throw std::runtime_error("Unsupported dtype for `pto_histogram` kernel");
  }

  const uint32_t reduce_dim = 1;
  EXEC_KERNEL_CMD(histogram_final, reduce_dim, z_local, z, num_bins, block_dim);

  return z;
}
}  // namespace pto_isa_ops
