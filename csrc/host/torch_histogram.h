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

#include "aclrtlaunch_vhistogram_local_fp16.h"
#include "aclrtlaunch_vhistogram_local_fp32.h"
#include "aclrtlaunch_vhistogram_reduce_fp16.h"
#include "aclrtlaunch_vhistogram_reduce_fp32.h"
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
  // // FIXME: tile length is fixed to 64 for now
  // constexpr uint32_t TILE_LEN = 64;
  // const uint32_t total_tiles = total_len / TILE_LEN;
  uint32_t num_cores = GetNumVectorCores();
  // if (total_tiles < num_cores) {
  //   num_cores = total_tiles;
  // }

  const auto dtype = x.options().dtype();
  const auto device = x.options().device();
  auto z_opts = at::TensorOptions()
    .dtype(at::kInt)      // Set data type to int32 for histogram counts
    .device(device);
  // Allocate a 1D tensor sized `[bins]` for the histogram.
  at::Tensor z = at::zeros({bins}, z_opts);
  at::Tensor z_local = at::zeros({num_cores, bins}, z_opts);
  
  const auto num_bins = static_cast<int32_t>(bins);

  if (min_val == 0.0 && max_val == 0.0) {
    const double tensor_min = x.min().item<double>();
    const double tensor_max = x.max().item<double>();
    min_val = tensor_min;
    max_val = tensor_max;
  }

  const auto f_min_val = static_cast<float>(min_val);
  const auto f_max_val = static_cast<float>(max_val);

  // Phase 1: Launch one kernel per core to compute local histograms
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(vhistogram_local_fp16, num_cores, x, z_local, total_len, num_bins,
                    f_min_val, f_max_val);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(vhistogram_local_fp32, num_cores, x, z_local, total_len, num_bins,
                    f_min_val, f_max_val);
  } else {
    throw std::runtime_error("Unsupported dtype for `pto_histogram` kernel");
  }

  // Phase 2: Launch a single kernel to reduce all local histograms
  const uint32_t num_reduce_cores = 1;
  EXEC_KERNEL_CMD(vhistogram_reduce_fp32, num_reduce_cores, z, z_local, num_bins, num_cores);

  return z;
}
}  // namespace pto_isa_ops
