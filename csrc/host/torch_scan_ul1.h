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

#include "aclrtlaunch_scan_ul1_fp16.h"
#include "aclrtlaunch_scan_ul1_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Single Cube scan
 *
 * @param [in] x Input vector
 * @return at::Tensor vector result of the scan operation.
 */
at::Tensor run_scan_ul1(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = at::kFloat;

  if (!(dtype == at::kHalf or dtype == at::kFloat)) {
    throw std::runtime_error(
        "Unsupported dtype for scan_ul1 kernel. Supports only fp16/fp32");
  }

  const uint32_t scan_size = static_cast<uint32_t>(x.size(-1));
  if (x.dim() != 1) {
    throw std::runtime_error("Only 1D scan is supported.\n");
  }

  constexpr uint32_t block_dim = 1;

  const at::Tensor scan = at::zeros(
      {scan_size}, at::TensorOptions().dtype(dtype_out).device(device));

  const uint32_t matrix_size = ceil(sqrt(scan_size));

  // FIXME: pad to support other sizes
  if (matrix_size % 16 != 0) {
    throw std::runtime_error(
        "Matrix size must be a multiple of 16. Matrix size: " +
        std::to_string(matrix_size));
  }

  // FIXME: use vector or scalar cores to generate O, U and L directly on the
  // device Upper triangular matrix
  const at::Tensor u =
      torch::triu(torch::ones({matrix_size, matrix_size},
                              at::TensorOptions().dtype(dtype).device(device)));
  // Lower triangular matrix
  const at::Tensor l =
      torch::tril(torch::ones({matrix_size, matrix_size},
                              at::TensorOptions().dtype(dtype).device(device)),
                  -1);
  // Ones matrix
  const at::Tensor o =
      torch::ones({matrix_size, matrix_size},
                  at::TensorOptions().dtype(dtype).device(device));

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(scan_ul1_fp16, block_dim, x, o, u, l, scan, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(scan_ul1_fp32, block_dim, x, o, u, l, scan, matrix_size);
  }

  return scan;
}
}  // namespace pto_isa_ops
