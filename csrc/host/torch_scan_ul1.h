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

  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "scan_ul1: dtype must be fp16 or float32, got ", dtype);

  const uint32_t scan_size = static_cast<uint32_t>(x.size(-1));
  TORCH_CHECK(x.dim() == 1, "scan_ul1: only 1D tensors are supported, got ",
              x.dim(), "D");

  constexpr uint32_t block_dim = 1;

  const at::Tensor scan = at::zeros(
      {scan_size}, at::TensorOptions().dtype(dtype_out).device(device));

  const uint32_t matrix_size = ceil(sqrt(scan_size));

  // FIXME: pad to support other sizes
  TORCH_CHECK(matrix_size % 16 == 0,
              "scan_ul1: matrix size must be a multiple of 16, got ",
              matrix_size);

  // FIXME: use vector or scalar cores to generate O, U and L directly on the
  // device

  // Ones matrix
  const at::Tensor o =
      torch::ones({matrix_size, matrix_size},
                  at::TensorOptions().dtype(dtype).device(device));
  // Upper triangular matrix
  const at::Tensor u = torch::triu(o);
  // Lower triangular matrix
  const at::Tensor l = torch::tril(o, -1);

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(scan_ul1_fp16, block_dim, x, o, u, l, scan, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(scan_ul1_fp32, block_dim, x, o, u, l, scan, matrix_size);
  }

  return scan;
}
}  // namespace pto_isa_ops
