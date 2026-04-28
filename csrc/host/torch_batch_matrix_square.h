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

#include "aclrtlaunch_batch_matrix_square_fp16.h"
#include "aclrtlaunch_batch_matrix_square_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Batch matrix square given a tensor whose last two dimensions form a
 * square matrix.
 *
 * @param [in] x Input tensor whose last two dimensions form a square matrix.
 * @return at::Tensor Tensor of same shape as `x` that contains matrix squares.
 */
at::Tensor run_batch_matrix_square(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = at::kFloat;

  TORCH_CHECK(device.type() == DEVICE_TYPE,
              "batch_matrix_square: tensor must be on NPU, got ", device);
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "batch_matrix_square: dtype must be fp16 or float32, got ",
              dtype);

  const uint32_t matrix_size = static_cast<uint32_t>(x.size(-1));
  TORCH_CHECK(matrix_size == static_cast<uint32_t>(x.size(-2)),
              "batch_matrix_square: only square matrices are supported");

  const uint32_t block_dim = x.size(0);

  const at::Tensor z =
      at::zeros({block_dim, matrix_size, matrix_size},
                at::TensorOptions().dtype(dtype_out).device(device));

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(batch_matrix_square_fp16, block_dim, z, x, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(batch_matrix_square_fp32, block_dim, z, x, matrix_size);
  }

  return z;
}
}  // namespace pto_isa_ops
