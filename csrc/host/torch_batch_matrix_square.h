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

at::Tensor run_batch_matrix_square(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = at::kFloat;

  if (!(dtype == at::kHalf or dtype == at::kFloat)) {
    throw std::runtime_error(
        "Unsupported dtype for batch_matrix_square kernel. Supports only "
        "fp16/fp32");
  }

  const uint32_t matrix_size = static_cast<uint32_t>(x.size(-1));
  if (matrix_size != x.size(-2)) {
    throw std::runtime_error("Only square matrices are supported.\n");
  }

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
