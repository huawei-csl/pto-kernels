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

#include "aclrtlaunch_simple_matmul_bf16.h"
#include "aclrtlaunch_simple_matmul_fp16.h"
#include "aclrtlaunch_simple_matmul_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Simple matrix multiplication A @ B.
 *
 * @param [in] a Input left martrix
 * @param [in] b Input right matrix
 * @return at::Tensor Matrix multiplication result of a @ b.
 */
at::Tensor run_simple_matmul(const at::Tensor& a, const at::Tensor& b) {
  const at::Device device = a.options().device();
  const auto dtype = a.options().dtype();
  const auto dtype_out = at::kFloat;

  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kFloat || dtype == at::kBFloat16,
      "simple_matmul: dtype must be fp16, bf16, or float32, got ", dtype);

  const uint32_t matrix_size = static_cast<uint32_t>(a.size(-1));
  TORCH_CHECK(matrix_size == static_cast<uint32_t>(a.size(-2)),
              "simple_matmul: only square matrices are supported");

  constexpr uint32_t block_dim = 1;

  const at::Tensor c =
      at::ones({matrix_size, matrix_size},
               at::TensorOptions().dtype(dtype_out).device(device));

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(simple_matmul_fp16, block_dim, a, b, c, matrix_size);
  } else if (dtype == at::kBFloat16) {
    EXEC_KERNEL_CMD(simple_matmul_bf16, block_dim, a, b, c, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(simple_matmul_fp32, block_dim, a, b, c, matrix_size);
  }

  return c;
}
}  // namespace pto_isa_ops
