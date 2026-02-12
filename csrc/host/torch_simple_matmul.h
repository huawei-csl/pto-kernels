/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "aclrtlaunch_simple_matmul_fp16.h"
#include "aclrtlaunch_simple_matmul_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

at::Tensor run_simple_matmul(const at::Tensor& a, const at::Tensor& b) {
  const at::Device device = a.options().device();
  const auto dtype = a.options().dtype();
  const auto dtype_out = at::kFloat;

  if (!(dtype == at::kHalf or dtype == at::kFloat)) {
    throw std::runtime_error(
        "Unsupported dtype for simple_matmul kernel. Supports only fp16/fp32");
  }

  const uint32_t matrix_size = static_cast<uint32_t>(a.size(-1));
  if (matrix_size != a.size(-2)) {
    throw std::runtime_error("Only square matrices are supported.\n");
  }

  constexpr uint32_t block_dim = 1;

  const at::Tensor c =
      at::ones({matrix_size, matrix_size},
               at::TensorOptions().dtype(dtype_out).device(device));

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(simple_matmul_fp16, block_dim, a, b, c, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(simple_matmul_fp32, block_dim, a, b, c, matrix_size);
  }

  return c;
}
}  // namespace pto_isa_ops
