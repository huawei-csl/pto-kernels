/**
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it
 under the terms and conditions of CANN Open Software License Agreement
 Version 2.0 (the "License"). Please refer to the License for details. You may
 not use this file except in compliance with the License. THIS SOFTWARE IS
 PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
 repository for the full text of the License.
*/
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "aclrtlaunch_triv_inv_col_sweep_fp16.h"
#include "aclrtlaunch_triv_inv_col_sweep_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

at::Tensor run_tri_inv(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  if (x.dim() < 2) {
    throw std::runtime_error("Input tensor must have at least 2 dimensions.\n");
  }

  const uint32_t matrix_size = static_cast<uint32_t>(x.size(-1));
  if (matrix_size != x.size(-2)) {
    throw std::runtime_error("Only square matrices are supported.\n");
  }

  const uint32_t num_elems = static_cast<uint32_t>(x.numel());
  const uint32_t block_dim =
      static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));
  const at::Tensor z = at::empty_like(x);

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(triv_inv_col_sweep_fp16, block_dim, x, z, num_elems,
                    matrix_size);

  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(triv_inv_col_sweep_fp32, block_dim, x, z, num_elems,
                    matrix_size);

  } else {
    throw std::runtime_error("Unsupported dtype for `tri_inv` kernel");
  }

  return z;
}
}  // namespace pto_isa_ops
