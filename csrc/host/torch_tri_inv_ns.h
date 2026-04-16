/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include <cmath>

#include "aclrtlaunch_tri_inv_ns_fp16.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Triangular inverse using Newton–Schulz iterations.
 *
 * Implements the following algorithm:
 * A = I + M
 * X = I * scale
 * for _ in range(num_iters):
 *     Y = A @ X
 *     X = X @ (2*I - Y)
 * return X
 *
 * @param M  Input tensor of strictly upper-triangular matrices (..., n,
 * n), dtype fp16. The full matrix inverted by the algorithm is A = I + M.
 * @param num_iters  Number of Newton–Schulz iterations (0 = auto).
 * @param scale_value  Value to scale the initial guess. Defaults to zero, which
 * sets scale_value = 2 * n, where n is the size of the matrices.
 * @return at::Tensor Tensor of approximate inverses in fp32, same batch shape
 * as M.
 */
at::Tensor run_tri_inv_ns(const at::Tensor& M, uint32_t num_iters = 0,
                          float scale_value = 0) {
  const at::Device device = M.options().device();
  const auto dtype = M.options().dtype();
  const auto dtype_out = at::kFloat;

  if (!(dtype == at::kHalf)) {
    throw std::runtime_error(
        "Unsupported dtype for tri_inv_ns kernel. Supports only fp16");
  }
  const uint32_t n = static_cast<uint32_t>(M.size(-1));
  if (n != static_cast<uint32_t>(M.size(-2))) {
    throw std::runtime_error("Only square matrices are supported.\n");
  }

  if (scale_value == 0) {
    scale_value = 2 * n;
  }

  const uint32_t num_matrices = static_cast<uint32_t>(M.numel()) / (n * n);

  const auto opts_in = at::TensorOptions().dtype(dtype).device(device);

  if (num_iters == 0) {
    num_iters = static_cast<uint32_t>(std::ceil(2.0f * std::log2(n)));
    num_iters = std::max<uint32_t>(num_iters, 8);
  }

  const at::Tensor I_eye = at::eye(n, opts_in);
  const at::Tensor I_scaled =
      (I_eye / scale_value).to(dtype).contiguous();  // per matrix

  const at::Tensor I_neg = -I_eye.contiguous();

  const at::Tensor M_inv_raw =
      at::zeros_like(M, at::TensorOptions().dtype(dtype_out).device(device));

  EXEC_KERNEL_CMD(tri_inv_ns_fp16, num_matrices, M_inv_raw, M, I_neg, I_scaled,
                  n, num_iters);

  return M_inv_raw;
}
}  // namespace pto_isa_ops
