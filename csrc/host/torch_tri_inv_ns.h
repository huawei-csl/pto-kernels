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

  TORCH_CHECK(dtype == at::kHalf, "tri_inv_ns: dtype must be fp16, got ",
              dtype);
  const uint32_t n = static_cast<uint32_t>(M.size(-1));
  TORCH_CHECK(n == static_cast<uint32_t>(M.size(-2)),
              "tri_inv_ns: only square matrices are supported");

  if (scale_value == 0) {
    scale_value = 2 * n;
  }

  const uint32_t num_matrices = static_cast<uint32_t>(M.numel()) / (n * n);

  if (num_iters == 0) {
    num_iters = static_cast<uint32_t>(std::ceil(2.0f * std::log2(n)));
    num_iters = std::max<uint32_t>(num_iters, 12);
  }
  uint32_t block_dim = GetNumCubeCores();
  if (num_matrices < block_dim) {
    block_dim = num_matrices;
  }
  const at::Tensor I_neg = -1 * at::eye(n, M.options());
  const at::Tensor I_scaled = I_neg / (-scale_value);

  const at::Tensor M_inv =
      at::zeros_like(M, at::TensorOptions().dtype(dtype_out).device(device));

  EXEC_KERNEL_CMD(tri_inv_ns_fp16, block_dim, M_inv, M, I_neg, I_scaled, n,
                  num_iters, num_matrices);

  return M_inv;
}
}  // namespace pto_isa_ops
