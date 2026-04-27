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

#include "aclrtlaunch_triv_inv_col_sweep_fp16.h"
#include "aclrtlaunch_triv_inv_col_sweep_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Runs triangular inverse using vector-only column sweep method.
 *
 * @param [in] x Input tensor whose last two dimensions contain square
 * triangular matrices to invert.
 * @return at::Tensor Tensor of same shape as input where matrices are inverted.
 */

at::Tensor run_tri_inv(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  TORCH_CHECK(x.dim() >= 2,
              "tri_inv: input tensor must have at least 2 dimensions, got ",
              x.dim());

  const uint32_t matrix_size = static_cast<uint32_t>(x.size(-1));
  TORCH_CHECK(matrix_size == static_cast<uint32_t>(x.size(-2)),
              "tri_inv: only square matrices are supported");

  const uint32_t num_elems = static_cast<uint32_t>(x.numel());
  const uint32_t total_tiles =
      static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));

  uint32_t block_dim = GetNumCubeCores();
  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  const at::Tensor z = at::empty_like(x);

  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "tri_inv: dtype must be fp16 or float32, got ", dtype);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(triv_inv_col_sweep_fp16, block_dim, x, z, num_elems,
                    matrix_size);
  } else {
    EXEC_KERNEL_CMD(triv_inv_col_sweep_fp32, block_dim, x, z, num_elems,
                    matrix_size);
  }

  return z;
}
}  // namespace pto_isa_ops
