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

#include "aclrtlaunch_tri_inv_rec_unroll_bf16.h"
#include "aclrtlaunch_tri_inv_rec_unroll_fp16.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Triangular inverse using the "recursive unroll" method
 *
 * @param M Input tensor containing square matrices on the last two dimensions.
 * @param is_bsnd_format A boolean flag indicating if the matrix is in BSND
 * format. If false, then each matrix / tile is stored in consecutive positions
 * in memory, and thus we define num_bsnd_heads=0. If true, then the matrices
 * are stored in "strided mode". In this case we define:
 * num_bsnd_heads=M.size(-2), which is used to do strided load / store ops.
 * @return at::Tensor Tensor containing inverses of input matrices.
 */
at::Tensor run_tri_inv_rec_unroll(const at::Tensor& M,
                                  const bool is_bsnd_format = false) {
  const at::Device device = M.options().device();
  const auto dtype = M.options().dtype();
  const auto dtype_out = at::kFloat;
  if (!(dtype == at::kHalf) && !(dtype == at::kBFloat16)) {
    throw std::runtime_error(
        "Unsupported dtype for tri_inv_rec_unroll kernel. Supports only "
        "fp16 and bf16");
  }

  const uint32_t matrix_size = static_cast<uint32_t>(M.size(-1));
  const uint32_t num_bsnd_heads =
      is_bsnd_format ? static_cast<uint32_t>(M.size(-2)) : 0;

  const uint32_t num_elems = static_cast<uint32_t>(M.numel());
  const uint32_t total_tiles =
      static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));

  uint32_t block_dim = GetNumCubeCores();
  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  const at::Tensor M_inv =
      at::zeros_like(M, at::TensorOptions().dtype(dtype_out).device(device));

  const at::Tensor I_neg =
      at::zeros({matrix_size, matrix_size},
                at::TensorOptions().dtype(dtype).device(device));
  I_neg.fill_diagonal_(-1);

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(tri_inv_rec_unroll_fp16, block_dim, M_inv, M, I_neg,
                    matrix_size, total_tiles, num_bsnd_heads);
  } else if (dtype == at::kBFloat16) {
    EXEC_KERNEL_CMD(tri_inv_rec_unroll_bf16, block_dim, M_inv, M, I_neg,
                    matrix_size, total_tiles, num_bsnd_heads);
  }

  return M_inv;
}
}  // namespace pto_isa_ops
