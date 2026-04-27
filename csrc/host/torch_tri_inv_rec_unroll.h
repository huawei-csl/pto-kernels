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

#include "aclrtlaunch_tri_inv_rec_unroll_fp16.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Triangular inverse using the "recursive unroll" method.
 *
 * Note: supports fp16 and bf16 input dtypes. Output is always fp16.
 *
 * @param M Input tensor containing square matrices on the last two dimensions.
 * @param is_bsnd_format A boolean flag indicating if the matrix is in BSND
 * format. If false, then each matrix / tile is stored in consecutive positions
 * in memory, and thus we define num_bsnd_heads=0. If true, then the matrices
 * are stored in "strided mode". In this case we define:
 * num_bsnd_heads=M.size(-2), which is used to do strided load / store ops.
 * @param cu_seqlens A 1-dimensional torch tensor that contains the lengths
 * of each input sequence (it is the cummulative sum of the lengths)
 * @return at::Tensor Tensor containing inverses of input matrices.
 */
at::Tensor run_tri_inv_rec_unroll(
    const at::Tensor& M, const bool is_bsnd_format = false,
    const at::Tensor& cu_seqlens = at::zeros({1})) {
  const at::Device device = M.options().device();
  const auto dtype = M.options().dtype();

  at::Tensor M_half;
  if (dtype == at::kBFloat16) {
    M_half = M.to(at::kHalf);
  }
  if ((dtype != at::kHalf) and (dtype != at::kBFloat16)) {
    throw std::runtime_error(
        "Unsupported dtype for tri_inv_rec_unroll kernel. Supports only "
        "fp16 and bf16.");
  }

  const uint32_t matrix_size = static_cast<uint32_t>(M.size(-1));
  const uint32_t num_bsnd_heads =
      is_bsnd_format ? static_cast<uint32_t>(M.size(-2)) : 0;

  const uint32_t num_elems = static_cast<uint32_t>(M.numel());
  uint32_t total_tiles = 0;
  if (is_bsnd_format && (cu_seqlens.numel() > 1)) {
    for (int j = 1; j < cu_seqlens.size(0); ++j) {
      const uint32_t this_seq_len = static_cast<uint32_t>(
          cu_seqlens[j].item<int>() - cu_seqlens[j - 1].item<int>());
      total_tiles +=
          static_cast<uint32_t>((this_seq_len + matrix_size - 1) / matrix_size);
    }
    total_tiles = total_tiles * num_bsnd_heads;
  } else {
    total_tiles =
        static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));
  }
  uint32_t block_dim = GetNumCubeCores();
  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  const at::Tensor M_inv =
      at::zeros_like(M, at::TensorOptions().dtype(at::kHalf).device(device));

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  if (dtype == at::kBFloat16) {
    EXEC_KERNEL_CMD(tri_inv_rec_unroll_fp16, block_dim, M_inv, M_half,
                    matrix_size, total_tiles, num_bsnd_heads, cu_seqlens_ptr);
  } else if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(tri_inv_rec_unroll_fp16, block_dim, M_inv, M, matrix_size,
                    total_tiles, num_bsnd_heads, cu_seqlens_ptr);
  }

  return M_inv;
}
}  // namespace pto_isa_ops
