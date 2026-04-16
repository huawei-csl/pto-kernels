/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch/library.h>

#include "utils.h"

extern "C" void call_tri_inv_rec_unroll_fp16(
    uint32_t blockDim, aclrtStream stream, void* M_inv, void* M, void* I_neg,
    uint32_t matrix_size, uint32_t num_matrices, uint32_t num_bsnd_heads,
    void* cu_seqlens);

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
 * @param cu_seqlens A 1-dimensional torch tensor that contains the lengths
 * of each input sequence (it is the cummulative sum of the lengths)
 * @return at::Tensor Tensor containing inverses of input matrices.
 */
at::Tensor run_tri_inv_rec_unroll(
    const at::Tensor& M, const bool is_bsnd_format = false,
    const at::Tensor& cu_seqlens = at::zeros({1})) {
  const at::Device device = M.options().device();
  const auto dtype = M.options().dtype();
  const auto dtype_out = at::kFloat;
  if (!(dtype == at::kHalf)) {
    throw std::runtime_error(
        "Unsupported dtype for tri_inv_rec_unroll kernel. Supports only "
        "fp16");
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
      at::zeros_like(M, at::TensorOptions().dtype(dtype_out).device(device));

  const at::Tensor I_neg =
      at::zeros({matrix_size, matrix_size},
                at::TensorOptions().dtype(dtype).device(device));
  I_neg.fill_diagonal_(-1);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  if (dtype == at::kHalf) {
    if (cu_seqlens.numel() == 1) {
      void* void_null_ptr = nullptr;
      EXEC_KERNEL_CMD(call_tri_inv_rec_unroll_fp16, block_dim, M_inv, M, I_neg,
                      matrix_size, total_tiles, num_bsnd_heads, void_null_ptr);
    } else {
      EXEC_KERNEL_CMD(call_tri_inv_rec_unroll_fp16, block_dim, M_inv, M, I_neg,
                      matrix_size, total_tiles, num_bsnd_heads, cu_seqlens);
    }
  }

  return M_inv;
}
}  // namespace pto_isa_ops
