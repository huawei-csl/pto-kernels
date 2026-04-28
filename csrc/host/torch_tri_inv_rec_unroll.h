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

#include "aclrtlaunch_tri_inv_rec_unroll_fp16fp16.h"
#include "aclrtlaunch_tri_inv_rec_unroll_fp16fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Triangular inverse using the "recursive unroll" method.
 *
 * Note: supports fp16 and bf16 input dtypes. Output is fp16 or fp32.
 *
 * @param M Input tensor containing square matrices on the last two dimensions.
 * @param cu_seqlens A 1-dimensional torch tensor that contains the lengths
 * of each input sequence (it is the cummulative sum of the lengths)
 * @param is_bsnd_format A boolean flag indicating if the matrix is in BSND
 * format. If false, then each matrix / tile is stored in consecutive positions
 * in memory, and thus we define num_bsnd_heads=0. If true, then the matrices
 * are stored in "strided mode". In this case we define:
 * num_bsnd_heads=M.size(-2), which is used to do strided load / store ops.
 * @param dtype_out Output dtype, either fp16 or fp32.
 * @return at::Tensor Tensor containing inverses of input matrices (dtype is
 * fp16/fp32).
 */
at::Tensor run_tri_inv_rec_unroll(
    const at::Tensor& M, const at::Tensor& cu_seqlens = at::zeros({1}),
    const bool is_bsnd_format = false,
    const at::ScalarType dtype_out = at::ScalarType::Half) {
  const at::Device device = M.options().device();
  const auto dtype = M.options().dtype();

  TORCH_CHECK(device.type() == DEVICE_TYPE,
              "tri_inv_ns: tensor must be on NPU, got ", device);
  TORCH_CHECK(dtype_out == at::kHalf || dtype_out == at::kFloat,
              "tri_inv_rec_unroll: dtype_out must be fp16 or float32, got ",
              dtype_out);
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "tri_inv_rec_unroll: input dtype must be fp16 or bfloat16, got ",
              dtype);

  at::Tensor M_half;
  if (dtype == at::kBFloat16) {
    M_half = M.to(at::kHalf);
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

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  const at::Tensor M_input = (dtype == at::kBFloat16) ? M_half : M;
  if (dtype_out == at::kHalf) {
    EXEC_KERNEL_CMD(tri_inv_rec_unroll_fp16fp16, block_dim, M_inv, M_input,
                    matrix_size, total_tiles, num_bsnd_heads, cu_seqlens_ptr);
  } else if (dtype_out == at::kFloat) {
    EXEC_KERNEL_CMD(tri_inv_rec_unroll_fp16fp32, block_dim, M_inv, M_input,
                    matrix_size, total_tiles, num_bsnd_heads, cu_seqlens_ptr);
  }

  return M_inv;
}
}  // namespace pto_isa_ops
