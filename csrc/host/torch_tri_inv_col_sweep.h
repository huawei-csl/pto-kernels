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

#include "utils.h"

extern "C" {

void pto_launch_triv_inv_col_sweep_fp16(uint32_t blockDim, void* stream,
                                        void* x, void* z, uint32_t total_tiles,
                                        uint32_t matrix_size, uint32_t is_lower,
                                        uint32_t num_bsnd_heads,
                                        void* cu_seqlens);

void pto_launch_triv_inv_col_sweep_fp32(uint32_t blockDim, void* stream,
                                        void* x, void* z, uint32_t total_tiles,
                                        uint32_t matrix_size, uint32_t is_lower,
                                        uint32_t num_bsnd_heads,
                                        void* cu_seqlens);

}  // extern "C"

namespace pto_isa_ops {

/**
 * @brief Runs triangular inverse using vector-only column sweep method.
 *
 * @param [in] x Input tensor whose last two dimensions contain square
 * unit triangular matrices to invert. In BSND format (see is_bsnd_format)
 * the dimensions are instead B batch size, S sequence length (chunked in
 * tiles of size D), N number of heads and D chunk size, and the matrices to
 * invert live on the chunked S and the D dimensions.
 * @param [in] is_lower If input matrices are lower-triangular (true)
 * or upper-triangular (false). Default is lower triangular.
 * @param [in] is_bsnd_format If true, the tiles are strided inside a BSND
 * tensor and are loaded / stored with strided accesses; if false, each tile
 * is stored contiguously and row-wise in memory.
 * @param [in] cu_seqlens A 1-dimensional tensor with the cumulative sums of
 * the input sequence lengths, for variable-length BSND input. The last chunk
 * of a sequence is then shorter than the chunk size, and only that leading
 * square block of it is inverted; the rest of its valid rows is set to zero.
 * @return at::Tensor Tensor of same shape as input where matrices are inverted.
 */

at::Tensor run_tri_inv_col_sweep(
    const at::Tensor& x, const bool is_lower = true,
    const bool is_bsnd_format = false,
    const at::Tensor& cu_seqlens = at::zeros({1})) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  TORCH_CHECK(device.type() == DEVICE_TYPE,
              "tri_inv: tensor must be on NPU, got ", device);
  TORCH_CHECK(x.dim() >= 2,
              "tri_inv: input tensor must have at least 2 dimensions, got ",
              x.dim());

  const uint32_t matrix_size = static_cast<uint32_t>(x.size(-1));
  TORCH_CHECK(
      is_bsnd_format || matrix_size == static_cast<uint32_t>(x.size(-2)),
      "tri_inv: only square matrices are supported");

  // BSND: the tile rows come from the (chunked) sequence dimension, so the
  // tile count is the number of chunks times the number of heads rather than
  // numel / matrix_size^2, which over-counts once chunks are partly filled.
  const uint32_t num_bsnd_heads =
      is_bsnd_format ? static_cast<uint32_t>(x.size(-2)) : 0;
  const uint32_t num_elems = static_cast<uint32_t>(x.numel());
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

  uint32_t block_dim = GetNumVectorCores();
  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  // Every element of the output is written by the kernel: tail chunks of a
  // varlen BSND tensor store their valid rows at full width, zero padded.
  const at::Tensor z = at::empty_like(x);

  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "tri_inv: dtype must be fp16 or float32, got ", dtype);

  void* cu_seqlens_ptr = nullptr;
  if (is_bsnd_format && (cu_seqlens.numel() != 1)) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(triv_inv_col_sweep_fp16, block_dim, x, z, total_tiles,
                    matrix_size, is_lower, num_bsnd_heads, cu_seqlens_ptr);
  } else {
    EXEC_KERNEL_CMD(triv_inv_col_sweep_fp32, block_dim, x, z, total_tiles,
                    matrix_size, is_lower, num_bsnd_heads, cu_seqlens_ptr);
  }

  return z;
}
}  // namespace pto_isa_ops
