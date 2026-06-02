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

#include "aclrtlaunch_cube_reduce_fp16.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Pick the largest S in {128, 64, 32, 16} such that S*S divides
 * vec_len.  This determines the matmul tile size used by the Cube kernel.
 */
inline uint32_t ChooseCubeReduceMatmulSize(uint32_t vec_len) {
  for (uint32_t s : {128u, 64u, 32u, 16u}) {
    if (vec_len % (s * s) == 0) return s;
  }
  return 16;
}

/**
 * @brief Block-wise sum reduction of a 1D tensor using Cube matmul.
 *
 * Splits `x` into `num_blocks` equal partitions and returns a float32 tensor
 * of shape [num_blocks] where entry i contains the sum of elements in
 * partition i.
 *
 * The kernel uses a two-phase algorithm:
 *   1. AIC (Cube) cores multiply each S×S input tile by an S×16 all-ones
 *      matrix, accumulating partial sums into an S×16 intermediate buffer.
 *   2. AIV (Vector) cores sum column 0 of each intermediate buffer to produce
 *      the final scalar per block.
 *
 * @param [in] x         Input 1D tensor.  dtype must be fp16.
 * @param [in] num_blocks Number of output reductions.
 * @return Tensor of shape [num_blocks] with dtype float32.
 */
at::Tensor run_cube_reduce(const at::Tensor& x, uint32_t num_blocks) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "pto_cube_reduce: tensor must be on NPU, got ", x.device());
  TORCH_CHECK(x.scalar_type() == at::kHalf,
              "pto_cube_reduce: only fp16 input is supported, got ",
              x.scalar_type());
  TORCH_CHECK(x.is_contiguous(), "pto_cube_reduce: input must be contiguous");
  TORCH_CHECK(num_blocks > 0,
              "pto_cube_reduce: num_blocks must be positive, got ", num_blocks);

  const uint32_t vec_len = static_cast<uint32_t>(x.numel());
  const uint32_t matmul_size = ChooseCubeReduceMatmulSize(vec_len);

  TORCH_CHECK(
      vec_len % (matmul_size * matmul_size) == 0, "pto_cube_reduce: vec_len (",
      vec_len, ") must be divisible by matmul_size^2 (",
      matmul_size * matmul_size,
      "). Ensure the input length is aligned to a supported tile size.");

  const at::Device device = x.options().device();

  // All-ones B matrix fed to the Cube unit: shape [matmul_size, 16], fp16.
  const at::Tensor all_ones_b =
      at::ones({static_cast<int64_t>(matmul_size), 16},
               at::TensorOptions().dtype(at::kHalf).device(device));

  // Intermediate workspace: block_num × matmul_size × 16 float32 values.
  const at::Tensor workspace =
      at::empty({static_cast<int64_t>(num_blocks * matmul_size * 16)},
                at::TensorOptions().dtype(at::kFloat).device(device));

  // Output: one float32 scalar per block.
  const at::Tensor z =
      at::empty({static_cast<int64_t>(num_blocks)},
                at::TensorOptions().dtype(at::kFloat).device(device));

  // Launch the combined AIC+AIV kernel; num_blocks gives the AIC block count.
  // The AIV phase runs in the same kernel via #elif __DAV_VEC__ guards.
  EXEC_KERNEL_CMD(cube_reduce_fp16, num_blocks, x, all_ones_b, workspace, z,
                  vec_len, num_blocks, matmul_size);

  return z;
}

}  // namespace pto_isa_ops
