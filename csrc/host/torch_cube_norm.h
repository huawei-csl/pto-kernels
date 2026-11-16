/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <limits>

#include "utils.h"

extern "C" {
void pto_launch_cube_norm_fp16(uint32_t blockDim, void* stream, void* input,
                               void* partials, void* output, uint32_t num_tiles,
                               uint64_t tail_offset, uint32_t tail_len,
                               uint32_t size);
void pto_launch_cube_norm_fp32(uint32_t blockDim, void* stream, void* input,
                               void* partials, void* output, uint32_t num_tiles,
                               uint64_t tail_offset, uint32_t tail_len,
                               uint32_t size);
}

namespace pto_isa_ops {

/**
 * @brief L2 norm of all elements of an input tensor.
 *
 * The input is read as S x S tiles that the Cube unit squares into
 * F = sum_i A_i^T @ A_i, every block adding its F atomically into one zeroed
 * S x S workspace tile; trace(F) is the sum of the squares, which the Vector
 * unit finishes and takes the square root of. The elements past the last whole
 * tile are squared and summed on the Vector unit instead.
 *
 * @param [in] x Input tensor on the NPU with dtype fp16 or fp32.
 * @return at::Tensor Single-element fp32 tensor containing sqrt(sum(x * x)).
 */
at::Tensor run_cube_norm(const at::Tensor& x) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "cube_norm: tensor must be on NPU, got ", x.device());
  const auto dtype = x.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "cube_norm: only float16 and float32 inputs are supported, got ",
              dtype);
  TORCH_CHECK(x.numel() <= std::numeric_limits<uint32_t>::max(),
              "cube_norm: input length exceeds uint32 range");
  const c10::DeviceGuard device_guard(x.device());
  const auto output_options = x.options().dtype(at::kFloat);
  if (x.numel() == 0) return at::zeros({1}, output_options);
  const at::Tensor output = at::empty({1}, output_options);

  // The largest tile the input fills, so that short inputs still reach the
  // Cube unit instead of being left entirely to the Vector stage.
  const int64_t size =
      x.numel() >= 128 * 128 ? 128 : (x.numel() >= 64 * 64 ? 64 : 32);
  const int64_t tile_elems = size * size;
  const uint32_t num_tiles = static_cast<uint32_t>(x.numel() / tile_elems);
  const uint64_t tail_offset = static_cast<uint64_t>(num_tiles) * tile_elems;
  const uint32_t tail_len =
      static_cast<uint32_t>(x.numel() - static_cast<int64_t>(tail_offset));
  const uint32_t matrix_size = size;
  // Nothing for a block beyond the tile count to do, and no point launching
  // one that would only contend for the shared accumulator.
  const uint32_t block_dim =
      std::max(1u, std::min(num_tiles, std::max(1u, GetNumCubeCores())));

  // Materialize strides and preserve storage offsets.
  const at::Tensor input = x.contiguous().reshape({-1});
  // Every block adds its S x S result into this one tile atomically, so it
  // has to start at zero; an input short of a whole tile leaves it at zero
  // and the Vector stage then traces zeros.
  const at::Tensor partials = at::zeros({size, size}, output_options);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(cube_norm_fp16, block_dim, input, partials, output,
                    num_tiles, tail_offset, tail_len, matrix_size);
  } else {
    EXEC_KERNEL_CMD(cube_norm_fp32, block_dim, input, partials, output,
                    num_tiles, tail_offset, tail_len, matrix_size);
  }
  return output;
}
}  // namespace pto_isa_ops
