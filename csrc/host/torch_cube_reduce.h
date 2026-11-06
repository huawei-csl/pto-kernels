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
void pto_launch_cube_reduce_fp16(uint32_t blockDim, void* stream, void* input,
                                 void* ones, void* partials, void* output,
                                 uint32_t num_tiles, uint32_t size);
void pto_launch_cube_reduce_fp32(uint32_t blockDim, void* stream, void* input,
                                 void* ones, void* partials, void* output,
                                 uint32_t num_tiles, uint32_t size);
void pto_launch_cube_reduce_int8(uint32_t blockDim, void* stream, void* input,
                                 void* ones, void* partials, void* output,
                                 uint32_t num_tiles, uint32_t size);
}

namespace pto_isa_ops {

/**
 * @brief Sums all elements of an input tensor into one value.
 *
 * @param [in] x Input tensor on the NPU with dtype fp16 or int8.
 * @return at::Tensor Single-element tensor containing the sum as fp32 for fp16
 * input or int32 for int8 input.
 */
at::Tensor run_cube_reduce(const at::Tensor& x) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "cube_reduce: tensor must be on NPU, got ", x.device());
  const auto dtype = x.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kFloat || dtype == at::kChar,
      "cube_reduce: only float16, float32 and int8 inputs are supported, "
      "got ",
      dtype);
  TORCH_CHECK(x.numel() <= std::numeric_limits<uint32_t>::max(),
              "cube_reduce: input length exceeds uint32 range");
  const c10::DeviceGuard device_guard(x.device());
  const auto output_options =
      x.options().dtype(dtype == at::kChar ? at::kInt : at::kFloat);
  const at::Tensor output = at::zeros({1}, output_options);
  if (x.numel() == 0) return output;

  const int64_t size =
      x.numel() >= 128 * 128 ? 128 : (x.numel() >= 64 * 64 ? 64 : 32);
  const int64_t tile_elems = size * size;
  const uint32_t num_tiles = (x.numel() + tile_elems - 1) / tile_elems;
  const uint32_t matrix_size = size;
  const uint32_t block_dim =
      std::min(num_tiles, std::max(1u, GetNumCubeCores()));

  // Materialize strides, preserve storage offsets, and pad the tail with zeros.
  at::Tensor input = x.contiguous().reshape({-1});
  const int64_t padded_len = static_cast<int64_t>(num_tiles) * tile_elems;
  if (input.numel() != padded_len) {
    at::Tensor padded = at::zeros({padded_len}, x.options());
    padded.narrow(0, 0, input.numel()).copy_(input);
    input = padded;
  }
  const at::Tensor ones = at::ones({size, 16}, x.options());
  const at::Tensor partials = at::empty({block_dim, size, 16}, output_options);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(cube_reduce_fp16, block_dim, input, ones, partials, output,
                    num_tiles, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(cube_reduce_fp32, block_dim, input, ones, partials, output,
                    num_tiles, matrix_size);
  } else {
    EXEC_KERNEL_CMD(cube_reduce_int8, block_dim, input, ones, partials, output,
                    num_tiles, matrix_size);
  }
  return output;
}
}  // namespace pto_isa_ops
