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
                                 void* partials, void* output,
                                 uint32_t num_tiles, uint32_t blocks,
                                 uint64_t tail_offset, uint32_t tail_len,
                                 uint32_t size);
void pto_launch_cube_reduce_fp32(uint32_t blockDim, void* stream, void* input,
                                 void* partials, void* output,
                                 uint32_t num_tiles, uint32_t blocks,
                                 uint64_t tail_offset, uint32_t tail_len,
                                 uint32_t size);
void pto_launch_cube_reduce_int8(uint32_t blockDim, void* stream, void* input,
                                 void* partials, void* output,
                                 uint32_t num_tiles, uint32_t blocks,
                                 uint64_t tail_offset, uint32_t tail_len,
                                 uint32_t size);
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
  const uint32_t num_tiles = static_cast<uint32_t>(x.numel() / tile_elems);
  const uint64_t tail_offset = static_cast<uint64_t>(num_tiles) * tile_elems;
  const uint32_t tail_len =
      static_cast<uint32_t>(x.numel() - static_cast<int64_t>(tail_offset));
  const uint32_t matrix_size = size;
  // `blocks` is what the Vector stage folds together, so it must be zero when
  // the Cube stage was not given a single whole tile and wrote no partials.
  const uint32_t blocks = std::min(num_tiles, std::max(1u, GetNumCubeCores()));
  const uint32_t block_dim = std::max(1u, blocks);

  // Materialize strides and preserve storage offsets.
  const at::Tensor input = x.contiguous().reshape({-1});
  const at::Tensor partials =
      at::empty({static_cast<int64_t>(block_dim), size, 16}, output_options);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(cube_reduce_fp16, block_dim, input, partials, output,
                    num_tiles, blocks, tail_offset, tail_len, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(cube_reduce_fp32, block_dim, input, partials, output,
                    num_tiles, blocks, tail_offset, tail_len, matrix_size);
  } else {
    EXEC_KERNEL_CMD(cube_reduce_int8, block_dim, input, partials, output,
                    num_tiles, blocks, tail_offset, tail_len, matrix_size);
  }
  return output;
}
}  // namespace pto_isa_ops
