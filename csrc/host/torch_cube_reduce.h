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
                                 uint32_t num_tiles, uint32_t tail_rows,
                                 uint32_t blocks, uint64_t tail_offset,
                                 uint32_t tail_len, uint32_t size);
void pto_launch_cube_reduce_fp32(uint32_t blockDim, void* stream, void* input,
                                 void* partials, void* output,
                                 uint32_t num_tiles, uint32_t tail_rows,
                                 uint32_t blocks, uint64_t tail_offset,
                                 uint32_t tail_len, uint32_t size);
void pto_launch_cube_reduce_int8(uint32_t blockDim, void* stream, void* input,
                                 void* partials, void* output,
                                 uint32_t num_tiles, uint32_t tail_rows,
                                 uint32_t blocks, uint64_t tail_offset,
                                 uint32_t tail_len, uint32_t size);
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
  // An empty input has no kernel launch to produce its sum, so that one case
  // is zeroed on the host. Everywhere else the Vector stage writes this
  // element unconditionally, so it does not need initializing -- and zeroing
  // it would cost an extra kernel launch, which is a third of the floor on
  // small inputs.
  if (x.numel() == 0) return at::zeros({1}, output_options);
  const at::Tensor output = at::empty({1}, output_options);

  // The tile side is one of 32, 64 and 128, so all of the tiling arithmetic
  // is shifts and masks. Carrying the exponent instead of the side keeps three
  // 64-bit divisions off a host path that is itself a large part of the
  // latency at small lengths.
  const uint32_t size_log2 =
      x.numel() >= 128 * 128 ? 7 : (x.numel() >= 64 * 64 ? 6 : 5);
  const int64_t size = int64_t{1} << size_log2;
  const uint32_t num_tiles =
      static_cast<uint32_t>(x.numel() >> (2 * size_log2));
  const uint64_t tail_offset = static_cast<uint64_t>(num_tiles)
                               << (2 * size_log2);
  const uint32_t tail_len =
      static_cast<uint32_t>(x.numel() - static_cast<int64_t>(tail_offset));
  const uint32_t matrix_size = static_cast<uint32_t>(size);
  // The tail's whole rows are `size` elements each, which the Cube stage can
  // sum with the same GEMV once the rest of the tile is zeroed. Give them to
  // it as one extra work item and leave the Vector stage only the final
  // partial row -- under `size` elements, so always a single fold chunk. The
  // Cube stage is skipped entirely below one tile, and splitting the tail
  // would not pay for a launch there, so that case keeps its whole tail.
  const uint32_t tail_rows = num_tiles > 0 ? tail_len >> size_log2 : 0;
  const uint32_t cube_work = num_tiles + (tail_rows != 0 ? 1u : 0u);
  const uint64_t vec_tail_offset =
      tail_offset + (static_cast<uint64_t>(tail_rows) << size_log2);
  const uint32_t vec_tail_len = tail_len - (tail_rows << size_log2);
  // `blocks` is what the Vector stage folds together, so it must be zero when
  // the Cube stage was given no work at all and wrote no partials.
  const uint32_t blocks = std::min(cube_work, std::max(1u, GetNumCubeCores()));
  const uint32_t block_dim = std::max(1u, blocks);

  // Materialize strides. The kernel is handed data_ptr() and takes its length
  // from the scalars above, so it never reads the shape -- and data_ptr()
  // already points past a view's storage offset. Flattening would only cost a
  // view (about 1.2us of the host path) to produce the same pointer.
  const at::Tensor input = x.contiguous();
  // The Cube stage leaves one row of column sums per block. With `blocks == 0`
  // nothing writes or reads them, so `output` stands in as a valid pointer.
  const at::Tensor partials =
      blocks == 0
          ? output
          : at::empty({static_cast<int64_t>(block_dim), size}, output_options);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(cube_reduce_fp16, block_dim, input, partials, output,
                    num_tiles, tail_rows, blocks, vec_tail_offset, vec_tail_len,
                    matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(cube_reduce_fp32, block_dim, input, partials, output,
                    num_tiles, tail_rows, blocks, vec_tail_offset, vec_tail_len,
                    matrix_size);
  } else {
    EXEC_KERNEL_CMD(cube_reduce_int8, block_dim, input, partials, output,
                    num_tiles, tail_rows, blocks, vec_tail_offset, vec_tail_len,
                    matrix_size);
  }
  return output;
}
}  // namespace pto_isa_ops
