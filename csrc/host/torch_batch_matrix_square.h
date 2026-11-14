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
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "utils.h"

extern "C" {

void pto_launch_batch_matrix_square_fp16(uint32_t blockDim, void* stream,
                                         void* z, void* x,
                                         uint32_t matrix_size);

void pto_launch_batch_matrix_square_fp32(uint32_t blockDim, void* stream,
                                         void* z, void* x,
                                         uint32_t matrix_size);

}  // extern "C"

namespace pto_isa_ops {

/**
 * @brief Batch matrix square given a tensor whose last two dimensions form a
 * square matrix.
 *
 * @param [in] x Contiguous base-format tensor [..., M, M], where M is 0, 16,
 *               32, 64, 96, or 128. Leading dimensions enumerate matrices.
 *               At most 65535 nonempty matrices can be launched in one call.
 * @return at::Tensor Float32 tensor of the same shape as `x` containing matrix
 * squares.
 */
at::Tensor run_batch_matrix_square(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = at::kFloat;

  TORCH_CHECK(device.type() == DEVICE_TYPE,
              "batch_matrix_square: tensor must be on NPU, got ", device);
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "batch_matrix_square: dtype must be fp16 or float32, got ",
              dtype);

  TORCH_CHECK(x.dim() >= 2,
              "batch_matrix_square: input must have at least two dimensions");
  const int64_t matrix_size = x.size(-1);
  TORCH_CHECK(matrix_size == x.size(-2),
              "batch_matrix_square: only square matrices are supported");
  TORCH_CHECK(
      matrix_size == 0 || matrix_size == 16 || matrix_size == 32 ||
          matrix_size == 64 || matrix_size == 96 || matrix_size == 128,
      "batch_matrix_square: matrix size must be 0, 16, 32, 64, 96, or 128");
  TORCH_CHECK(x.is_contiguous(),
              "batch_matrix_square: input must be contiguous");
  const auto format = at_npu::native::get_npu_format(x);
  TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
                  format == ACL_FORMAT_NHWC || format == ACL_FORMAT_NCDHW,
              "batch_matrix_square: input must use a base NPU format; convert "
              "with torch_npu.npu_format_cast(x, 2)");

  const int64_t num_matrices =
      matrix_size == 0 ? 0 : x.numel() / (matrix_size * matrix_size);
  // Ascend C limits a launch grid to 65535 logical blocks.
  constexpr int64_t kMaxLaunchBlocks = 65535;
  TORCH_CHECK(num_matrices <= kMaxLaunchBlocks, "batch_matrix_square: at most ",
              kMaxLaunchBlocks, " matrices are supported per launch, got ",
              num_matrices);

  const c10::DeviceGuard device_guard(device);
  const at::Tensor z =
      at::zeros(x.sizes(), at::TensorOptions().dtype(dtype_out).device(device));
  if (num_matrices == 0) {
    return z;
  }
  const uint32_t block_dim = static_cast<uint32_t>(num_matrices);
  // A lazy negative bit cancels because both matrix operands are x.

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(batch_matrix_square_fp16, block_dim, z, x, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(batch_matrix_square_fp32, block_dim, z, x, matrix_size);
  }

  return z;
}
}  // namespace pto_isa_ops
