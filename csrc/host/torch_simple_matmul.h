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

void pto_launch_simple_matmul_bf16(uint32_t blockDim, void* stream, void* a,
                                   void* b, void* c, uint32_t matrix_size);

void pto_launch_simple_matmul_fp16(uint32_t blockDim, void* stream, void* a,
                                   void* b, void* c, uint32_t matrix_size);

void pto_launch_simple_matmul_fp32(uint32_t blockDim, void* stream, void* a,
                                   void* b, void* c, uint32_t matrix_size);

}  // extern "C"

namespace pto_isa_ops {

/**
 * @brief Simple matrix multiplication A @ B.
 *
 * @param [in] a Contiguous base-format matrix [M, M], where M is 0, 16,
 *               32, 64, 96, or 128, with fp16, bf16, or float32 elements.
 * @param [in] b Contiguous base-format matrix with the same shape, dtype,
 *               and NPU device as a.
 * @return at::Tensor Float32 matrix multiplication result of a @ b.
 */
at::Tensor run_simple_matmul(const at::Tensor& a, const at::Tensor& b) {
  const at::Device device = a.options().device();
  const auto dtype = a.options().dtype();
  const auto dtype_out = at::kFloat;

  TORCH_CHECK(device.type() == DEVICE_TYPE,
              "simple_matmul: tensor must be on NPU, got ", device);
  TORCH_CHECK(b.device() == device,
              "simple_matmul: a and b must be on the same NPU device, got ",
              device, " and ", b.device());
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kFloat || dtype == at::kBFloat16,
      "simple_matmul: dtype must be fp16, bf16, or float32, got ", dtype);

  TORCH_CHECK(b.scalar_type() == a.scalar_type(),
              "simple_matmul: a and b must have the same dtype");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
              "simple_matmul: a and b must be two-dimensional matrices");
  const int64_t matrix_size = a.size(0);
  TORCH_CHECK(
      a.size(1) == matrix_size && b.sizes() == a.sizes(),
      "simple_matmul: a and b must be square matrices of the same size");
  TORCH_CHECK(matrix_size == 0 || matrix_size == 16 || matrix_size == 32 ||
                  matrix_size == 64 || matrix_size == 96 || matrix_size == 128,
              "simple_matmul: matrix size must be 0, 16, 32, 64, 96, or 128");
  for (const auto& input : {a, b}) {
    TORCH_CHECK(input.is_contiguous(),
                "simple_matmul: a and b must be contiguous");
    const auto format = at_npu::native::get_npu_format(input);
    TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
                    format == ACL_FORMAT_NHWC || format == ACL_FORMAT_NCDHW,
                "simple_matmul: a and b must use base NPU formats; convert "
                "with torch_npu.npu_format_cast(input, 2)");
  }
  // Negating both operands leaves the product of their stored values unchanged.
  TORCH_CHECK(
      matrix_size == 0 || a.is_neg() == b.is_neg(),
      "simple_matmul: resolve a single negative view with resolve_neg() "
      "before multiplying");

  constexpr uint32_t block_dim = 1;

  const c10::DeviceGuard device_guard(device);
  const at::Tensor c =
      at::ones({matrix_size, matrix_size},
               at::TensorOptions().dtype(dtype_out).device(device));
  if (matrix_size == 0) {
    return c;
  }

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(simple_matmul_fp16, block_dim, a, b, c, matrix_size);
  } else if (dtype == at::kBFloat16) {
    EXEC_KERNEL_CMD(simple_matmul_bf16, block_dim, a, b, c, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(simple_matmul_fp32, block_dim, a, b, c, matrix_size);
  }

  return c;
}
}  // namespace pto_isa_ops
