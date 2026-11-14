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

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "utils.h"

extern "C" {

void pto_launch_scan_ul1_fp16(uint32_t blockDim, void* stream, void* x, void* o,
                              void* u, void* l, void* s, uint32_t matrix_size);

void pto_launch_scan_ul1_fp32(uint32_t blockDim, void* stream, void* x, void* o,
                              void* u, void* l, void* s, uint32_t matrix_size);

}  // extern "C"

namespace pto_isa_ops {

/**
 * @brief Single Cube scan
 *
 * @param [in] x Contiguous base-format input vector with resolved negative
 *               bit and 256, 1024, 4096, 9216, or 16384 elements.
 * @return at::Tensor vector result of the scan operation.
 */
at::Tensor run_scan_ul1(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = at::kFloat;

  TORCH_CHECK(device.type() == DEVICE_TYPE,
              "scan_ul1: tensor must be on NPU, got ", device);
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "scan_ul1: dtype must be fp16 or float32, got ", dtype);

  TORCH_CHECK(x.dim() == 1, "scan_ul1: only 1D tensors are supported, got ",
              x.dim(), "D");
  TORCH_CHECK(x.is_contiguous(), "scan_ul1: input must be contiguous");
  TORCH_CHECK(
      !x.is_neg(),
      "scan_ul1: resolve negative views with resolve_neg() before calling");

  const auto format = at_npu::native::get_npu_format(x);
  TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
                  format == ACL_FORMAT_NHWC || format == ACL_FORMAT_NCDHW,
              "scan_ul1: input must use a base NPU format; convert with "
              "torch_npu.npu_format_cast(x, 2)");

  const int64_t scan_size = x.numel();
  const uint32_t matrix_size = sqrt(scan_size);
  TORCH_CHECK(
      (matrix_size == 16 || matrix_size == 32 || matrix_size == 64 ||
       matrix_size == 96 || matrix_size == 128) &&
          matrix_size * matrix_size == scan_size,
      "scan_ul1: input size must be 256, 1024, 4096, 9216, or 16384, got ",
      scan_size);

  constexpr uint32_t block_dim = 1;

  const at::Tensor scan = at::zeros(
      {scan_size}, at::TensorOptions().dtype(dtype_out).device(device));

  // FIXME: use vector or scalar cores to generate O, U and L directly on the
  // device

  // Ones matrix
  const at::Tensor o =
      torch::ones({matrix_size, matrix_size},
                  at::TensorOptions().dtype(dtype).device(device));
  // Upper triangular matrix
  const at::Tensor u = torch::triu(o);
  // Lower triangular matrix
  const at::Tensor l = torch::tril(o, -1);

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(scan_ul1_fp16, block_dim, x, o, u, l, scan, matrix_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(scan_ul1_fp32, block_dim, x, o, u, l, scan, matrix_size);
  }

  return scan;
}
}  // namespace pto_isa_ops
