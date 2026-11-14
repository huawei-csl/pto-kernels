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

#include <algorithm>
#include <limits>
#include <utility>

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "utils.h"

extern "C" {

void pto_launch_csr_gather_fp16(uint32_t blockDim, void* stream, void* values,
                                void* indices, void* x, void* z,
                                uint32_t x_size, uint32_t indices_size);

void pto_launch_csr_gather_fp32(uint32_t blockDim, void* stream, void* values,
                                void* indices, void* x, void* z,
                                uint32_t x_size, uint32_t indices_size);

}  // extern "C"

namespace pto_isa_ops {

/**
 * @brief Runs the CSR gather operation.
 *  z = values * x[indices]
 *
 * @param [in] values Contiguous base-format fp16 or fp32 entries.
 * @param [in] indices Contiguous base-format int32 indices, one per value.
 *                     Indices must be in [0, x.numel()); device contents are
 *                     not synchronized to the host for validation.
 * @param [in] x Contiguous base-format tensor with the same dtype and device
 *               as values. All inputs are read in flattened order.
 * @return at::Tensor Tensor containing the gathered values.
 */

at::Tensor run_csr_gather(const at::Tensor& values, const at::Tensor& indices,
                          const at::Tensor& x) {
  const auto device = x.device();
  const auto dtype = x.scalar_type();
  TORCH_CHECK(device.type() == DEVICE_TYPE,
              "csr_gather: x must be on NPU, got ", device);
  TORCH_CHECK(values.device() == device && indices.device() == device,
              "csr_gather: values, indices and x must be on the same device");
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "csr_gather: x dtype must be fp16 or float32, got ", dtype);
  TORCH_CHECK(values.scalar_type() == dtype,
              "csr_gather: values and x must have the same dtype");
  TORCH_CHECK(indices.scalar_type() == at::kInt,
              "csr_gather: indices must have dtype int32");
  TORCH_CHECK(
      values.numel() == indices.numel(),
      "csr_gather: values and indices must have the same number of elements");
  for (const auto& [name, tensor] :
       {std::pair{"values", &values}, std::pair{"indices", &indices},
        std::pair{"x", &x}}) {
    TORCH_CHECK(tensor->is_contiguous(), "csr_gather: ", name,
                " must be contiguous");
    TORCH_CHECK(!tensor->is_neg(), "csr_gather: resolve negative views of ",
                name, " with resolve_neg() before calling");
    const auto format = at_npu::native::get_npu_format(*tensor);
    TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
                    format == ACL_FORMAT_NHWC || format == ACL_FORMAT_NCDHW,
                "csr_gather: ", name,
                " must use a base NPU format; convert "
                "with torch_npu.npu_format_cast(tensor, 2)");
  }

  TORCH_CHECK(x.numel() <= 40960,
              "csr_gather: input x size exceeds the maximum supported size of "
              "40960 elements, got ",
              x.numel());
  const c10::DeviceGuard device_guard(device);
  if (indices.numel() == 0) {
    return at::empty_like(values);
  }
  TORCH_CHECK(x.numel() > 0,
              "csr_gather: x must be nonempty when indices are nonempty");
  TORCH_CHECK(indices.numel() <= std::numeric_limits<uint32_t>::max(),
              "csr_gather: entry count exceeds the kernel's 32-bit tile range");
  const uint32_t x_size = static_cast<uint32_t>(x.numel());
  const uint32_t indices_size = static_cast<uint32_t>(indices.numel());
  constexpr uint32_t TILE_SIZE = 512;
  const uint64_t total_tiles =
      (uint64_t{indices_size} + TILE_SIZE - 1) / TILE_SIZE;
  const uint32_t block_dim =
      std::min<uint64_t>(total_tiles, GetNumVectorCores());
  TORCH_CHECK(block_dim > 0, "csr_gather: device has no vector cores");
  // The kernel computes each core's padded end offset in uint32_t.
  const uint64_t padded_end =
      ((total_tiles + block_dim - 1) / block_dim) * block_dim * TILE_SIZE;
  TORCH_CHECK(padded_end <= std::numeric_limits<uint32_t>::max(),
              "csr_gather: entry count exceeds the kernel's 32-bit tile range");

  at::Tensor z = at::empty_like(values);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(csr_gather_fp16, block_dim, values, indices, x, z, x_size,
                    indices_size);
  } else {
    EXEC_KERNEL_CMD(csr_gather_fp32, block_dim, values, indices, x, z, x_size,
                    indices_size);
  }

  return z;
}
}  // namespace pto_isa_ops
