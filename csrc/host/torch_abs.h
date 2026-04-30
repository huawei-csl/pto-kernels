/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch/library.h>

#include "utils.h"

extern "C" void call_vabs_fp16(uint32_t blockDim, aclrtStream stream, void* x,
                               void* y, uint32_t num_elements);

extern "C" void call_vabs_fp32(uint32_t blockDim, aclrtStream stream, void* x,
                               void* y, uint32_t num_elements);

namespace pto_isa_ops {
/**
 * @brief Runs element-wise absolute value.
 *
 * @param [in] x Input tensor of dtype fp16 or fp32.
 * @return at::Tensor Tensor containing the absolute value on each entry.
 */

at::Tensor run_abs(const at::Tensor& x) {
  const auto dtype = x.options().dtype();
  const at::Tensor z = at::empty_like(x);
  // Define the number of blocks of vector core
  const uint32_t total_size = x.numel();
  // FIXME: tile length is fixed to 128 for now
  constexpr uint32_t TILE_SIZE = 128;

  // Persistent kernel launch parameter
  uint32_t total_tiles = (total_size + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t block_dim = GetNumCubeCores();

  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
  if (dtype == at::kHalf) {
    call_vabs_fp16(block_dim, acl_stream, ConvertType(x), ConvertType(z),
                   total_size);
  } else if (dtype == at::kFloat) {
    call_vabs_fp32(block_dim, acl_stream, ConvertType(x), ConvertType(z),
                   total_size);
  } else {
    throw std::runtime_error("Unsupported dtype for `pto_abs` kernel");
  }

  return z;
}
}  // namespace pto_isa_ops
