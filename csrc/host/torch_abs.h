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

extern "C" void call_vabs_fp16(uint32_t blockDim, void* stream, void* x,
                               void* y, uint32_t num_elements);

extern "C" void call_vabs_fp32(uint32_t blockDim, void* stream, void* x,
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
  at::Tensor z = at::empty_like(x);
  // Define the number of blocks of vector core
  const uint32_t total_len = x.numel();
  // FIXME: tile length is fixed to 64 for now
  constexpr uint32_t TILE_LEN = 64;
  const uint32_t block_dim = total_len / TILE_LEN;

  if (total_len % TILE_LEN != 0) {
    throw std::runtime_error(
        "pto_abs supports only inputs with length that is multiple of 64.");
  }

  if (dtype == at::kHalf) {
    call_vabs_fp16(block_dim, nullptr, ConvertType(x), ConvertType(z),
                   total_len);

  } else if (dtype == at::kFloat) {
    call_vabs_fp32(block_dim, nullptr, ConvertType(x), ConvertType(z),
                   total_len);

  } else {
    throw std::runtime_error("Unsupported dtype for `pto_abs` kernel");
  }

  return z;
}
}  // namespace pto_isa_ops
