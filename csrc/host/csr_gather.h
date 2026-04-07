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

#include "aclrtlaunch_vcsr_gather_fp16.h"
#include "aclrtlaunch_vcsr_gather_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Runs the CSR gather operation. 
 *  z = values[indices] * x
 *
 * @param [in] values Input tensor of dtype fp16 or fp32.
 * @param [in] indices Input tensor of dtype int32.
 * @param [in] x Input x tensor of dtype fp16 or fp32.
 * @return at::Tensor Tensor containing the gathered values.
 */

at::Tensor run_csr_gather(const at::Tensor& values, const at::Tensor& indices, const at::Tensor& x) {
  const auto dtype = x.options().dtype();
  at::Tensor z = at::empty_like(values);
  
  // FIXME: expand to support bigger sizes
  const uint32_t x_size = x.numel();
  if (x_size > 16384) {
    throw std::runtime_error("Input x size exceeds the maximum supported size of 16384 elements");
  }

  // Define the number of blocks of vector core
  const uint32_t indices_size = indices.numel();
  // FIXME: tile length is fixed to 256 for now
  constexpr uint32_t TILE_SIZE = 256;

  // Persistent kernel launch parameter
  uint32_t total_tiles = (indices_size + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t block_dim = GetNumVectorCores();

  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(vcsr_gather_fp16, block_dim, values, indices, x, z, x_size, indices_size);

  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(vcsr_gather_fp32, block_dim, values, indices, x, z, x_size, indices_size);

  } else {
    throw std::runtime_error("Unsupported dtype for `pto_csr_gather` kernel");
  }

  return z;
}
}  // namespace pto_isa_ops
