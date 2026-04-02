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

#include "aclrtlaunch_vabs_fp16.h"
#include "aclrtlaunch_vabs_fp32.h"
#include "utils.h"

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
  const uint32_t total_size = x.numel();
  // FIXME: tile length is fixed to 64 for now
  constexpr uint32_t TILE_SIZE = 64;

  // Persistent kernel launch parameter
  uint32_t total_tiles = (total_size + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t block_dim = GetNumVectorCores();

  std::cout << "Num vec cores: " << block_dim << std::endl;
  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  block_dim = 2; // debug

  std::cout << "Launching vabs kernel with block_dim: " << block_dim
            << ", total_tiles: " << total_tiles << std::endl;

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(vabs_fp16, block_dim, x, z, total_size);

  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(vabs_fp32, block_dim, x, z, total_size);

  } else {
    throw std::runtime_error("Unsupported dtype for `pto_abs` kernel");
  }

  return z;
}
}  // namespace pto_isa_ops
