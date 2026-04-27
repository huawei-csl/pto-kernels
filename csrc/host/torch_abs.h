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
  // FIXME: tile length is fixed to 128 for now
  constexpr uint32_t TILE_SIZE = 128;

  // Persistent kernel launch parameter
  uint32_t total_tiles = (total_size + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t block_dim = GetNumVectorCores();

  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "pto_abs: dtype must be fp16 or float32, got ", dtype);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(vabs_fp16, block_dim, x, z, total_size);
  } else {
    EXEC_KERNEL_CMD(vabs_fp32, block_dim, x, z, total_size);
  }

  return z;
}
}  // namespace pto_isa_ops
