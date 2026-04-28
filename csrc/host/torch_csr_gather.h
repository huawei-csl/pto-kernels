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

#include "aclrtlaunch_csr_gather_fp16.h"
#include "aclrtlaunch_csr_gather_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Runs the CSR gather operation.
 *  z = values * x[indices]
 *
 * @param [in] values Input tensor of dtype fp16 or fp32.
 * @param [in] indices Input tensor of dtype int32.
 * @param [in] x Input x tensor of dtype fp16 or fp32.
 * @return at::Tensor Tensor containing the gathered values.
 */

at::Tensor run_csr_gather(const at::Tensor& values, const at::Tensor& indices,
                          const at::Tensor& x) {
  const auto dtype = x.options().dtype();
  at::Tensor z = at::empty_like(values);

  // FIXME: expand to support bigger sizes
  const uint32_t x_size = x.numel();
  TORCH_CHECK(x_size <= 40960,
              "csr_gather: input x size exceeds the maximum supported size of "
              "40960 elements, got ",
              x_size);

  // Define the number of blocks of vector core
  const uint32_t indices_size = indices.numel();
  // FIXME: tile length is fixed to 512 for now
  constexpr uint32_t TILE_SIZE = 512;

  // Persistent kernel launch parameter
  uint32_t total_tiles = (indices_size + TILE_SIZE - 1) / TILE_SIZE;
  uint32_t block_dim = GetNumVectorCores();

  if (total_tiles < block_dim) {
    block_dim = total_tiles;
  }

  TORCH_CHECK(values.device().type() == DEVICE_TYPE,
              "csr_gather: tensor must be on NPU, got ", values.device());
  TORCH_CHECK(indices.device().type() == DEVICE_TYPE,
              "csr_gather: tensor must be on NPU, got ", indices.device());
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "csr_gather: tensor must be on NPU, got ", x.device());
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
              "csr_gather: dtype must be fp16 or float32, got ", dtype);
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
