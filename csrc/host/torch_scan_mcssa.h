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
#include <runtime/rt.h>

#include "aclrtlaunch_scan_mcssa_fp16.h"
#include "aclrtlaunch_scan_mcssa_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Single Cube scan
 *
 * @param [in] x Input vector
 * @return at::Tensor vector result of the scan operation.
 */
at::Tensor run_scan_mcssa(const at::Tensor& x) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out = at::kFloat;

  if (!(dtype == at::kHalf or dtype == at::kFloat)) {
    throw std::runtime_error(
        "Unsupported dtype for scan_mcssa kernel. Supports only fp16/fp32");
  }

  const uint32_t scan_size = static_cast<uint32_t>(x.size(-1));
  if (x.dim() != 1) {
    throw std::runtime_error("Only 1D scan is supported.\n");
  }

  const at::Tensor scan = at::zeros(
      {scan_size}, at::TensorOptions().dtype(dtype_out).device(device));


  // FIXME: pad to support other sizes
  constexpr uint32_t tile_size = 16;
  uint32_t number_of_tiles = (scan_size + tile_size*tile_size - 1) / tile_size*tile_size;
  const uint32_t block_dim = number_of_tiles;
  
  // FIXME: use vector or scalar cores to generate O, U and L directly on the
  // device

  // Ones matrix
  const at::Tensor o =
      torch::ones({tile_size, tile_size},
                  at::TensorOptions().dtype(dtype).device(device));
  // Upper triangular matrix
  const at::Tensor u = torch::triu(o);
  // Lower triangular matrix
  const at::Tensor l = torch::tril(o, -1);


  // void *ffts_addr;
  // uint32_t ffts_len;
  // rtGetC2cCtrlAddr((uint64_t *)&ffts_addr, &ffts_len);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(scan_mcssa_fp16, block_dim, x, o, u, l, scan, scan_size, tile_size);
  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(scan_mcssa_fp32, block_dim, x, o, u, l, scan, scan_size, tile_size);
  }

  return scan;
}
}  // namespace pto_isa_ops
