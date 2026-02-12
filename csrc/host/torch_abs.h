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

at::Tensor run_abs(const at::Tensor& x) {
  const auto dtype = x.options().dtype();
  at::Tensor z = at::empty_like(x);
  // Define the number of blocks of vector core
  const uint32_t total_len = x.numel();
  // FIXME: fixed for now
  const uint32_t tile_len = 64 * 64;
  const uint32_t block_dim = total_len / tile_len;

  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(vabs_fp16, block_dim, x, z, total_len);

  } else if (dtype == at::kFloat) {
    EXEC_KERNEL_CMD(vabs_fp32, block_dim, x, z, total_len);

  } else {
    throw std::runtime_error("Unsupported dtype for `pto_abs` kernel");
  }

  return z;
}
}  // namespace pto_isa_ops
