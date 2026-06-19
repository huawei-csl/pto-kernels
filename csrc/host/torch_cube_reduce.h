/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "../tiling/heuristics/heuristics_cube_reduce.h"
#include "../tiling/tiling_cube_reduce.h"
#include "aclrtlaunch_cube_reduce_fp16.h"
#include "aclrtlaunch_cube_reduce_int8.h"
#include "commons.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "workspace.h"

namespace tcuscan {

/**
 * @brief Returns the sum-reductions over each block of an input 1D vector.
 *
 * @param [in] x Input 1D vector.
 * @param [in] num_blocks Number of blocks
 * @return Returns a vector of length `num_blocks` where each entry contains the
 * i-th block reduction.
 */
at::Tensor run_cube_reduce(const at::Tensor& x, uint32_t num_blocks) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t vec_len = x.numel();

  const at::Tensor z = at::zeros(
      {num_blocks}, at::TensorOptions().dtype(dtype_out).device(device));

  const kernel_utils::CubeReduceTiling tiling =
      kernel_utils::tiling::heuristics::cube_reduce::CalculateTiling(
          vec_len, num_blocks);
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  if (dtype == torch::kInt8) {
    const uint32_t workspace_size =
        kernel_utils::get_workspace_size<int8_t>(tiling);
    const at::Tensor workspace_tensor = alloc_workspace(workspace_size, device);
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    ACLRT_LAUNCH_KERNEL(cube_reduce_int8)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);

    aclrtFree(tiling_device);
    aclrtSynchronizeStream(acl_stream);

  } else if (dtype == torch::kHalf) {
    const uint32_t workspace_size =
        kernel_utils::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor = alloc_workspace(workspace_size, device);
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    ACLRT_LAUNCH_KERNEL(cube_reduce_fp16)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
    aclrtFree(tiling_device);
    aclrtSynchronizeStream(acl_stream);
  }

  return z;
}

}  // namespace tcuscan
