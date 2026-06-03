/**
 * @file torch_reduce.h
 * @brief Torch wrapper for reduce kernels.
 * @date 2025-05-13
 *
 * @copyright Copyright Huawei (c) 2025
 */
#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "aclrtlaunch_cube_reduce_fp16.h"
#include "aclrtlaunch_cube_reduce_int8.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace tcuscan {

/**
 * @brief Returns the sum-reductions over each tile of an input 1D vector.
 *
 * @param x Input 1D vector.
 * @param tile_len Length of tiles to reduce on.
 * @param num_blocks Number of blocks
 * @return Returns a vector of length `num_blocks` where each entry contains the
 * i-th block reduction.
 */
at::Tensor run_reduce_tiles(const at::Tensor& x, uint32_t tile_len,
                            uint32_t num_blocks) {
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kHalf ? torch::kFloat32 : torch::kInt32;

  const uint32_t total_len = x.numel();

  const at::Tensor z = at::empty(
      {num_blocks}, at::TensorOptions().dtype(dtype_out).device(device));

  const ReduceTilesTiling tiling{total_len, tile_len};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);
  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  if (dtype == torch::kInt8) {
    ACLRT_LAUNCH_KERNEL(reduce_tiles_int8)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kHalf) {
    ACLRT_LAUNCH_KERNEL(reduce_tiles_fp16)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Returns the sum-reductions over each tile of an input 1D vector.
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

  const at::Tensor z = at::empty(
      {num_blocks}, at::TensorOptions().dtype(dtype_out).device(device));

  const tcuscan::CubeReduceTiling tiling =
      tcuscan::tiling::heuristics::cube_reduce::CalculateTiling(vec_len,
                                                                num_blocks);
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kInt8) {
    const uint32_t workspace_size = tcuscan::get_workspace_size<int8_t>(tiling);
    const at::Tensor workspace_tensor = alloc_workspace(workspace_size, device);
    ACLRT_LAUNCH_KERNEL(cube_reduce_int8)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kHalf) {
    const uint32_t workspace_size =
        tcuscan::get_workspace_size<int16_t>(tiling);
    const at::Tensor workspace_tensor = alloc_workspace(workspace_size, device);
    ACLRT_LAUNCH_KERNEL(cube_reduce_fp16)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);
  aclrtSynchronizeStream(acl_stream);

  return z;
}

/**
 * @brief Broadcast-adds the block-sums into each block on the second phase
 * (down-sweep) of MCSCAN.
 *
 * Assumption: input vector x is split into `len(sums)` blocks.
 *
 * @param [in] x Input 1D vector.
 * @param [in] sums Tensor containing sum-reduction of each block of x.
 * len(sums) == number of blocks
 * @param [in] tile_width Length of the row used by `KernelRowScan` kernel.
 * @param [in] tile_height Number of rows processed in a single iteration
 * `KernelRowScan` kernel.
 * @return Returns a vector where each i-th block of x has been added sims[i].
 */
at::Tensor run_complete_rows(const at::Tensor& x, const at::Tensor& sums,
                             int tile_width, int tile_height) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();
  const auto dtype_out =
      dtype == torch::kFloat32 ? torch::kFloat32 : torch::kInt32;

  const uint32_t block_dim = sums.numel();
  const uint32_t total_len = x.numel();
  const uint32_t width = static_cast<uint32_t>(tile_width);
  const uint32_t height = static_cast<uint32_t>(tile_height);

  const at::Tensor z = at::empty(
      {total_len}, at::TensorOptions().dtype(dtype_out).device(device));

  const CompleteRowsTiling tiling{total_len, width, height};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kLong) {
    ACLRT_LAUNCH_KERNEL(complete_rows_int32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(sums.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kFloat) {
    ACLRT_LAUNCH_KERNEL(complete_rows_fp32)
    (block_dim, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(sums.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);

  return z;
}

/**
 * @brief Broadcast-adds the block-sums into each block on the second phase
 * (down-sweep) of block scan.
 *
 * Assumption: input vector x is split into blocks.
 *
 * @param [in] x Input 1D vector.
 * @param [in] sums Tensor containing sum-reduction of consecutive blocks of x.
 * @param [in] tile_length Length of kernel tiles to be used for performance
 * optimization.
 * @return Returns a vector where each i-th block of x has been added
 * sum(sums[:i]).
 */
at::Tensor run_complete_blocks(const at::Tensor& x, const at::Tensor& sums,
                               int tile_length) {
  const at::Device device = x.options().device();
  const auto dtype = x.options().dtype();

  const uint32_t vec_len = x.numel();
  const uint32_t num_blocks = sums.numel();
  const uint32_t tile_len = static_cast<uint32_t>(tile_length);
  const at::Tensor z = at::empty_like(x);

  const CompleteBlocksTiling tiling{vec_len, num_blocks, tile_len};
  uint8_t* tiling_device = alloc_copy_tiling(tiling);

  const at::Tensor workspace_tensor = alloc_workspace(0, device);

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  if (dtype == torch::kFloat) {
    ACLRT_LAUNCH_KERNEL(complete_blocks_fp32)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(sums.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  } else if (dtype == torch::kInt32) {
    ACLRT_LAUNCH_KERNEL(complete_blocks_int32)
    (num_blocks, acl_stream, const_cast<void*>(x.storage().data()),
     const_cast<void*>(sums.storage().data()),
     const_cast<void*>(z.storage().data()),
     const_cast<void*>(workspace_tensor.storage().data()), tiling_device);
  }

  aclrtFree(tiling_device);

  return z;
}

}  // namespace tcuscan
