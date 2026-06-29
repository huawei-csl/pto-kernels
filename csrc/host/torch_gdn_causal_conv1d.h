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

#include "aclrtlaunch_gdn_causal_conv1d_batched_bf16.h"
#include "aclrtlaunch_gdn_causal_conv1d_batched_fp16.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Depthwise causal conv1d + per-channel bias + optional SiLU.
 *
 * Computes y[b,i,c] = act(bias[c] + sum_{k} W[k,c] * x[b,i-K+1+k,c])
 * with zero-padding for i < 0. Supports fp16 and bf16 I/O with fp32
 * accumulation. A 2D input [seqLen, channels] is treated as batch=1 and the
 * returned tensor has the same rank as the input.
 *
 * @param [in] x          Input tensor [seqLen, channels] or [batch, seqLen,
 *                        channels] fp16 or bf16, contiguous.
 * @param [in] weights    Filter weights [K, channels] fp32, contiguous.
 * @param [in] bias       Per-channel bias [channels] fp32, contiguous.
 * @param [in] activation Whether to apply SiLU after bias add (default true).
 * @return at::Tensor     Output same shape and dtype as x.
 */
at::Tensor run_gdn_causal_conv1d(const at::Tensor& x, const at::Tensor& weights,
                                 const at::Tensor& bias,
                                 bool activation = true) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "gdn_causal_conv1d: x must be on NPU, got ", x.device());
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "gdn_causal_conv1d: x must be fp16 or bf16, got ",
              x.scalar_type());
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
              "gdn_causal_conv1d: x must be 2D [seqLen, channels] or 3D "
              "[batch, seqLen, channels], got ",
              x.dim(), "D");
  TORCH_CHECK(x.is_contiguous(), "gdn_causal_conv1d: x must be contiguous");
  TORCH_CHECK(weights.scalar_type() == at::kFloat,
              "gdn_causal_conv1d: weights must be fp32, got ",
              weights.scalar_type());
  TORCH_CHECK(weights.is_contiguous(),
              "gdn_causal_conv1d: weights must be contiguous");
  TORCH_CHECK(bias.dim() == 1 && bias.scalar_type() == at::kFloat,
              "gdn_causal_conv1d: bias must be 1D fp32");
  TORCH_CHECK(bias.is_contiguous(),
              "gdn_causal_conv1d: bias must be contiguous");

  const bool was2d = x.dim() == 2;
  const at::Tensor x3d = was2d ? x.unsqueeze(0) : x;

  const uint32_t batch = static_cast<uint32_t>(x3d.size(0));
  const uint32_t seqLen = static_cast<uint32_t>(x3d.size(1));
  const uint32_t channels = static_cast<uint32_t>(x3d.size(2));
  const uint32_t applyActivation = activation ? 1u : 0u;

  at::Tensor output = at::empty_like(x3d);
  const uint32_t block_dim = GetNumVectorCores();

  if (x.scalar_type() == at::kHalf) {
    EXEC_KERNEL_CMD(gdn_causal_conv1d_batched_fp16, block_dim, x3d, output,
                    weights, bias, batch, seqLen, channels, applyActivation);
  } else {
    EXEC_KERNEL_CMD(gdn_causal_conv1d_batched_bf16, block_dim, x3d, output,
                    weights, bias, batch, seqLen, channels, applyActivation);
  }

  return was2d ? output.squeeze(0) : output;
}

}  // namespace pto_isa_ops
