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
#include "aclrtlaunch_gdn_causal_conv1d_fp16.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Depthwise causal conv1d + per-channel bias + SiLU (single sequence).
 *
 * Computes y[i,c] = silu(bias[c] + sum_{k} W[k,c] * x[i-K+1+k, c])
 * with zero-padding for i < 0. Always applies SiLU activation.
 *
 * @param [in] x       Input tensor [seqLen, channels] fp16, contiguous.
 * @param [in] weights Filter weights [K, channels] fp32, contiguous.
 * @param [in] bias    Per-channel bias [channels] fp32, contiguous.
 * @return at::Tensor  Output [seqLen, channels] fp16.
 */
at::Tensor run_gdn_causal_conv1d(const at::Tensor& x, const at::Tensor& weights,
                                 const at::Tensor& bias) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "gdn_causal_conv1d: x must be on NPU, got ", x.device());
  TORCH_CHECK(x.scalar_type() == at::kHalf,
              "gdn_causal_conv1d: x must be fp16, got ", x.scalar_type());
  TORCH_CHECK(x.dim() == 2,
              "gdn_causal_conv1d: x must be 2D [seqLen, channels], got ",
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

  const uint32_t seqLen = static_cast<uint32_t>(x.size(0));
  const uint32_t channels = static_cast<uint32_t>(x.size(1));

  at::Tensor output = at::empty_like(x);
  const uint32_t block_dim = GetNumVectorCores();

  EXEC_KERNEL_CMD(gdn_causal_conv1d_fp16, block_dim, x, output, weights, bias,
                  seqLen, channels);
  return output;
}

/**
 * @brief Depthwise causal conv1d + per-channel bias + optional SiLU (batched).
 *
 * Computes y[b,i,c] = act(bias[c] + sum_{k} W[k,c] * x[b,i-K+1+k,c])
 * with zero-padding for i < 0. Supports fp16 and bf16 I/O with fp32
 * accumulation.
 *
 * @param [in] x          Input tensor [batch, seqLen, channels] fp16 or bf16.
 * @param [in] weights    Filter weights [K, channels] fp32, contiguous.
 * @param [in] bias       Per-channel bias [channels] fp32, contiguous.
 * @param [in] activation Whether to apply SiLU after bias add (default true).
 * @return at::Tensor     Output same shape and dtype as x.
 */
at::Tensor run_gdn_causal_conv1d_batched(const at::Tensor& x,
                                         const at::Tensor& weights,
                                         const at::Tensor& bias,
                                         bool activation = true) {
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "gdn_causal_conv1d_batched: x must be on NPU, got ", x.device());
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "gdn_causal_conv1d_batched: x must be fp16 or bf16, got ",
              x.scalar_type());
  TORCH_CHECK(
      x.dim() == 3,
      "gdn_causal_conv1d_batched: x must be 3D [batch, seqLen, channels], "
      "got ",
      x.dim(), "D");
  TORCH_CHECK(x.is_contiguous(),
              "gdn_causal_conv1d_batched: x must be contiguous");
  TORCH_CHECK(weights.scalar_type() == at::kFloat,
              "gdn_causal_conv1d_batched: weights must be fp32, got ",
              weights.scalar_type());
  TORCH_CHECK(weights.is_contiguous(),
              "gdn_causal_conv1d_batched: weights must be contiguous");
  TORCH_CHECK(bias.dim() == 1 && bias.scalar_type() == at::kFloat,
              "gdn_causal_conv1d_batched: bias must be 1D fp32");
  TORCH_CHECK(bias.is_contiguous(),
              "gdn_causal_conv1d_batched: bias must be contiguous");

  const uint32_t batch = static_cast<uint32_t>(x.size(0));
  const uint32_t seqLen = static_cast<uint32_t>(x.size(1));
  const uint32_t channels = static_cast<uint32_t>(x.size(2));
  const uint32_t applyActivation = activation ? 1u : 0u;

  at::Tensor output = at::empty_like(x);
  const uint32_t block_dim = GetNumVectorCores();

  if (x.scalar_type() == at::kHalf) {
    EXEC_KERNEL_CMD(gdn_causal_conv1d_batched_fp16, block_dim, x, output,
                    weights, bias, batch, seqLen, channels, applyActivation);
  } else if (x.scalar_type() == at::kBFloat16) {
    EXEC_KERNEL_CMD(gdn_causal_conv1d_batched_bf16, block_dim, x, output,
                    weights, bias, batch, seqLen, channels, applyActivation);
  }
  return output;
}

}  // namespace pto_isa_ops
