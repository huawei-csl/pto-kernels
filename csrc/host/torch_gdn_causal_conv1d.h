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

// (ringSize, maxTileWidth)
// Must match FOR_EACH_RING_SIZE in kernel_gdn_causal_conv1d.cpp.
// A larger ring needs a smaller tile to fit the 192 KiB UB.
#define FOR_EACH_RING_SIZE(DO) \
  DO(2, 4096) DO(4, 3072) DO(8, 1536) DO(16, 896) DO(32, 384) DO(64, 128)

// ACLRT_LAUNCH_KERNEL resolves a kernel name to its aclrtlaunch_ stub.
#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

// Forward-declare all 12 (6 ring sizes × 2 dtypes) launch stubs instead of
// includes.
// clang-format off
#define DECLARE_BATCHED_STUB(rs, mw, dtype)                                             \
  extern "C" uint32_t aclrtlaunch_gdn_causal_conv1d_ ## dtype ## _rs ## rs(             \
      uint32_t, aclrtStream, void*, void*, void*, void*, void*,                         \
      uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
#define DECLARE_BATCHED_STUB_FP16(rs, mw) DECLARE_BATCHED_STUB(rs, mw, fp16)
#define DECLARE_BATCHED_STUB_BF16(rs, mw) DECLARE_BATCHED_STUB(rs, mw, bf16)

FOR_EACH_RING_SIZE(DECLARE_BATCHED_STUB_FP16)
FOR_EACH_RING_SIZE(DECLARE_BATCHED_STUB_BF16)

#undef DECLARE_BATCHED_STUB_BF16
#undef DECLARE_BATCHED_STUB_FP16
#undef DECLARE_BATCHED_STUB
// clang-format on

#include "utils.h"

namespace pto_isa_ops {

namespace {  // file-local helpers

constexpr uint32_t roundUpToPow2(uint32_t w) {
  uint32_t n = (w != 0u) ? w - 1u : 0u;
  n |= n >> 1u;
  n |= n >> 2u;
  n |= n >> 4u;
  n |= n >> 8u;
  n |= n >> 16u;
  return n + 1u;
}

constexpr bool isSupportedWidth(uint32_t w) { return w >= 2u && w <= 64u; }

}  // namespace

/**
 * @brief Depthwise causal conv1d + per-channel bias + optional SiLU.
 *
 * Computes y[b,i,c] = act( bias[c] + sum_{k} W[k,c] * x_ext[b, i-K+1+k, c] )
 * where x_ext[-K+1..-1] = conv_states[b, 0..K-2, c] (last K-1 rows, when
 * conv_states is provided) or zero (default), and x_ext[0..L-1] = x[b,...,c].
 *
 * K is deduced from weights.shape[0] and must be in [2..64].
 * Weights and bias must have the same dtype as x (fp16 or bf16).
 *
 * @param [in] x                  Input [L, C] or [B, L, C], fp16 or bf16,
 *                                contiguous.
 * @param [in] weights            Filter [K, C], same dtype as x, contiguous.
 * @param [in] bias               Bias [C], same dtype as x, contiguous; or
 *                                None (no bias).
 * @param [in] conv_states        History [K-1, C] or [B, K-1, C] (same rank
 *                                as x), same dtype as x, contiguous; or
 *                                None (zero-padding).
 * @param [in] activation         Apply SiLU after bias add. Default true.
 * @return at::Tensor             Output, same shape and dtype as x.
 */
at::Tensor run_gdn_causal_conv1d(const at::Tensor& x, const at::Tensor& weights,
                                 const c10::optional<at::Tensor>& bias,
                                 const c10::optional<at::Tensor>& conv_states,
                                 bool activation) {
  // ---- input validation ----
  TORCH_CHECK(x.device().type() == DEVICE_TYPE,
              "gdn_causal_conv1d: x must be on NPU, got ", x.device());
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
              "gdn_causal_conv1d: x must be fp16 or bf16, got ",
              x.scalar_type());
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
              "gdn_causal_conv1d: x must be 2D [L, C] or 3D [B, L, C], got ",
              x.dim(), "D");
  TORCH_CHECK(x.is_contiguous(), "gdn_causal_conv1d: x must be contiguous");

  const at::ScalarType dtype = x.scalar_type();

  TORCH_CHECK(weights.dim() == 2,
              "gdn_causal_conv1d: weights must be 2D [K, C], got shape ",
              weights.sizes());
  TORCH_CHECK(weights.scalar_type() == dtype,
              "gdn_causal_conv1d: weights dtype must match x dtype (", dtype,
              "), got ", weights.scalar_type());
  TORCH_CHECK(weights.is_contiguous(),
              "gdn_causal_conv1d: weights must be contiguous");

  const bool was2d = x.dim() == 2;
  const at::Tensor x3d = was2d ? x.unsqueeze(0) : x;

  const uint32_t batch = static_cast<uint32_t>(x3d.size(0));
  const uint32_t seqLen = static_cast<uint32_t>(x3d.size(1));
  const uint32_t channels = static_cast<uint32_t>(x3d.size(2));
  const uint32_t K = static_cast<uint32_t>(weights.size(0));

  TORCH_CHECK(weights.size(1) == static_cast<int64_t>(channels),
              "gdn_causal_conv1d: weights.shape[1] (", weights.size(1),
              ") must equal channels (", channels, ")");
  TORCH_CHECK(isSupportedWidth(K),
              "gdn_causal_conv1d: filter width K must be in [2..64], got ", K);
  TORCH_CHECK(
      channels % 16 == 0,
      "gdn_causal_conv1d: channels must be a multiple of 16 for fp16/bf16 "
      "vector alignment, got ",
      channels);

  const bool has_bias = bias.has_value();
  if (has_bias) {
    TORCH_CHECK(
        bias->dim() == 1 && bias->size(0) == static_cast<int64_t>(channels),
        "gdn_causal_conv1d: bias must be 1D [C] with C=", channels,
        ", got shape ", bias->sizes());
    TORCH_CHECK(bias->scalar_type() == dtype,
                "gdn_causal_conv1d: bias dtype must match x dtype (", dtype,
                "), got ", bias->scalar_type());
    TORCH_CHECK(bias->is_contiguous(),
                "gdn_causal_conv1d: bias must be contiguous");
  }

  const bool use_states = conv_states.has_value();
  if (use_states) {
    const std::vector<int64_t> expected_sizes =
        was2d ? std::vector<int64_t>{(int64_t)(K - 1), (int64_t)channels}
              : std::vector<int64_t>{(int64_t)batch, (int64_t)(K - 1), (int64_t)channels};
    TORCH_CHECK(conv_states->sizes() == at::IntArrayRef(expected_sizes),
                "gdn_causal_conv1d: conv_states must have shape ", at::IntArrayRef(expected_sizes),
                " (same rank as x, middle dim K-1), got ", conv_states->sizes());
    TORCH_CHECK(conv_states->scalar_type() == dtype,
                "gdn_causal_conv1d: conv_states dtype must match x dtype (",
                dtype, "), got ", conv_states->scalar_type());
    TORCH_CHECK(conv_states->is_contiguous(),
                "gdn_causal_conv1d: conv_states must be contiguous");
  }

  // ---- kernel launch ----
  const at::Tensor biasArg = has_bias ? *bias : at::empty({0}, x.options());

  // When has_initial_state is false, later sequence chunks with large K (K >
  // 32) can still have jstart < 0 and read from conv_states. Provide zeros so
  // those halo reads return the correct zero-pad value without dereferencing an
  // invalid pointer.
  at::Tensor convStatesArg;
  uint32_t hasConvStates, stateLen;
  if (use_states) {
    convStatesArg = was2d ? conv_states->unsqueeze(0) : *conv_states;
    hasConvStates = 1u;
    stateLen = K - 1u;
  } else {
    convStatesArg = at::zeros(
        {(int64_t)batch, (int64_t)(K - 1), (int64_t)channels}, x.options());
    hasConvStates = 0u;
    stateLen = K - 1u;
  }

  const uint32_t applyActivation = activation ? 1u : 0u;
  const uint32_t hasBias = has_bias ? 1u : 0u;
  const uint32_t ringSize = roundUpToPow2(K);
  const uint32_t block_dim = GetNumVectorCores();

  at::Tensor output = at::empty_like(x3d);

  // clang-format off
#define DISPATCH_DTYPE(rs, mw, dtype)                                             \
  case rs:                                                                        \
    EXEC_KERNEL_CMD(gdn_causal_conv1d_ ## dtype ## _rs ## rs, block_dim,          \
                    x3d, output, weights, biasArg, convStatesArg,                 \
                    batch, seqLen, channels, stateLen, K,                         \
                    applyActivation, hasBias, hasConvStates);                     \
    break;
#define DISPATCH_FP16(rs, mw) DISPATCH_DTYPE(rs, mw, fp16)
#define DISPATCH_BF16(rs, mw) DISPATCH_DTYPE(rs, mw, bf16)

  switch (dtype) {
    case at::kHalf:
      switch (ringSize) {
        FOR_EACH_RING_SIZE(DISPATCH_FP16)
        default:
          break;
      }
      break;
    case at::kBFloat16:
      switch (ringSize) {
        FOR_EACH_RING_SIZE(DISPATCH_BF16)
        default:
          break;
      }
      break;
    default:
      break;
  }

#undef DISPATCH_BF16
#undef DISPATCH_FP16
#undef DISPATCH_DTYPE
  // clang-format on

  return was2d ? output.squeeze(0) : output;
}

}  // namespace pto_isa_ops
