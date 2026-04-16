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

#include <limits>

extern "C" uint32_t swiglu_fp16(void* x, void* y, uint32_t batch,
                                uint32_t input_n);
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Runs fp16 SwiGLU over the last dimension of a 2D tensor.
 *
 * The input is interpreted as `x = [gate | up]` along `dim`, and the output is
 * `silu(gate) * up`.
 *
 * @param [in] x Contiguous fp16 input tensor with shape [batch, 2 * N].
 * @param [in] dim Split dimension. Only -1/1 is currently supported.
 * @return at::Tensor Contiguous fp16 output tensor with shape [batch, N].
 */
at::Tensor run_swiglu(const at::Tensor& x, int64_t dim = -1) {
  if (x.dim() != 2) {
    throw std::runtime_error("`pto_swiglu` expects a 2D input tensor.");
  }
  if (dim < 0) {
    dim += x.dim();
  }
  if (dim != 1) {
    throw std::runtime_error("`pto_swiglu` currently supports only dim=-1.");
  }
  if (x.scalar_type() != at::kHalf) {
    throw std::runtime_error("`pto_swiglu` supports only fp16 input.");
  }
  if (!x.is_contiguous()) {
    throw std::runtime_error("`pto_swiglu` expects a contiguous input tensor.");
  }

  const auto batch_i64 = x.size(0);
  const auto input_n_i64 = x.size(1);
  if (batch_i64 <= 0 || input_n_i64 <= 0 || (input_n_i64 & 1) != 0) {
    throw std::runtime_error(
        "`pto_swiglu` input shape must be [batch, 2 * N] with positive N.");
  }

  const auto output_n_i64 = input_n_i64 / 2;
  if (batch_i64 > std::numeric_limits<uint32_t>::max() ||
      input_n_i64 > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("`pto_swiglu` dimensions exceed uint32_t range.");
  }

  const uint32_t batch = static_cast<uint32_t>(batch_i64);
  const uint32_t input_n = static_cast<uint32_t>(input_n_i64);
  const uint32_t block_dim = GetNumCubeCores();

  at::Tensor y = at::empty({batch_i64, output_n_i64}, x.options());
  EXEC_KERNEL_CMD(swiglu_fp16, block_dim, x, y, batch, input_n);
  return y;
}

}  // namespace pto_isa_ops
