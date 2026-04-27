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

#include "aclrtlaunch_swiglu_fp16.h"
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
  TORCH_CHECK(x.dim() == 2, "swiglu: expects a 2D input tensor, got ", x.dim(),
              "D");
  if (dim < 0) {
    dim += x.dim();
  }
  TORCH_CHECK(dim == 1, "swiglu: currently supports only dim=-1");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "swiglu: dtype must be fp16, got ",
              x.scalar_type());
  TORCH_CHECK(x.is_contiguous(), "swiglu: expects a contiguous input tensor");

  const auto batch_i64 = x.size(0);
  const auto input_n_i64 = x.size(1);
  TORCH_CHECK(
      batch_i64 > 0 && input_n_i64 > 0 && (input_n_i64 & 1) == 0,
      "swiglu: input shape must be [batch, 2 * N] with positive N, got [",
      batch_i64, ", ", input_n_i64, "]");

  const auto output_n_i64 = input_n_i64 / 2;
  TORCH_CHECK(batch_i64 <= std::numeric_limits<uint32_t>::max() &&
                  input_n_i64 <= std::numeric_limits<uint32_t>::max(),
              "swiglu: dimensions exceed uint32_t range");

  const uint32_t batch = static_cast<uint32_t>(batch_i64);
  const uint32_t input_n = static_cast<uint32_t>(input_n_i64);
  const uint32_t block_dim = GetNumCubeCores();

  at::Tensor y = at::empty({batch_i64, output_n_i64}, x.options());
  EXEC_KERNEL_CMD(swiglu_fp16, block_dim, x, y, batch, input_n);
  return y;
}

}  // namespace pto_isa_ops
