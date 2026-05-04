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

#include "aclrtlaunch_chunk_cumsum_fp32.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Computes a chunked prefix sum (cumulative sum) of gate values along
 * the time dimension.
 *
 * Per chunk of GDN_C tokens, independently per head h:
 *   g_sum[t, h] = sum_{i=0}^{t} g[i, h]   for t = 0 .. valid-1
 *
 * This enables downstream kernels to compute exponential decay coefficients:
 *   exp(g_sum[i] - g_sum[j]) gives the cumulative gate from token j to i.
 *
 * @param [in] g          Float32 input tensor with shape [total_tokens, H].
 * @param [in] batch_size Number of sequences in the batch.
 * @param [in] seq_len    Sequence length (tokens per sequence). Used only when
 *                        cu_seqlens has a single element (fixed-length path).
 * @param [in] cu_seqlens Optional cumulative sequence lengths tensor of shape
 *                        [batch_size + 1], dtype int32. When provided (numel >
 * 1) the variable-length path is taken.
 * @return at::Tensor Float32 output tensor with shape [total_tokens, H]
 *                    containing the per-chunk prefix sums.
 */
at::Tensor run_chunk_cumsum(const at::Tensor& g, int64_t batch_size,
                            int64_t seq_len,
                            const at::Tensor& cu_seqlens = at::zeros({1})) {
  TORCH_CHECK(g.device().type() == DEVICE_TYPE,
              "pto_chunk_cumsum: tensor must be on NPU, got ", g.device());
  TORCH_CHECK(g.scalar_type() == at::kFloat,
              "pto_chunk_cumsum: g must be float32, got ", g.scalar_type());
  TORCH_CHECK(g.dim() == 2,
              "pto_chunk_cumsum: g must be 2D [total_tokens, H], got ", g.dim(),
              "D");
  TORCH_CHECK(g.is_contiguous(), "pto_chunk_cumsum: g must be contiguous");
  TORCH_CHECK(batch_size > 0,
              "pto_chunk_cumsum: batch_size must be positive, got ",
              batch_size);
  TORCH_CHECK(seq_len > 0, "pto_chunk_cumsum: seq_len must be positive, got ",
              seq_len);

  at::Tensor g_sum = at::empty_like(g);

  const uint32_t block_dim = GetNumVectorCores();

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(chunk_cumsum_fp32, block_dim, g, g_sum, cu_seqlens_ptr,
                  batch_size, seq_len);

  return g_sum;
}

}  // namespace pto_isa_ops
