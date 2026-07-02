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

#include "aclrtlaunch_kda_gate_cumsum.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Within-chunk prefix sum of KDA per-dimension gate vectors.
 *
 * Per chunk of GDN_C tokens, independently per head h and key-dim d:
 *   g_sum[t, h, d] = sum_{i=0}^{t} g[i, h, d]   for t = 0 .. valid-1
 *
 * Input g is fp16 (model dtype); accumulation is done in fp32 to prevent
 * precision loss (per-chunk cumulative sum reaches ~-64, where fp16 step
 * size ~0.06 would corrupt downstream exp(g_cs) computations).
 *
 * Difference from pto_chunk_cumsum (GDN):
 *   GDN gate shape: [T, H]     — scalar gate per (token, head)
 *   KDA gate shape: [T, HV, D] — vector gate per (token, head, key-dim)
 *
 * @param [in] g          fp16 input [total_tokens, HV, D], BSND layout.
 * @param [in] batch_size Number of sequences.
 * @param [in] seq_len    Uniform sequence length. Ignored when cu_seqlens
 *                        has more than one element.
 * @param [in] cu_seqlens Cumulative sequence lengths [batch+1] int32.
 *                        Pass a single-element tensor to use fixed seq_len.
 * @return at::Tensor     fp32 output [total_tokens, HV, D] — prefix sums.
 */
at::Tensor run_kda_gate_cumsum(const at::Tensor& g, int64_t batch_size,
                               int64_t seq_len,
                               const at::Tensor& cu_seqlens = at::zeros({1})) {
  TORCH_CHECK(g.device().type() == DEVICE_TYPE,
              "kda_gate_cumsum: g must be on NPU, got ", g.device());
  TORCH_CHECK(g.scalar_type() == at::kHalf,
              "kda_gate_cumsum: g must be fp16, got ", g.scalar_type());
  TORCH_CHECK(g.dim() == 3,
              "kda_gate_cumsum: g must be 3D [total_tokens, HV, D], got ",
              g.dim(), "D");
  TORCH_CHECK(g.is_contiguous(), "kda_gate_cumsum: g must be contiguous");
  TORCH_CHECK(batch_size > 0,
              "kda_gate_cumsum: batch_size must be positive, got ", batch_size);
  TORCH_CHECK(cu_seqlens.numel() > 1 || seq_len > 0,
              "kda_gate_cumsum: seq_len must be positive if no cu_seqlens "
              "provided, got ",
              seq_len);

  const int64_t total_tokens = g.size(0);
  const int64_t HV = g.size(1);
  const int64_t D = g.size(2);

  const at::Tensor g_sum =
      at::zeros({total_tokens, HV, D}, g.options().dtype(at::kFloat));

  const uint32_t block_dim = GetNumVectorCores();

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() > 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(kda_gate_cumsum, block_dim, g, g_sum, cu_seqlens_ptr,
                  batch_size, seq_len);

  return g_sum;
}

}  // namespace pto_isa_ops
