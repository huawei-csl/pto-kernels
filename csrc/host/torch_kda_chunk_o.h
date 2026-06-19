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

#include "aclrtlaunch_kda_chunk_o.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Output computation kernel for KDA (per-dim gate).
 *
 * Computes per chunk:
 *   q_eff = Q * exp(g_cs)
 *   k_eff = K * exp(-g_cs)
 *   Aqk   = tril(q_eff @ k_eff^T, diagonal=0)   (inclusive causal mask)
 *   O     = q_eff @ S + Aqk @ V_corr
 *
 * Requires the kernel to be compiled with -DGDN_H, -DGDN_D, -DGDN_C.
 *
 * @param [in] Q          Queries [HV, total_tokens, D] fp16, head-major.
 * @param [in] K          Keys    [HV, total_tokens, D] fp16, head-major.
 * @param [in] V_corr     Corrected values from kda_chunk_h [total_tokens, HV,
 * D] fp16, BSND.
 * @param [in] S          State snapshots from kda_chunk_h [total_chunks, HV, D,
 * D] fp16.
 * @param [in] G          Per-dim cumulative gate [HV, total_tokens, D] fp32,
 * head-major.
 * @param [in] Mask       Inclusive lower-triangular mask [C, C] fp32
 *                        (1 where row >= col, 0 elsewhere).
 * @param [in] cu_seqlens Cumulative sequence lengths [batch+1] int32.
 *                        Pass a single-element tensor to use fixed seq_len.
 * @param [in] batch_size Number of sequences.
 * @param [in] seq_len    Uniform sequence length (ignored when cu_seqlens
 *                        has more than one element).
 * @param [in] total_chunks Total chunks across all sequences.
 * @return at::Tensor     Output O [total_tokens, HV, D] fp16, BSND.
 */
at::Tensor run_kda_chunk_o(const at::Tensor& Q, const at::Tensor& K,
                           const at::Tensor& V_corr, const at::Tensor& S,
                           const at::Tensor& G, const at::Tensor& Mask,
                           const at::Tensor& cu_seqlens, int64_t batch_size,
                           int64_t seq_len, int64_t total_chunks) {
  TORCH_CHECK(Q.device().type() == DEVICE_TYPE,
              "kda_chunk_o: Q must be on NPU, got ", Q.device());
  TORCH_CHECK(Q.scalar_type() == at::kHalf, "kda_chunk_o: Q must be fp16, got ",
              Q.scalar_type());
  TORCH_CHECK(K.scalar_type() == at::kHalf, "kda_chunk_o: K must be fp16, got ",
              K.scalar_type());
  TORCH_CHECK(V_corr.scalar_type() == at::kHalf,
              "kda_chunk_o: V_corr must be fp16, got ", V_corr.scalar_type());
  TORCH_CHECK(S.scalar_type() == at::kHalf, "kda_chunk_o: S must be fp16, got ",
              S.scalar_type());
  TORCH_CHECK(G.scalar_type() == at::kFloat,
              "kda_chunk_o: G must be fp32, got ", G.scalar_type());
  TORCH_CHECK(Mask.scalar_type() == at::kFloat,
              "kda_chunk_o: Mask must be fp32, got ", Mask.scalar_type());
  TORCH_CHECK(Q.dim() == 3, "kda_chunk_o: Q must be 3D [HV, total_tokens, D]");
  TORCH_CHECK(K.dim() == 3, "kda_chunk_o: K must be 3D [HV, total_tokens, D]");
  TORCH_CHECK(V_corr.dim() == 3,
              "kda_chunk_o: V_corr must be 3D [total_tokens, HV, D]");
  TORCH_CHECK(S.dim() == 4,
              "kda_chunk_o: S must be 4D [total_chunks, HV, D, D]");
  TORCH_CHECK(G.dim() == 3, "kda_chunk_o: G must be 3D [HV, total_tokens, D]");
  TORCH_CHECK(Mask.dim() == 2, "kda_chunk_o: Mask must be 2D [C, C]");

  const int64_t HV = Q.size(0);
  const int64_t total_tokens = Q.size(1);
  const int64_t D = Q.size(2);
  const int64_t C = Mask.size(0);  // chunk_size (must match GDN_C)

  const at::TensorOptions half_opts = Q.options();
  const at::Tensor O = at::zeros({total_tokens, HV, D}, half_opts);

  const uint32_t block_dim = GetNumCubeCores();
  // Per-core workspace (fp32): 7 slots — WS_Q, WS_K, WS_V, WS_S, WS_QK, WS_QS,
  // WS_QKV. Total = 2*C*D + C*D + D*D + C*C + 2*C*D = 5*C*D + D*D + C*C fp32
  // elements.
  const int64_t ws_per_core = 5 * C * D + D * D + C * C;
  const at::TensorOptions float_opts = G.options();
  const at::Tensor workspace =
      at::zeros({static_cast<int64_t>(block_dim) * ws_per_core}, float_opts);

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() > 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(kda_chunk_o, block_dim, Q, K, V_corr, S, G, Mask, workspace,
                  O, cu_seqlens_ptr, batch_size, seq_len, total_tokens);

  return O;
}

}  // namespace pto_isa_ops
