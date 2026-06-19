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

#include "aclrtlaunch_kda_chunk_h.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Recurrent hidden-state update kernel for KDA (per-dim gate).
 *
 * Advances the [K, V] hidden state S chunk by chunk:
 *   v_corr = U - W @ S
 *   k_rest = K * exp(g_total - g_cs)          (per-dim decay)
 *   S_next = diag(exp(g_total)) @ S + k_rest^T @ v_corr
 *
 * Requires the kernel to be compiled with -DGDN_H, -DGDN_D, -DGDN_C.
 *
 * @param [in] K          Keys [HV, total_tokens, D] fp16, head-major.
 * @param [in] W          wy_kda output [total_tokens, HV, D] fp16, BSND.
 * @param [in] U          Pre-residual values [total_tokens, HV, D] fp16, BSND.
 * @param [in] G          Per-dim cumulative gate [HV, total_tokens, D] fp32,
 * head-major.
 * @param [in] cu_seqlens Cumulative sequence lengths [batch+1] int32.
 *                        Pass a single-element tensor to use fixed seq_len.
 * @param [in] batch_size Number of sequences.
 * @param [in] seq_len    Uniform sequence length (ignored when cu_seqlens
 *                        has more than one element).
 * @param [in] total_chunks Total chunks across all sequences, used to
 *                          pre-allocate the S snapshot output.
 * @param [in] chunk_size Chunk size C (must match the GDN_C compile flag).
 *                        Used only to size the per-core workspace.
 * @return std::tuple<at::Tensor, at::Tensor>
 *         (S  [total_chunks, HV, D, D] fp16  — state snapshots,
 *          V_corr [total_tokens, HV, D] fp16 — corrected values)
 */
std::tuple<at::Tensor, at::Tensor> run_kda_chunk_h(
    const at::Tensor& K, const at::Tensor& W, const at::Tensor& U,
    const at::Tensor& G, const at::Tensor& cu_seqlens, int64_t batch_size,
    int64_t seq_len, int64_t total_chunks, int64_t chunk_size) {
  TORCH_CHECK(K.device().type() == DEVICE_TYPE,
              "kda_chunk_h: K must be on NPU, got ", K.device());
  TORCH_CHECK(K.scalar_type() == at::kHalf, "kda_chunk_h: K must be fp16, got ",
              K.scalar_type());
  TORCH_CHECK(W.scalar_type() == at::kHalf, "kda_chunk_h: W must be fp16, got ",
              W.scalar_type());
  TORCH_CHECK(U.scalar_type() == at::kHalf, "kda_chunk_h: U must be fp16, got ",
              U.scalar_type());
  TORCH_CHECK(G.scalar_type() == at::kFloat,
              "kda_chunk_h: G must be fp32, got ", G.scalar_type());
  TORCH_CHECK(K.dim() == 3, "kda_chunk_h: K must be 3D [HV, total_tokens, D]");
  TORCH_CHECK(W.dim() == 3, "kda_chunk_h: W must be 3D [total_tokens, HV, D]");
  TORCH_CHECK(U.dim() == 3, "kda_chunk_h: U must be 3D [total_tokens, HV, D]");
  TORCH_CHECK(G.dim() == 3, "kda_chunk_h: G must be 3D [HV, total_tokens, D]");

  const int64_t HV = K.size(0);
  const int64_t total_tokens = K.size(1);
  const int64_t D = K.size(2);

  const at::TensorOptions half_opts = K.options();
  const at::Tensor S = at::zeros({total_chunks, HV, D, D}, half_opts);
  const at::Tensor V_corr = at::zeros({total_tokens, HV, D}, half_opts);

  const uint32_t block_dim = GetNumCubeCores();
  // Per-core workspace: (3*C*D + 2*D*D) fp16 elements.
  // Layout: WS_WS[C,V] WS_K[C,K] WS_V[C,V] WS_S[K,V] WS_KV[K,V]
  const int64_t ws_per_core = 3 * chunk_size * D + 2 * D * D;
  const at::Tensor workspace =
      at::zeros({static_cast<int64_t>(block_dim) * ws_per_core}, half_opts);

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() > 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(kda_chunk_h, block_dim, K, W, U, G, S, V_corr, workspace,
                  cu_seqlens_ptr, batch_size, seq_len, total_tokens);

  return {S, V_corr};
}

}  // namespace pto_isa_ops
