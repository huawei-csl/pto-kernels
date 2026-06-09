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
 * Advances the K×V hidden state S chunk by chunk:
 *   v_corr = u - w @ S
 *   k_rest = k * exp(g_total - g_cs)
 *   S_new  = exp(g_total).unsqueeze(-1) * S + k_rest^T @ v_corr
 *
 * Requires the kernel to be compiled with -DGDN_H, -DGDN_D, -DGDN_C.
 *
 * @param [in]  K           Keys [B, HV, T, K_DIM] fp16, head-major layout.
 * @param [in]  W           wy_kda output [B, T, HV, K_DIM] fp16, BSND layout.
 * @param [in]  U           wy_kda values [B, T, HV, V_DIM] fp16, BSND layout.
 * @param [in]  G           Per-dim cumulative gate sum [B, HV, T, K_DIM] fp16.
 * @param [out] S_out       State snapshots [total_chunks, HV, K_DIM, V_DIM]
 * fp16.
 * @param [out] V_corr_out  Corrected values [B, T, HV, V_DIM] fp16, BSND
 * layout.
 * @param [in]  cu_seqlens  Cumulative sequence lengths [batch+1] int32.
 *                          Pass a single-element tensor to use fixed seq_len.
 * @param [in]  batch_size  Number of sequences.
 * @param [in]  seq_len     Uniform sequence length (ignored when cu_seqlens has
 *                          more than one element).
 * @param [in]  total_tokens Total token count across all sequences.
 */
void run_kda_chunk_h(const at::Tensor& K, const at::Tensor& W,
                     const at::Tensor& U, const at::Tensor& G,
                     const at::Tensor& S_out, const at::Tensor& V_corr_out,
                     const at::Tensor& cu_seqlens, int64_t batch_size,
                     int64_t seq_len, int64_t total_tokens) {
  TORCH_CHECK(K.device().type() == DEVICE_TYPE,
              "chunk_h_kda: K must be on NPU, got ", K.device());
  TORCH_CHECK(K.scalar_type() == at::kHalf, "chunk_h_kda: K must be fp16, got ",
              K.scalar_type());
  TORCH_CHECK(W.scalar_type() == at::kHalf, "chunk_h_kda: W must be fp16, got ",
              W.scalar_type());
  TORCH_CHECK(U.scalar_type() == at::kHalf, "chunk_h_kda: U must be fp16, got ",
              U.scalar_type());
  TORCH_CHECK(G.scalar_type() == at::kHalf, "chunk_h_kda: G must be fp16, got ",
              G.scalar_type());
  TORCH_CHECK(K.dim() == 4, "chunk_h_kda: K must be 4D [B, HV, T, K_DIM]");
  TORCH_CHECK(W.dim() == 4, "chunk_h_kda: W must be 4D [B, T, HV, K_DIM]");
  TORCH_CHECK(U.dim() == 4, "chunk_h_kda: U must be 4D [B, T, HV, V_DIM]");
  TORCH_CHECK(G.dim() == 4, "chunk_h_kda: G must be 4D [B, HV, T, K_DIM]");

  // K_DIM from head-major K: [B, HV, T, K_DIM]
  const int64_t K_DIM = K.size(3);
  const int64_t V_DIM = U.size(3);
  const int64_t KV = K_DIM * V_DIM;

  const uint32_t block_dim = GetNumCubeCores();
  // Per-core workspace: 5 slots × K*V fp16 elements.
  const at::Tensor workspace =
      at::zeros({static_cast<int64_t>(block_dim) * 5 * KV}, K.options());

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() > 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(kda_chunk_h, block_dim, K, W, U, G, S_out, V_corr_out,
                  workspace, cu_seqlens_ptr, batch_size, seq_len, total_tokens);
}

}  // namespace pto_isa_ops
