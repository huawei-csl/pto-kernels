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

#include "aclrtlaunch_gdn_chunk_h.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Recurrent hidden-state update kernel for GatedDeltaNet (chunk_h).
 *
 * Advances the D×D hidden state S chunk by chunk:
 *   ws      = W @ S
 *   V_new   = U - ws
 *   K_tilde = exp(g_last - g) * K
 *   S_next  = exp(g_last) * S + K_tilde^T @ V_new
 *
 * Requires the kernel to be compiled with -DGDN_H, -DGDN_HG, -DGDN_D, -DGDN_C.
 *
 * @param [in] K          Keys [total_tokens, Hg, D] fp16.
 * @param [in] W          wy_fast output [total_tokens, H, D] fp16.
 * @param [in] U          Pre-residual values [total_tokens, H, D] fp16.
 * @param [in] G          Cumulative gates [H, total_tokens] fp32.
 * @param [in] cu_seqlens Cumulative sequence lengths [batch+1] int32.
 *                        Pass a single-element tensor to use fixed seq_len.
 * @param [in] batch_size Number of sequences.
 * @param [in] seq_len    Uniform sequence length (ignored when cu_seqlens
 *                        has more than one element).
 * @param [in] total_chunks Total chunks across all sequences, used to
 *                          pre-allocate the S output.
 * @return std::tuple<at::Tensor, at::Tensor, at::Tensor>
 *         (S [total_chunks, H, D, D] fp16,
 *          V [total_tokens, H, D] fp16,
 *          FS [batch_size, H, D, D] fp16)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> run_gdn_chunk_h(
    const at::Tensor& K, const at::Tensor& W, const at::Tensor& U,
    const at::Tensor& G, const at::Tensor& cu_seqlens, int64_t batch_size,
    int64_t seq_len, int64_t total_chunks) {
  TORCH_CHECK(K.device().type() == DEVICE_TYPE,
              "chunk_h: K must be on NPU, got ", K.device());
  TORCH_CHECK(K.scalar_type() == at::kHalf, "chunk_h: K must be fp16, got ",
              K.scalar_type());
  TORCH_CHECK(W.scalar_type() == at::kHalf, "chunk_h: W must be fp16, got ",
              W.scalar_type());
  TORCH_CHECK(U.scalar_type() == at::kHalf, "chunk_h: U must be fp16, got ",
              U.scalar_type());
  TORCH_CHECK(G.scalar_type() == at::kFloat, "chunk_h: G must be fp32, got ",
              G.scalar_type());
  TORCH_CHECK(K.dim() == 3, "chunk_h: K must be 3D [total_tokens, Hg, D]");
  TORCH_CHECK(W.dim() == 3, "chunk_h: W must be 3D [total_tokens, H, D]");
  TORCH_CHECK(U.dim() == 3, "chunk_h: U must be 3D [total_tokens, H, D]");
  TORCH_CHECK(G.dim() == 2, "chunk_h: G must be 2D [H, total_tokens]");

  const int64_t total_tokens = K.size(0);
  const int64_t H = W.size(1);
  const int64_t D = K.size(2);
  const int64_t DD = D * D;

  const at::TensorOptions half_opts = K.options();
  const at::Tensor S = at::zeros({total_chunks, H, D, D}, half_opts);
  const at::Tensor V = at::zeros({total_tokens, H, D}, half_opts);
  const at::Tensor FS = at::zeros({batch_size, H, D, D}, half_opts);

  const uint32_t block_dim = GetNumCubeCores();
  // Per-core workspace: 4 * D*D half elements (WS_WS, WS_K, WS_S, WS_KV).
  const at::Tensor workspace =
      at::zeros({static_cast<int64_t>(block_dim) * DD * 4}, half_opts);

  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() > 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(gdn_chunk_h, block_dim, K, W, U, G, S, V, FS, workspace,
                  cu_seqlens_ptr, batch_size, seq_len, total_tokens);

  return {S, V, FS};
}

}  // namespace pto_isa_ops
