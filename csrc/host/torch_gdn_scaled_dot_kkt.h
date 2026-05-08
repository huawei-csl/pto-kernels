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

#include "aclrtlaunch_gdn_scaled_dot_kkt.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Runs the gdn_scaled_dot_kkt kernel for GatedDeltaNet attention.
 *
 * Computes per chunk: A = KK^T · coeff · mask, where
 *   KK^T[i,j]    = K[i] · K[j]^T                           (Cube GEMM)
 *   coeff[i,j]   = exp(min(g[i]+log(β[i]) - g[j], 0))      (Vec gating)
 *   A[i,j]       = KK^T[i,j] · coeff[i,j] · mask[i,j]
 *
 * Tensor layouts:
 *   K    — [total_tokens, Hg, D]  fp16  (key vectors, BSND; stride Hg*D)
 *   Beta — [H, total_tokens]      fp16  (gate bias per value head,
 * pre-transposed) G    — [H, total_tokens]      fp32  (cumulative gate sum per
 * value head) Msk  — [C, C]                 fp32  (lower-triangular causal
 * mask, C = GDN_C)
 *
 * @param K          fp16 key tensor          [total_tokens, Hg, D]
 * @param Beta       fp16 gate bias tensor    [H, total_tokens]
 * @param G          fp32 gate sum tensor     [H, total_tokens]
 * @param Msk        fp32 causal mask         [C, C]
 * @param batch_size Number of sequences in the batch.
 * @param seq_len    Tokens per sequence (fixed-length path).
 *                   Ignored when cu_seqlens is provided.
 * @param cu_seqlens Optional int32 cumulative sequence lengths [batch_size+1].
 *                   Pass at::zeros({1}) (default) to use the fixed-length path.
 * @return at::Tensor fp16 output A [total_tokens, H, C] (BSND layout).
 */
at::Tensor run_gdn_scaled_dot_kkt(
    const at::Tensor& K, const at::Tensor& Beta, const at::Tensor& G,
    const at::Tensor& Msk, int64_t batch_size, int64_t seq_len,
    const at::Tensor& cu_seqlens = at::zeros({1})) {
  TORCH_CHECK(K.device().type() == DEVICE_TYPE,
              "gdn_scaled_dot_kkt: tensors must be on NPU, got ", K.device());
  TORCH_CHECK(K.scalar_type() == at::kHalf,
              "gdn_scaled_dot_kkt: K must be fp16, got ", K.scalar_type());
  TORCH_CHECK(Beta.scalar_type() == at::kHalf,
              "gdn_scaled_dot_kkt: Beta must be fp16, got ",
              Beta.scalar_type());
  TORCH_CHECK(G.scalar_type() == at::kFloat,
              "gdn_scaled_dot_kkt: G must be fp32, got ", G.scalar_type());
  TORCH_CHECK(Msk.scalar_type() == at::kFloat,
              "gdn_scaled_dot_kkt: Msk must be fp32, got ", Msk.scalar_type());
  TORCH_CHECK(K.is_contiguous(), "gdn_scaled_dot_kkt: K must be contiguous");
  TORCH_CHECK(Beta.is_contiguous(),
              "gdn_scaled_dot_kkt: Beta must be contiguous");
  TORCH_CHECK(G.is_contiguous(), "gdn_scaled_dot_kkt: G must be contiguous");
  TORCH_CHECK(Msk.is_contiguous(),
              "gdn_scaled_dot_kkt: Msk must be contiguous");

  // Derive shapes: G is [H, total_tokens], Msk is [C, C]
  const int64_t num_heads = G.size(0);
  const int64_t total_tokens = G.size(1);
  const int64_t chunk_c = Msk.size(0);  // GDN_C

  // total_work = batch_size * H (one work item per sequence × head pair)
  const int64_t total_work = batch_size * num_heads;
  uint32_t block_dim = GetNumCubeCores();
  if (static_cast<int64_t>(block_dim) > total_work) {
    block_dim = static_cast<uint32_t>(total_work);
  }

  // Workspace: 2 slots per core for double-buffering KK^T (fp16, [C×C] each)
  const at::Tensor workspace = at::empty(
      {static_cast<int64_t>(block_dim) * 2, chunk_c, chunk_c}, K.options());

  // Output A: [total_tokens, H, C] fp16 in BSND layout
  at::Tensor A = at::empty({total_tokens, num_heads, chunk_c}, K.options());

  // Optional cu_seqlens pointer (nullptr for the fixed-length path)
  uint8_t* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = static_cast<uint8_t*>(ConvertType(cu_seqlens));
  }

  EXEC_KERNEL_CMD(gdn_scaled_dot_kkt, block_dim, K, Beta, G, Msk, workspace, A,
                  cu_seqlens_ptr, batch_size, seq_len, total_tokens);

  return A;
}

}  // namespace pto_isa_ops
