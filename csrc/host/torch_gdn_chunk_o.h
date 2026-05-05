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

#include "aclrtlaunch_gdn_chunk_o.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Runs the chunk_o kernel for GatedDeltaNet output computation.
 *
 * Computes per chunk: O = (QK_gated @ V) + exp(g) * (Q @ S)
 *
 * Tensor layouts (BSND, fp16 unless noted):
 *   Q, K — [total_tokens, Hg, D]  (Hg = grouped-query heads = GDN_HG)
 *   V    — [total_tokens, H,  D]  (H  = value heads         = GDN_H)
 *   S    — [num_chunks * H, D, D] (accumulated hidden states)
 *   G    — [H, total_tokens]      fp32, cumulative gate values (pre-transposed)
 *   Msk  — [C, C]                 fp32, lower-triangular causal mask (C =
 * GDN_C)
 *
 * @param Q          fp16 query tensor  [total_tokens, Hg, D]
 * @param K          fp16 key tensor    [total_tokens, Hg, D]
 * @param V          fp16 value tensor  [total_tokens, H,  D]
 * @param S          fp16 state tensor  [num_chunks * H, D, D]
 * @param G          fp32 gate tensor   [H, total_tokens]
 * @param Msk        fp32 causal mask   [C, C]
 * @param batch_size Number of sequences in the batch.
 * @param seq_len    Sequence length per batch (fixed-length path).
 *                   Ignored when cu_seqlens is provided.
 * @param cu_seqlens Optional int32 tensor of cumulative sequence lengths
 *                   [batch_size + 1] for variable-length sequences.
 *                   Pass at::zeros({1}) (default) to use the fixed-length path.
 * @return at::Tensor fp16 output tensor with the same shape as V.
 */
at::Tensor run_gdn_chunk_o(const at::Tensor& Q, const at::Tensor& K,
                           const at::Tensor& V, const at::Tensor& S,
                           const at::Tensor& G, const at::Tensor& Msk,
                           int64_t batch_size, int64_t seq_len,
                           const at::Tensor& cu_seqlens = at::zeros({1})) {
  TORCH_CHECK(Q.device().type() == DEVICE_TYPE,
              "chunk_o: tensors must be on NPU, got ", Q.device());
  TORCH_CHECK(Q.scalar_type() == at::kHalf,
              "chunk_o: Q, K, V, S must be fp16, got ", Q.scalar_type());
  TORCH_CHECK(K.scalar_type() == at::kHalf, "chunk_o: K must be fp16, got ",
              K.scalar_type());
  TORCH_CHECK(V.scalar_type() == at::kHalf, "chunk_o: V must be fp16, got ",
              V.scalar_type());
  TORCH_CHECK(S.scalar_type() == at::kHalf, "chunk_o: S must be fp16, got ",
              S.scalar_type());
  TORCH_CHECK(G.scalar_type() == at::kFloat, "chunk_o: G must be fp32, got ",
              G.scalar_type());
  TORCH_CHECK(Msk.scalar_type() == at::kFloat,
              "chunk_o: Msk must be fp32, got ", Msk.scalar_type());
  TORCH_CHECK(Q.is_contiguous(), "chunk_o: Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "chunk_o: K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "chunk_o: V must be contiguous");
  TORCH_CHECK(S.is_contiguous(), "chunk_o: S must be contiguous");
  TORCH_CHECK(G.is_contiguous(), "chunk_o: G must be contiguous");
  TORCH_CHECK(Msk.is_contiguous(), "chunk_o: Msk must be contiguous");

  // Derive shapes from tensors: G is [H, total_tokens], V is [total_tokens, H,
  // D]
  const int64_t num_heads = G.size(0);
  const int64_t total_tokens = G.size(1);
  const int64_t chunk_c = Msk.size(0);  // GDN_C
  const int64_t hidden_d = V.size(-1);  // GDN_D

  // Compute block_dim (number of AI cores), capped at available work.
  int64_t block_dim = static_cast<int64_t>(GetNumCubeCores());
  if (cu_seqlens.numel() == 1) {
    // Fixed-length path: total work = batch * chunks_per_seq * H
    const int64_t chunks_per_seq = (seq_len + chunk_c - 1) / chunk_c;
    const int64_t total_work = batch_size * chunks_per_seq * num_heads;
    if (block_dim > total_work) {
      block_dim = total_work;
    }
  }

  // Allocate per-core workspace tensors (fp16).
  //   workspace_qk:      [block_dim, C, C]  — GEMM1 result QK
  //   workspace_qs_qkv:  [block_dim, C, D]  — shared buffer for QS (GEMM2) and
  //   QKV (GEMM3) workspace_qk_gated:[block_dim, C, C]  — gated QK written back
  //   by Vec for GEMM3
  const auto ws_opts = V.options();
  at::Tensor workspace_qk = at::empty({block_dim, chunk_c, chunk_c}, ws_opts);
  at::Tensor workspace_qs_qkv =
      at::empty({block_dim, chunk_c, hidden_d}, ws_opts);
  at::Tensor workspace_qk_gated =
      at::empty({block_dim, chunk_c, chunk_c}, ws_opts);

  // Output tensor: same shape and dtype as V.
  const at::Tensor tensor_out = at::empty_like(V);

  // Optional cu_seqlens pointer (nullptr for the fixed-length path).
  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  EXEC_KERNEL_CMD(gdn_chunk_o, block_dim, Q, K, V, S, G, Msk, workspace_qk,
                  workspace_qs_qkv, workspace_qk_gated, tensor_out,
                  cu_seqlens_ptr, batch_size, seq_len, total_tokens);

  return tensor_out;
}

}  // namespace pto_isa_ops
