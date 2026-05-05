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

#include <vector>

#include "aclrtlaunch_gdn_wy_fast.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Runs the gdn_wy_fast kernel for GatedDeltaNet WY-representation.
 *
 * Computes per chunk:
 *   A2 = A * beta_2d                     (beta broadcast along columns)
 *   A1 = A * (exp(g) * beta)_2d          (gate+beta broadcast along columns)
 *   U  = A2 @ V                           (beta-scaled branch)
 *   W  = A1 @ K                           (gate+beta-scaled branch)
 *
 * Tensor layouts (fp16 unless noted):
 *   K    — [total_tokens, Hg, D]  (key vectors, BSND; stride Hg*D)
 *   V    — [total_tokens, H,  D]  (value vectors, BSND; stride H*D)
 *   Beta — [H, total_tokens]      (decay factor per value head, pre-transposed)
 *   G    — [H, total_tokens] fp32 (gate values per value head, pre-transposed)
 *   A    — [total_tokens, H, C]   (attention matrix from kkt kernel, BSND;
 * stride H*C)
 *
 * @param K          fp16 key tensor          [total_tokens, Hg, D]
 * @param V          fp16 value tensor        [total_tokens, H,  D]
 * @param Beta       fp16 decay factor        [H, total_tokens]
 * @param G          fp32 gate values         [H, total_tokens]
 * @param A          fp16 attention matrix    [total_tokens, H, C]
 * @param batch_size Number of sequences in the batch.
 * @param seq_len    Tokens per sequence (fixed-length path).
 *                   Ignored when cu_seqlens is provided.
 * @param cu_seqlens Optional int32 cumulative sequence lengths [batch_size+1].
 *                   Pass at::zeros({1}) (default) to use the fixed-length path.
 * @return std::vector<at::Tensor> {W, U}, both fp16 [total_tokens, H, D].
 */
std::vector<at::Tensor> run_gdn_wy_fast(
    const at::Tensor& K, const at::Tensor& V, const at::Tensor& Beta,
    const at::Tensor& G, const at::Tensor& A, int64_t batch_size,
    int64_t seq_len, const at::Tensor& cu_seqlens = at::zeros({1})) {
  TORCH_CHECK(K.device().type() == DEVICE_TYPE,
              "gdn_wy_fast: tensors must be on NPU, got ", K.device());
  TORCH_CHECK(K.scalar_type() == at::kHalf, "gdn_wy_fast: K must be fp16, got ",
              K.scalar_type());
  TORCH_CHECK(V.scalar_type() == at::kHalf, "gdn_wy_fast: V must be fp16, got ",
              V.scalar_type());
  TORCH_CHECK(Beta.scalar_type() == at::kHalf,
              "gdn_wy_fast: Beta must be fp16, got ", Beta.scalar_type());
  TORCH_CHECK(G.scalar_type() == at::kFloat,
              "gdn_wy_fast: G must be fp32, got ", G.scalar_type());
  TORCH_CHECK(A.scalar_type() == at::kHalf, "gdn_wy_fast: A must be fp16, got ",
              A.scalar_type());
  TORCH_CHECK(K.is_contiguous(), "gdn_wy_fast: K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "gdn_wy_fast: V must be contiguous");
  TORCH_CHECK(Beta.is_contiguous(), "gdn_wy_fast: Beta must be contiguous");
  TORCH_CHECK(G.is_contiguous(), "gdn_wy_fast: G must be contiguous");
  TORCH_CHECK(A.is_contiguous(), "gdn_wy_fast: A must be contiguous");

  // Derive shapes: G is [H, total_tokens], A is [total_tokens, H, C]
  const int64_t num_heads = G.size(0);
  const int64_t total_tokens = G.size(1);
  const int64_t chunk_c = A.size(-1);  // GDN_C

  // total_work = batch_size * chunks_per_seq * H (fixed-length)
  uint32_t block_dim = GetNumCubeCores();
  if (cu_seqlens.numel() == 1) {
    const int64_t chunks_per_seq = (seq_len + chunk_c - 1) / chunk_c;
    const int64_t total_work = batch_size * chunks_per_seq * num_heads;
    if (static_cast<int64_t>(block_dim) > total_work) {
      block_dim = static_cast<uint32_t>(total_work);
    }
  }

  // Per-core workspaces (fp16):
  //   workspace_a1: [block_dim, C, C] — A1 = A*(exp(g)*beta)_2d
  //   workspace_a2: [block_dim, C, C] — A2 = A*beta_2d
  const auto ws_opts = K.options();
  const at::Tensor workspace_a1 =
      at::empty({static_cast<int64_t>(block_dim), chunk_c, chunk_c}, ws_opts);
  const at::Tensor workspace_a2 =
      at::empty({static_cast<int64_t>(block_dim), chunk_c, chunk_c}, ws_opts);

  // Outputs W and U share the same shape as V: [total_tokens, H, D]
  const at::Tensor W = at::empty_like(V);
  at::Tensor U = at::empty_like(V);

  // Optional cu_seqlens pointer (nullptr for the fixed-length path)
  uint8_t* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = static_cast<uint8_t*>(ConvertType(cu_seqlens));
  }

  EXEC_KERNEL_CMD(gdn_wy_fast, block_dim, K, V, Beta, G, A, workspace_a1,
                  workspace_a2, W, U, cu_seqlens_ptr, batch_size, seq_len,
                  total_tokens);

  return {W, U};
}

}  // namespace pto_isa_ops
