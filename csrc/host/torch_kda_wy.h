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

#include "utils.h"

// Declaration of the launch shim defined alongside the kernel in
// csrc/kernel/. It wraps the `<<<>>>` launch so that this host code can stay
// plain C++ and build under either kernel toolchain (see USE_ASC_LANGUAGE).
extern "C" {

void pto_launch_kda_wy(uint32_t blockDim, void* stream, void* K_handle,
                       void* V_handle, void* Beta_handle, void* G_handle,
                       void* A_handle, void* workspace_a2_handle,
                       void* workspace_keff_handle, void* U_handle,
                       void* W_handle, void* cu_seqlens, int64_t batch_size,
                       int64_t seq_len, int64_t total_tokens);

}  // extern "C"

namespace pto_isa_ops {

/**
 * @brief Runs the wy_kda kernel for the KDA WY-representation.
 *
 * Computes per chunk:
 *   A2[r, c]    = INV[r, c] * beta[c]            (column-scale by beta)
 *   K_eff[c, d] = k[c, d] * exp(g_cs[c, d])      (element-wise per-dim gate)
 *   U = A2 @ V
 *   W = A2 @ K_eff
 *
 * The key difference from gdn_wy_fast is that the gate g_cs is per-dimension
 * (KDA) rather than scalar (GDN), so K_eff must be precomputed explicitly
 * instead of folding the gate into the column scale.  This allows Cube to
 * reuse A2 across both GEMMs without a second workspace pass.
 *
 * Tensor layouts (fp16 unless noted):
 *   K    — [H, total_tokens, D]      fp16  (key vectors, head-major)
 *   V    — [total_tokens, H, D]      fp16  (value vectors, BSND)
 *   G    — [H, total_tokens, D]      fp32  (per-dim cumulative gate sum,
 *                                           head-major)
 *   Beta — [H, total_tokens]         fp16  (decay factor, head-major)
 *   INV  — [total_tokens, H, C]      fp16  (chunk-local (I+L)^{-1}, BSND)
 *
 * @param K          fp16 key tensor              [H, total_tokens, D]
 * @param V          fp16 value tensor            [total_tokens, H, D]
 * @param G          fp32 per-dim gate cumsum     [H, total_tokens, D]
 * @param Beta       fp16 decay factor            [H, total_tokens]
 * @param INV        fp16 chunk-local inverse     [total_tokens, H, C]
 * @param batch_size Number of sequences in the batch.
 * @param seq_len    Tokens per sequence (fixed-length path).
 *                   Ignored when cu_seqlens is provided.
 * @param cu_seqlens Optional int32 cumulative sequence lengths [batch_size+1].
 *                   Pass at::zeros({1}) (default) to use the fixed-length path.
 * @return std::vector<at::Tensor> {U, W}, both fp16 [total_tokens, H, D].
 */
std::tuple<at::Tensor, at::Tensor> run_kda_wy(
    const at::Tensor& K, const at::Tensor& V, const at::Tensor& G,
    const at::Tensor& Beta, const at::Tensor& INV, int64_t batch_size,
    int64_t seq_len, const at::Tensor& cu_seqlens = at::zeros({1})) {
  TORCH_CHECK(K.device().type() == DEVICE_TYPE,
              "wy_kda: tensors must be on NPU, got ", K.device());
  TORCH_CHECK(K.scalar_type() == at::kHalf, "wy_kda: K must be fp16, got ",
              K.scalar_type());
  TORCH_CHECK(V.scalar_type() == at::kHalf, "wy_kda: V must be fp16, got ",
              V.scalar_type());
  TORCH_CHECK(G.scalar_type() == at::kFloat, "wy_kda: G must be fp32, got ",
              G.scalar_type());
  TORCH_CHECK(Beta.scalar_type() == at::kHalf,
              "wy_kda: Beta must be fp16, got ", Beta.scalar_type());
  TORCH_CHECK(INV.scalar_type() == at::kHalf, "wy_kda: INV must be fp16, got ",
              INV.scalar_type());
  TORCH_CHECK(K.is_contiguous(), "wy_kda: K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "wy_kda: V must be contiguous");
  TORCH_CHECK(G.is_contiguous(), "wy_kda: G must be contiguous");
  TORCH_CHECK(Beta.is_contiguous(), "wy_kda: Beta must be contiguous");
  TORCH_CHECK(INV.is_contiguous(), "wy_kda: INV must be contiguous");

  // K is [H, total_tokens, D] (head-major); derive shared shape scalars from K.
  const int64_t num_heads = K.size(0);
  const int64_t total_tokens = K.size(1);
  const int64_t hidden_size = K.size(2);  // D (== V_dim in the current build)
  const int64_t chunk_c = INV.size(-1);   // GDN_C

  // Cap block_dim to actual work items for the fixed-length path.
  uint32_t block_dim = GetNumCubeCores();
  if (cu_seqlens.numel() == 1) {
    const int64_t chunks_per_seq = (seq_len + chunk_c - 1) / chunk_c;
    const int64_t total_work = batch_size * chunks_per_seq * num_heads;
    if (static_cast<int64_t>(block_dim) > total_work) {
      block_dim = static_cast<uint32_t>(total_work);
    }
  }

  // Per-core workspaces (fp16):
  //   ws_a2:   [block_dim, C, C]   — A2 = INV * beta_2d
  //   ws_keff: [block_dim, C, D]   — K_eff = k * exp(g_cs)
  const auto ws_opts = K.options();
  const at::Tensor ws_a2 =
      at::empty({static_cast<int64_t>(block_dim), chunk_c, chunk_c}, ws_opts);
  const at::Tensor ws_keff = at::empty(
      {static_cast<int64_t>(block_dim), chunk_c, hidden_size}, ws_opts);

  // Outputs U and W: [total_tokens, H, D] fp16 BSND (same layout as V).
  const at::Tensor U = at::empty_like(V);
  const at::Tensor W =
      at::empty({total_tokens, num_heads, hidden_size}, V.options());

  // Optional cu_seqlens pointer (nullptr on the fixed-length path).
  void* cu_seqlens_ptr = nullptr;
  if (cu_seqlens.numel() != 1) {
    cu_seqlens_ptr = ConvertType(cu_seqlens);
  }

  // Kernel parameter order matches launch_wy_kda / call_kernel:
  //   K, V, Beta, G, INV, ws_a2, ws_keff, U, W, cu_seqlens, batch, seq, total
  EXEC_KERNEL_CMD(kda_wy, block_dim, K, V, Beta, G, INV, ws_a2, ws_keff, U, W,
                  cu_seqlens_ptr, batch_size, seq_len, total_tokens);

  return {U, W};
}

}  // namespace pto_isa_ops
