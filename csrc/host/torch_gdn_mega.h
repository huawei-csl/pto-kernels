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

#include <cstdint>
#include <limits>

#include "utils.h"

extern "C" void pto_launch_gdn_mega(
    uint32_t blockDim, void* stream, void* q, void* k, void* v, void* g_in,
    void* beta, void* msk_lower, void* msk_full, void* minus_id,
    void* cu_seqlens, void* o, void* g_sum, void* g_t, void* beta_t, void* A,
    void* A_inv_f32, void* A_inv, void* w, void* u, void* s, void* v_new,
    void* fs, void* h0, int64_t has_initial_state, void* kkt_ws, void* wy_ws_a1,
    void* wy_ws_a2, void* h_ws, void* o_ws_qk, void* o_ws_qs, void* o_ws_gated,
    uint32_t num_heads, uint32_t num_key_heads, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, uint32_t num_matrices);

namespace pto_isa_ops {

namespace {

constexpr int64_t kGdnMegaHeadDim = 128;

constexpr bool gdn_mega_is_supported_head_pair(int64_t value_heads,
                                               int64_t key_heads) {
  return key_heads > 0 && value_heads % key_heads == 0 &&
         (value_heads == 8 || value_heads == 12 || value_heads == 16 ||
          value_heads == 24 || value_heads == 32 || value_heads == 48 ||
          value_heads == 64);
}

void gdn_mega_check_shape(const at::Tensor& q, const at::Tensor& k,
                          const at::Tensor& v, const at::Tensor& g,
                          const at::Tensor& beta, const at::Tensor& cu_seqlens,
                          const at::Tensor& initial_state,
                          bool has_initial_state) {
  auto check = [](bool condition, const char* message) {
    TORCH_CHECK(condition, "gdn_mega: ", message);
  };

  check(q.device().type() == DEVICE_TYPE, "q must be on NPU");

  check(q.dim() == 4, "q must have shape [B, T, Hg, D]");
  check(k.dim() == 4, "k must have shape [B, T, Hg, D]");
  check(v.dim() == 4, "v must have shape [B, T, H, D]");
  check(g.dim() == 3, "g must have shape [B, T, H]");
  check(beta.dim() == 3, "beta must have shape [B, T, H]");

  check(q.size(0) == 1, "only packed B=1 input is supported");
  check(q.sizes() == k.sizes(), "q and k must have the same shape");
  check(q.size(1) == v.size(1), "q/k and v sequence lengths must match");
  check(gdn_mega_is_supported_head_pair(v.size(2), q.size(2)),
        "unsupported (NumValueHeads, NumKeyHeads) pair");
  check(q.size(3) == kGdnMegaHeadDim && v.size(3) == kGdnMegaHeadDim,
        "only head dimension 128 is supported");

  check(g.size(0) == 1 && beta.size(0) == 1,
        "g and beta must use packed B=1 layout");
  check(g.size(1) == q.size(1) && beta.size(1) == q.size(1),
        "g/beta sequence lengths must match q");
  check(g.size(2) == v.size(2) && beta.size(2) == v.size(2),
        "g/beta must use the same NumValueHeads as v");

  check(q.scalar_type() == at::kHalf, "q must be float16");
  check(k.scalar_type() == at::kHalf, "k must be float16");
  check(v.scalar_type() == at::kHalf, "v must be float16");
  check(beta.scalar_type() == at::kHalf, "beta must be float16");
  check(g.scalar_type() == at::kFloat, "g must be float32");
  check(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");

  check(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
        "q, k, and v must be contiguous");
  check(g.is_contiguous() && beta.is_contiguous() && cu_seqlens.is_contiguous(),
        "g, beta, and cu_seqlens must be contiguous");

  if (has_initial_state) {
    check(initial_state.dim() == 4,
          "initial_state must have shape [N, H, D, D]");
    check(initial_state.size(0) == cu_seqlens.numel() - 1,
          "initial_state.size(0) must match the cu_seqlens sequence count");
    check(initial_state.size(1) == v.size(2),
          "initial_state.size(1) must match NumValueHeads");
    check(initial_state.size(2) == kGdnMegaHeadDim &&
              initial_state.size(3) == kGdnMegaHeadDim,
          "initial_state must use head dimensions 128 x 128");
    check(initial_state.scalar_type() == at::kHalf,
          "initial_state must be float16");
    check(initial_state.is_contiguous(), "initial_state must be contiguous");
  }
}

}  // namespace

/**
 * @brief Fused chunk GatedDeltaNet forward pass (GDN mega kernel).
 *
 * Runs the whole chunk-GDN pipeline in a single kernel launch:
 *   cumsum -> transpose -> kkt -> solve_tril -> wy_fast -> chunk_h -> chunk_o
 *
 * Value heads (H) and key heads (Hg) are passed at runtime; H dispatches to a
 * finite set of compile-time specialisations (8, 12, 16, 24, 32, 48, 64).
 * Head dimension and chunk size are fixed at 128 by the kernel build.
 *
 * This entry point is caller-allocates: every output, intermediate and
 * per-core workspace is supplied by the caller and written in place. See the
 * Python wrapper for the required shapes.
 *
 * @param [in]     q               Queries [1, T, Hg, D] fp16.
 * @param [in]     k               Keys [1, T, Hg, D] fp16.
 * @param [in]     v               Values [1, T, H, D] fp16.
 * @param [in]     g               Gates [1, T, H] fp32.
 * @param [in]     beta            Beta [1, T, H] fp16.
 * @param [in]     mask_lower      Strictly-lower chunk mask [C, C] fp32.
 * @param [in]     mask_full       Full chunk mask [C, C] fp32.
 * @param [in]     minus_identity  Negated identity [C, C] fp16.
 * @param [in]     cu_seqlens      Cumulative sequence lengths [N+1] int32.
 * @param [in,out] out             Output [1, T, H, D] fp16.
 * @param [in,out] g_sum           Cumulative gates [1, T, H] fp32.
 * @param [in,out] g_t             Transposed gates [H, T] fp32.
 * @param [in,out] beta_t          Transposed beta [H, T] fp16.
 * @param [in,out] a               K K^T [1, T, H, C] fp16.
 * @param [in,out] a_inv_f32       Inverse of (I - A) [1, T, H, C] fp32.
 * @param [in,out] a_inv           Inverse of (I - A) [1, T, H, C] fp16.
 * @param [in,out] w               wy_fast W [1, T, H, D] fp16.
 * @param [in,out] u               wy_fast U [1, T, H, D] fp16.
 * @param [in,out] s               Per-chunk states [num_chunks*H, D, D] fp16.
 * @param [in,out] v_new           Residual-corrected values [1, T, H, D] fp16.
 * @param [in,out] final_state     Final states [N*H, D, D] fp16.
 * @param [in]     initial_state   Initial states [N, H, D, D] fp16. Ignored
 *                                 when has_initial_state is false; pass any
 *                                 correctly typed tensor in that case.
 * @param [in]     has_initial_state Whether initial_state carries real data.
 * @param [in,out] kkt_workspace   [block_dim*2, C, C] fp16.
 * @param [in,out] wy_workspace_a1 [block_dim, C, C] fp16.
 * @param [in,out] wy_workspace_a2 [block_dim, C, C] fp16.
 * @param [in,out] h_workspace     [block_dim*4, D, D] fp16.
 * @param [in,out] o_workspace_qk  [block_dim, C, C] fp16.
 * @param [in,out] o_workspace_qs  [block_dim, C, D] fp16.
 * @param [in,out] o_workspace_gated [block_dim, C, C] fp16.
 * @param [in]     block_dim       Launch block dimension (cube core count).
 * @param [in]     batch_size      Number of sequences (cu_seqlens.numel()-1).
 * @param [in]     seq_len         Packed token count.
 * @param [in]     total_tokens    Packed token count.
 * @param [in]     num_matrices    num_chunks * H.
 */
void run_gdn_mega(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                  const at::Tensor& g, const at::Tensor& beta,
                  const at::Tensor& mask_lower, const at::Tensor& mask_full,
                  const at::Tensor& minus_identity,
                  const at::Tensor& cu_seqlens, at::Tensor& out,
                  at::Tensor& g_sum, at::Tensor& g_t, at::Tensor& beta_t,
                  at::Tensor& a, at::Tensor& a_inv_f32, at::Tensor& a_inv,
                  at::Tensor& w, at::Tensor& u, at::Tensor& s,
                  at::Tensor& v_new, at::Tensor& final_state,
                  const at::Tensor& initial_state, bool has_initial_state,
                  at::Tensor& kkt_workspace, at::Tensor& wy_workspace_a1,
                  at::Tensor& wy_workspace_a2, at::Tensor& h_workspace,
                  at::Tensor& o_workspace_qk, at::Tensor& o_workspace_qs,
                  at::Tensor& o_workspace_gated, int64_t block_dim,
                  int64_t batch_size, int64_t seq_len, int64_t total_tokens,
                  int64_t num_matrices) {
  gdn_mega_check_shape(q, k, v, g, beta, cu_seqlens, initial_state,
                       has_initial_state);

  TORCH_CHECK(
      block_dim > 0 && block_dim <= std::numeric_limits<uint32_t>::max(),
      "gdn_mega: block_dim is out of uint32 range");
  TORCH_CHECK(batch_size == cu_seqlens.numel() - 1,
              "gdn_mega: batch_size must match cu_seqlens");
  TORCH_CHECK(seq_len == q.size(1), "gdn_mega: seq_len must match q.shape[1]");
  TORCH_CHECK(total_tokens == q.size(1),
              "gdn_mega: total_tokens must match the packed token count");
  TORCH_CHECK(
      num_matrices >= 0 && num_matrices <= std::numeric_limits<uint32_t>::max(),
      "gdn_mega: num_matrices is out of uint32 range");

  const uint32_t block_dim_u32 = static_cast<uint32_t>(block_dim);
  const uint32_t num_heads_u32 = static_cast<uint32_t>(v.size(2));
  const uint32_t num_key_heads_u32 = static_cast<uint32_t>(q.size(2));
  const uint32_t num_matrices_u32 = static_cast<uint32_t>(num_matrices);
  const int64_t has_initial_state_i64 = has_initial_state ? 1 : 0;

  EXEC_KERNEL_CMD(
      gdn_mega, block_dim_u32, q, k, v, g, beta, mask_lower, mask_full,
      minus_identity, cu_seqlens, out, g_sum, g_t, beta_t, a, a_inv_f32, a_inv,
      w, u, s, v_new, final_state, initial_state, has_initial_state_i64,
      kkt_workspace, wy_workspace_a1, wy_workspace_a2, h_workspace,
      o_workspace_qk, o_workspace_qs, o_workspace_gated, num_heads_u32,
      num_key_heads_u32, batch_size, seq_len, total_tokens, num_matrices_u32);
}

}  // namespace pto_isa_ops
