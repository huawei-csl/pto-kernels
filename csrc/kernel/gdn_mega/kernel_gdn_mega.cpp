// GDN mega kernel: one launch, two global phases, one production schedule.
// Phase 1: gate prefix scan/layout and shared KK/QK products run concurrently.
// Phase 2: a head-persistent pipeline overlaps next-chunk inverse preparation
// with residual, state and output products for every input.
// See csrc/host/torch_gdn_mega.h for the caller workspace contract.
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

#include <pto/pto-inst.hpp>

#include "acl/acl.h"

// The legacy ccec compiler does not provide the ASC task-type annotation.
#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(kernel_type) ((void)0)
#endif
using namespace pto;

#include "common.hpp"
#include "fused_pipeline.hpp"
#include "gate_scan.hpp"
#include "inverse.hpp"
#include "recurrence_pipeline.hpp"
#include "shared_grams.hpp"

template <int32_t H>
AICORE inline void mega_kernel_impl(
    GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr,
    GM_ADDR beta_ptr, GM_ADDR msk_lower_ptr, GM_ADDR msk_full_ptr,
    GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr,
    GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr, GM_ADDR beta_t_ptr, GM_ADDR A_ptr,
    GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr, GM_ADDR w_ptr, GM_ADDR u_ptr,
    GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr, GM_ADDR h0_ptr,
    int64_t has_initial_state, GM_ADDR kkt_ws_ptr, GM_ADDR wy_ws_a1_ptr,
    GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr, GM_ADDR o_ws_qk_ptr,
    GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr, uint32_t num_key_heads,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint32_t num_matrices) {
  constexpr int32_t D = 128, C = 128;
  if (num_key_heads == 0 || (static_cast<uint32_t>(H) % num_key_heads) != 0) {
    return;
  }

  // Count packed 128-token chunks, including independent short tails.
  auto cu = reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr);
  uint32_t internal_matrices = 0;
  for (int64_t seq = 0; seq < batch_size; ++seq)
    internal_matrices += (cu[seq + 1] - cu[seq] + C - 1) / C * H;

  // Phase 1: vector gate scan/layout overlaps cube KK/QK products.
  mega_gate_scan<H, C>(reinterpret_cast<__gm__ float*>(g_in_ptr),
                       reinterpret_cast<__gm__ half*>(beta_ptr),
                       reinterpret_cast<__gm__ float*>(g_sum_ptr),
                       reinterpret_cast<__gm__ float*>(g_t_ptr),
                       reinterpret_cast<__gm__ half*>(beta_t_ptr),
                       reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr),
                       batch_size, total_tokens);

  mega_shared_grams<H, C, D>(reinterpret_cast<__gm__ half*>(k_ptr),
                             reinterpret_cast<__gm__ float*>(v_new_ptr),
                             reinterpret_cast<__gm__ float*>(s_ptr),
                             reinterpret_cast<__gm__ int32_t*>(cu_seqlens_ptr),
                             internal_matrices, num_key_heads,
                             reinterpret_cast<__gm__ half*>(q_ptr),
                             reinterpret_cast<__gm__ float*>(A_inv_f32_ptr));

  mega_sync_all();

  mega_fused_pipeline<H, C, D>(
      reinterpret_cast<__gm__ half*>(q_ptr),
      reinterpret_cast<__gm__ half*>(k_ptr),
      reinterpret_cast<__gm__ half*>(v_ptr),
      reinterpret_cast<__gm__ float*>(g_t_ptr),
      reinterpret_cast<__gm__ float*>(g_sum_ptr),
      reinterpret_cast<__gm__ half*>(beta_t_ptr),
      reinterpret_cast<__gm__ half*>(A_ptr),
      reinterpret_cast<__gm__ half*>(w_ptr),
      reinterpret_cast<__gm__ half*>(A_inv_ptr),
      reinterpret_cast<__gm__ half*>(u_ptr),
      reinterpret_cast<__gm__ half*>(o_ws_qs_ptr),
      reinterpret_cast<__gm__ half*>(o_ws_gated_ptr),
      reinterpret_cast<__gm__ half*>(kkt_ws_ptr),
      reinterpret_cast<__gm__ half*>(h_ws_ptr),
      reinterpret_cast<__gm__ half*>(o_ws_qk_ptr),
      reinterpret_cast<__gm__ half*>(o_ptr),
      reinterpret_cast<__gm__ half*>(fs_ptr),
      reinterpret_cast<__gm__ half*>(h0_ptr), has_initial_state, cu, batch_size,
      num_key_heads, total_tokens, reinterpret_cast<__gm__ float*>(v_new_ptr),
      reinterpret_cast<__gm__ float*>(s_ptr),
      reinterpret_cast<__gm__ float*>(A_inv_f32_ptr),
      reinterpret_cast<__gm__ float*>(msk_lower_ptr),
      reinterpret_cast<__gm__ float*>(msk_full_ptr));
  pipe_barrier(PIPE_ALL);
}

// Note the codegen parser does not support arguments of form "type *name", only
// "type* name"
template <int H>
__global__ AICORE void gdn_mega_specialized(
    GM_ADDR q_ptr, GM_ADDR k_ptr, GM_ADDR v_ptr, GM_ADDR g_in_ptr,
    GM_ADDR beta_ptr, GM_ADDR msk_lower_ptr, GM_ADDR msk_full_ptr,
    GM_ADDR minus_id_ptr, GM_ADDR cu_seqlens_ptr, GM_ADDR o_ptr,
    GM_ADDR g_sum_ptr, GM_ADDR g_t_ptr, GM_ADDR beta_t_ptr, GM_ADDR A_ptr,
    GM_ADDR A_inv_f32_ptr, GM_ADDR A_inv_ptr, GM_ADDR w_ptr, GM_ADDR u_ptr,
    GM_ADDR s_ptr, GM_ADDR v_new_ptr, GM_ADDR fs_ptr, GM_ADDR h0_ptr,
    int64_t has_initial_state, GM_ADDR kkt_ws_ptr, GM_ADDR wy_ws_a1_ptr,
    GM_ADDR wy_ws_a2_ptr, GM_ADDR h_ws_ptr, GM_ADDR o_ws_qk_ptr,
    GM_ADDR o_ws_qs_ptr, GM_ADDR o_ws_gated_ptr, uint32_t num_heads,
    uint32_t num_key_heads, int64_t batch_size, int64_t seq_len,
    int64_t total_tokens, uint32_t num_matrices) {
  // The fused pipeline runs chunk_h / chunk_o style mix stages and relies on
  // FFTS cross-core sync, so the launch must be marked as a mix kernel. The
  // staged GDN kernels in csrc/kernel/ use the same task type. Without this
  // the ASC toolchain warns that the type cannot be derived automatically.
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  mega_kernel_impl<H>(
      q_ptr, k_ptr, v_ptr, g_in_ptr, beta_ptr, msk_lower_ptr, msk_full_ptr,
      minus_id_ptr, cu_seqlens_ptr, o_ptr, g_sum_ptr, g_t_ptr, beta_t_ptr,
      A_ptr, A_inv_f32_ptr, A_inv_ptr, w_ptr, u_ptr, s_ptr, v_new_ptr, fs_ptr,
      h0_ptr, has_initial_state, kkt_ws_ptr, wy_ws_a1_ptr, wy_ws_a2_ptr,
      h_ws_ptr, o_ws_qk_ptr, o_ws_qs_ptr, o_ws_gated_ptr, num_key_heads,
      batch_size, seq_len, total_tokens, num_matrices);
}

// Host-callable launch shim: the `<<<>>>` syntax is only understood by the
// kernel compiler, so the launch lives here rather than in the host wrapper
// under csrc/host/.
extern "C" void pto_launch_gdn_mega(
    uint32_t blockDim, void* stream, void* q, void* k, void* v, void* g_in,
    void* beta, void* msk_lower, void* msk_full, void* minus_id,
    void* cu_seqlens, void* o, void* g_sum, void* g_t, void* beta_t, void* A,
    void* A_inv_f32, void* A_inv, void* w, void* u, void* s, void* v_new,
    void* fs, void* h0, int64_t has_initial_state, void* kkt_ws, void* wy_ws_a1,
    void* wy_ws_a2, void* h_ws, void* o_ws_qk, void* o_ws_qs, void* o_ws_gated,
    uint32_t num_heads, uint32_t num_key_heads, int64_t batch_size,
    int64_t seq_len, int64_t total_tokens, uint32_t num_matrices) {
#define LAUNCH_MEGA(H)                                                      \
  gdn_mega_specialized<H><<<blockDim, nullptr, stream>>>(                   \
      (GM_ADDR)q, (GM_ADDR)k, (GM_ADDR)v, (GM_ADDR)g_in, (GM_ADDR)beta,     \
      (GM_ADDR)msk_lower, (GM_ADDR)msk_full, (GM_ADDR)minus_id,             \
      (GM_ADDR)cu_seqlens, (GM_ADDR)o, (GM_ADDR)g_sum, (GM_ADDR)g_t,        \
      (GM_ADDR)beta_t, (GM_ADDR)A, (GM_ADDR)A_inv_f32, (GM_ADDR)A_inv,      \
      (GM_ADDR)w, (GM_ADDR)u, (GM_ADDR)s, (GM_ADDR)v_new, (GM_ADDR)fs,      \
      (GM_ADDR)h0, has_initial_state, (GM_ADDR)kkt_ws, (GM_ADDR)wy_ws_a1,   \
      (GM_ADDR)wy_ws_a2, (GM_ADDR)h_ws, (GM_ADDR)o_ws_qk, (GM_ADDR)o_ws_qs, \
      (GM_ADDR)o_ws_gated, num_heads, num_key_heads, batch_size, seq_len,   \
      total_tokens, num_matrices);

  switch (num_heads) {
    case 8:
      LAUNCH_MEGA(8);
      break;
    case 12:
      LAUNCH_MEGA(12);
      break;
    case 16:
      LAUNCH_MEGA(16);
      break;
    case 24:
      LAUNCH_MEGA(24);
      break;
    case 32:
      LAUNCH_MEGA(32);
      break;
    case 48:
      LAUNCH_MEGA(48);
      break;
    case 64:
      LAUNCH_MEGA(64);
      break;
    default:
      break;
  }
#undef LAUNCH_MEGA
}
