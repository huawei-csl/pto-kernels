#include "common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void chunk_h_kernel(
    __gm__ half *K_handle, __gm__ half *W_handle, __gm__ half *U_handle,
    __gm__ float *G_handle,
    __gm__ half *S_handle, __gm__ half *V_handle, __gm__ half *FS_handle,
    __gm__ half *workspace_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  set_ffts_base_addr(ffts_addr);

  constexpr int32_t D = HiddenSize;
  constexpr int32_t C = ChunkSize;
  constexpr int32_t H = NumHeads;
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t BSND_QKV_STRIDE = H * D;
  constexpr int32_t DD = D * D;

  constexpr int32_t WS_WS = 0;
  constexpr int32_t WS_K  = DD;
  constexpr int32_t WS_S  = DD * 2;
  constexpr int32_t WS_KV = DD * 3;
  constexpr int32_t WS_PER_CORE = DD * 4;

  chunk_gdn_pto::TileMatL1<half, D, D, D, D> s_l1;
  TASSIGN(s_l1, 0);
  chunk_gdn_pto::TileMatL1<half, C, D, C, D> w_l1;
  TASSIGN(w_l1, D * D * sizeof(half));
  TileAcc<float, C, D, C, D> ws_l0;
  TASSIGN(ws_l0, 0);
  chunk_gdn_pto::TileMatL1<half, D, C, D, C> k_l1;
  TASSIGN(k_l1, (DD + C * D) * sizeof(half));
  chunk_gdn_pto::TileMatL1<half, C, D, C, D> v_l1;
  TASSIGN(v_l1, (DD + C * D + D * C) * sizeof(half));
  TileAcc<float, D, D, D, D> kv_l0;
  TASSIGN(kv_l0, C * D * sizeof(float));

  constexpr int32_t G_BLOCK_UB = 0;
  constexpr int32_t G_BLOCK_SIZE = C * H * sizeof(float);
  constexpr int32_t ZERO_UB = G_BLOCK_SIZE;
  constexpr int32_t S_UB = ZERO_UB + 64 * sizeof(float);
  constexpr int32_t K_UB_HALF = S_UB + HalfC * D * sizeof(float);
  constexpr int32_t G_UB = K_UB_HALF + HalfC * D * sizeof(half);
  constexpr int32_t U_UB_HALF = G_UB + C * sizeof(float);
  constexpr int32_t K_UB = U_UB_HALF + HalfC * D * sizeof(half);
  constexpr int32_t G_V_UB = K_UB + HalfC * D * sizeof(float);
  constexpr int32_t COEFF_UB = G_V_UB + 64 * sizeof(float);
  constexpr int32_t U_UB = COEFF_UB + 64 * sizeof(float);
  constexpr int32_t WS_UB = U_UB + HalfC * D * sizeof(float);
  constexpr int32_t KV_UB = U_UB_HALF;
  constexpr int32_t S_UB_HALF = WS_UB + HalfC * D * sizeof(float);

  chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> zero_ub;
  TASSIGN(zero_ub, ZERO_UB);
  chunk_gdn_pto::TileUbDataND<float, HalfC, D, HalfC, D> s_ub;
  TASSIGN(s_ub, S_UB);
  chunk_gdn_pto::TileUbDataND<half, HalfC, D, HalfC, D> k_ub_half;
  TASSIGN(k_ub_half, K_UB_HALF);
  chunk_gdn_pto::TileUbDataND<float, 1, C, 1, C> g_ub;
  TASSIGN(g_ub, G_UB);
  chunk_gdn_pto::TileUbDataND<half, HalfC, D, HalfC, D> s_ub_half;
  TASSIGN(s_ub_half, S_UB_HALF);
  chunk_gdn_pto::TileUbDataND<half, HalfC, D, HalfC, D> u_ub_half;
  TASSIGN(u_ub_half, U_UB_HALF);
  chunk_gdn_pto::TileUbDataND<float, HalfC, D, HalfC, D> k_ub;
  TASSIGN(k_ub, K_UB);
  chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> g_v_ub;
  TASSIGN(g_v_ub, G_V_UB);
  chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> coeff_ub;
  TASSIGN(coeff_ub, COEFF_UB);
  chunk_gdn_pto::TileUbDataND<float, HalfC, D, HalfC, D> u_ub;
  TASSIGN(u_ub, U_UB);
  chunk_gdn_pto::TileUbDataND<float, HalfC, D, HalfC, D> ws_ub;
  TASSIGN(ws_ub, WS_UB);
  chunk_gdn_pto::TileUbDataND<float, HalfC, D, HalfC, D> kv_ub;
  TASSIGN(kv_ub, KV_UB);

  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * H;

#if defined(__DAV_C220_CUBE__)
  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;

      chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, D, D, 1, 1, 1, D, 1, D, D>(
          workspace_handle + ws_base + WS_S, 0, 0, D, D);

      int64_t w_offset = ((chunk_start) * H + head) * D;
      chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, C, D, 1, 1, 1, BSND_QKV_STRIDE, 1, C, D>(
          W_handle + w_offset, D * D * static_cast<int32_t>(sizeof(half)), 0,
          static_cast<int32_t>(valid), D);

      chunk_gdn_pto::gemm_v0<half, float, C, D, D, C, D, D, D, false, false>(w_l1, s_l1, ws_l0, (bool)1);

      chunk_gdn_pto::copy_l0c_to_gm<half, float, 1, 1, 1, C, D, 1, 1, 1, D, 1, C, D>(
          workspace_handle + ws_base + WS_WS, 0, 0, C, D);
      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(0, 2);

      chunk_gdn_pto::wait_cross_flag(1);

      chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, D, C, 1, 1, 1, C, 1, D, C>(
          workspace_handle + ws_base + WS_K, (DD + C * D) * static_cast<int32_t>(sizeof(half)), 0, D, C);

      int64_t v_offset = ((chunk_start) * H + head) * D;
      chunk_gdn_pto::copy_gm_to_l1<half, half, 1, 1, 1, C, D, 1, 1, 1, BSND_QKV_STRIDE, 1, C, D>(
          V_handle + v_offset, (DD + C * D + D * C) * static_cast<int32_t>(sizeof(half)), 0,
          static_cast<int32_t>(valid), D);

      chunk_gdn_pto::gemm_v0<half, float, D, D, C, D, D, C, C, true, false>(k_l1, v_l1, kv_l0, (bool)1);

      chunk_gdn_pto::copy_l0c_to_gm<half, float, 1, 1, 1, D, D, 1, 1, 1, D, 1, D, D>(
          workspace_handle + ws_base + WS_KV, C * D * static_cast<int32_t>(sizeof(float)), 0, D, D);
      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(2, 2);

      chunk_gdn_pto::wait_cross_flag(3);
    }
  }
#endif
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(zero_ub, 0.0f);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(s_ub, 0.0f);

    int64_t chunk_start_0 = bos;
    int64_t k_offset_0 = (chunk_start_0 * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
    chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, HalfC, D,
        1, 1, 1, BSND_QKV_STRIDE, 1,
        HalfC, D, pto::PadValue::Zero>(
        K_handle + k_offset_0, K_UB_HALF, 0, HalfC, D);

    {
      int64_t g_gm = chunk_start_0 * H;
      chunk_gdn_pto::copy_gm_to_ub<float, float, 1, 1, 1, C, H,
          1, 1, 1, H, 1,
          C, H, pto::PadValue::Zero>(
          G_handle + g_gm, G_BLOCK_UB, 0, C, H);
    }

    chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);

    {
      chunk_gdn_pto::TileUbDataND<float, C, H, C, H> g_block;
      TASSIGN(g_block, G_BLOCK_UB);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      for (int32_t gi = 0; gi < C; ++gi) {
        g_ub.SetValue(gi, g_block.GetValue(gi * H + static_cast<int32_t>(head)));
      }
    }

    chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_S>(0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_S>(0);

    for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;

      int64_t u_offset = (chunk_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
      chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, HalfC, D,
          1, 1, 1, BSND_QKV_STRIDE, 1,
          HalfC, D, pto::PadValue::Zero>(
          U_handle + u_offset, U_UB_HALF, 0, HalfC, D);

      TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);

      chunk_gdn_pto::TileUbDataND<float, 1, 64, 1, 64> g_ub_temp;
      TASSIGN(g_ub_temp, G_UB + vid * 64 * sizeof(float));
      TMOV(g_v_ub, g_ub_temp);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      TADDS(coeff_ub, g_v_ub, -g_last);
      pipe_barrier(PIPE_V);
      TSUB(coeff_ub, zero_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);

      TEXP(g_ub, g_ub);

      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);

      for (int32_t i_2 = 0; i_2 < HalfC / 4; ++i_2) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto c0 = coeff_ub.GetValue(i_2 * 4);
        chunk_gdn_pto::TileUbDataND<float, 1, D, 1, D> k0;
        TASSIGN(k0, K_UB + (i_2 * 4 * D) * sizeof(float));
        TMULS(k0, k0, c0);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto c1 = coeff_ub.GetValue(i_2 * 4 + 1);
        chunk_gdn_pto::TileUbDataND<float, 1, D, 1, D> k1;
        TASSIGN(k1, K_UB + ((i_2 * 4 + 1) * D) * sizeof(float));
        TMULS(k1, k1, c1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto c2 = coeff_ub.GetValue(i_2 * 4 + 2);
        chunk_gdn_pto::TileUbDataND<float, 1, D, 1, D> k2;
        TASSIGN(k2, K_UB + ((i_2 * 4 + 2) * D) * sizeof(float));
        TMULS(k2, k2, c2);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto c3 = coeff_ub.GetValue(i_2 * 4 + 3);
        chunk_gdn_pto::TileUbDataND<float, 1, D, 1, D> k3;
        TASSIGN(k3, K_UB + ((i_2 * 4 + 3) * D) * sizeof(float));
        TMULS(k3, k3, c3);
      }

      chunk_gdn_pto::wait_cross_flag(0);
      chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, HalfC, D,
          1, 1, 1, D, 1,
          HalfC, D, pto::PadValue::Zero>(
          workspace_handle + ws_base * sizeof(half) + WS_WS * sizeof(half) + vid * HalfC * D * sizeof(half),
          U_UB_HALF, 0, HalfC, D);

      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
      TSUB(u_ub, u_ub, ws_ub);
      TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);

      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3>(0);

      int64_t v_offset = (chunk_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
      chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, HalfC, D,
          1, 1, 1, BSND_QKV_STRIDE, 1,
          HalfC, D>(
          V_handle + v_offset, U_UB_HALF, 0, HalfC, D);

      chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, HalfC, D,
          1, 1, 1, D, 1,
          HalfC, D>(
          workspace_handle + ws_base * sizeof(half) + WS_K * sizeof(half) + vid * HalfC * D * sizeof(half),
          K_UB_HALF, 0, HalfC, D);

      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);

      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE3, PIPE_S>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE3, PIPE_S>(0);
      float exp_g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      TMULS(s_ub, s_ub, exp_g_last);

      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE2>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE2>(0);
      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        int64_t next_start = bos + static_cast<int64_t>(ci + 1) * C;
        int64_t next_valid = slen - static_cast<int64_t>(ci + 1) * C;
        if (next_valid > C) next_valid = C;

        int64_t nk_off = (next_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
        chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, HalfC, D,
            1, 1, 1, BSND_QKV_STRIDE, 1,
            HalfC, D, pto::PadValue::Zero>(
            K_handle + nk_off, K_UB_HALF, 0, HalfC, D);

        int64_t ng_gm = next_start * H;
        chunk_gdn_pto::copy_gm_to_ub<float, float, 1, 1, 1, C, H,
            1, 1, 1, H, 1,
            C, H, pto::PadValue::Zero>(
            G_handle + ng_gm, G_BLOCK_UB, 0, static_cast<int32_t>(next_valid), H);
      }

      chunk_gdn_pto::wait_cross_flag(2);
      chunk_gdn_pto::copy_gm_to_ub<half, half, 1, 1, 1, HalfC, D,
          1, 1, 1, D, 1,
          HalfC, D, pto::PadValue::Zero>(
          workspace_handle + ws_base * sizeof(half) + WS_KV * sizeof(half) + vid * HalfC * D * sizeof(half),
          S_UB_HALF, 0, HalfC, D);

      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      TCVT(kv_ub, s_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_ALL);
      TADD(s_ub, s_ub, kv_ub);
      TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
        chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
        chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, HalfC, D,
            1, 1, 1, D, 1,
            HalfC, D>(
            workspace_handle + ws_base * sizeof(half) + WS_S * sizeof(half) + vid * HalfC * D * sizeof(half),
            S_UB_HALF, 0, HalfC, D);

        int64_t s_out_offset = ((chunk_offset + ci + 1) * H + head) * DD;
        chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, HalfC, D,
            1, 1, 1, D, 1,
            HalfC, D>(
            S_handle + s_out_offset + vid * HalfC * D, S_UB_HALF, 0, HalfC, D);
      }

      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(3, 2);

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
        chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
        {
          chunk_gdn_pto::TileUbDataND<float, C, H, C, H> g_block;
          TASSIGN(g_block, G_BLOCK_UB);
          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
          for (int32_t gi = 0; gi < C; ++gi) {
            g_ub.SetValue(gi, g_block.GetValue(gi * H + static_cast<int32_t>(head)));
          }
        }
      }
    }

    chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
    chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
    int64_t fs_offset = (seq_idx * H + head) * DD;
    chunk_gdn_pto::copy_ub_to_gm<half, half, 1, 1, 1, HalfC, D,
        1, 1, 1, D, 1,
        HalfC, D>(
        FS_handle + fs_offset + vid * HalfC * D, S_UB_HALF, 0, HalfC, D);
  }
#endif
}

extern "C" __global__ AICORE void launch_chunk_h(
    __gm__ uint8_t *K, __gm__ uint8_t *W, __gm__ uint8_t *U,
    __gm__ uint8_t *G,
    __gm__ uint8_t *S, __gm__ uint8_t *V, __gm__ uint8_t *FS,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  chunk_h_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K),
      reinterpret_cast<__gm__ half *>(W),
      reinterpret_cast<__gm__ half *>(U),
      reinterpret_cast<__gm__ float *>(G),
      reinterpret_cast<__gm__ half *>(S),
      reinterpret_cast<__gm__ half *>(V),
      reinterpret_cast<__gm__ half *>(FS),
      reinterpret_cast<__gm__ half *>(workspace),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *K, uint8_t *W, uint8_t *U, uint8_t *G,
    uint8_t *S, uint8_t *V, uint8_t *FS,
    uint8_t *workspace,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_h<<<block_dim, nullptr, stream>>>(
      K, W, U, G, S, V, FS, workspace, cu_seqlens,
      batch_size, seq_len, fftsAddr);
}
