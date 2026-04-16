#include "common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void wy_fast_kernel(
    __gm__ half *K_handle, __gm__ half *V_handle,
    __gm__ half *Beta_handle, __gm__ float *G_handle,
    __gm__ half *A_handle,
    __gm__ half *workspace_a1_handle, __gm__ half *workspace_a2_handle,
    __gm__ half *W_handle, __gm__ half *U_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);

  constexpr int32_t GHeadTileCols = ((NumHeads + 7) / 8) * 8;
  constexpr int32_t BetaHeadTileCols = ((NumHeads + 15) / 16) * 16;

  constexpr int32_t BetaHalfUbAddr = 0;
  constexpr int32_t A1HalfUbAddr   = 256;
  constexpr int32_t BetaUbAddr     = 16640;
  constexpr int32_t BetaRUbAddr    = 17152;
  constexpr int32_t Beta2dUbAddr   = 17664;
  constexpr int32_t TmpUbAddr      = 50432;
  constexpr int32_t A1UbAddr       = 75008;
  constexpr int32_t A2UbAddr       = 107776;
  constexpr int32_t A2HalfUbAddr   = 140544;
  constexpr int32_t GUbAddr        = 156928;
  constexpr int32_t GRUbAddr       = 157440;
  constexpr int32_t G2dUbAddr      = 157952;

  constexpr int32_t GBlockUbAddr    = TmpUbAddr;
  constexpr int32_t BetaBlockUbAddr = TmpUbAddr;

  constexpr int32_t WsA1Size = ChunkSize * ChunkSize;
  constexpr int32_t WsA2Size = ChunkSize * ChunkSize;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  chunk_gdn_pto::TileUbDataND<half, 1, ChunkSize, 1, ChunkSize> beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  chunk_gdn_pto::TileUbDataND<half, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> a1_ub_half;
  TASSIGN(a1_ub_half, A1HalfUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> beta_r_ub;
  TASSIGN(beta_r_ub, BetaRUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> beta_2d_ub;
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  chunk_gdn_pto::TileUbDataND<uint8_t, 1, 24576, 1, 24576> tmp_ub;
  TASSIGN(tmp_ub, TmpUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> a1_ub;
  TASSIGN(a1_ub, A1UbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> a2_ub;
  TASSIGN(a2_ub, A2UbAddr);
  chunk_gdn_pto::TileUbDataND<half, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> a2_ub_half;
  TASSIGN(a2_ub_half, A2HalfUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> g_2d_ub;
  TASSIGN(g_2d_ub, G2dUbAddr);

  chunk_gdn_pto::TileMatL1<half, ChunkSize, HiddenSize,
                            ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  chunk_gdn_pto::TileMatL1<half, ChunkSize, HiddenSize,
                            ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 32768);
  chunk_gdn_pto::TileMatL1<half, ChunkSize, ChunkSize,
                            ChunkSize, ChunkSize> a2_l1;
  TASSIGN(a2_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> u_l0;
  TASSIGN(u_l0, 0);
  chunk_gdn_pto::TileMatL1<half, ChunkSize, ChunkSize,
                            ChunkSize, ChunkSize> a1_l1;
  TASSIGN(a1_l1, 98304);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> w_l0;
  TASSIGN(w_l0, 65536);

  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * NumHeads;
  }

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    bool first_iter = true;
    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;

      // Load beta from BSND [B,S,H]
      chunk_gdn_pto::TileUbDataND<half, ChunkSize, BetaHeadTileCols,
                                   ChunkSize, BetaHeadTileCols> beta_block_ub;
      TASSIGN(beta_block_ub, BetaBlockUbAddr);
      chunk_gdn_pto::copy_gm_to_ub<half, half,
          1, 1, 1, ChunkSize, BetaHeadTileCols,
          1, 1, 1, NumHeads, 1,
          ChunkSize, BetaHeadTileCols, pto::PadValue::Zero>(
          Beta_handle + chunk_token_start * NumHeads,
          BetaBlockUbAddr, 0, valid_rows, NumHeads);

      // Load A from BSND [B,S,H,C]
      int64_t a_gm_offset =
          ((chunk_token_start +
            static_cast<int64_t>(vid) * HalfChunk) *
           NumHeads + head_idx) *
          static_cast<int64_t>(ChunkSize);
      chunk_gdn_pto::copy_gm_to_ub<half, half,
          1, 1, 1, HalfChunk, ChunkSize,
          1, 1, 1, NumHeads * ChunkSize, 1,
          HalfChunk, ChunkSize, pto::PadValue::Zero>(
          A_handle + a_gm_offset,
          A1HalfUbAddr, 0, HalfChunk, ChunkSize);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

      for (int32_t i = 0; i < valid_rows; ++i) {
        beta_ub_half.SetValue(i,
            beta_block_ub.GetValue(i * BetaHeadTileCols + head_idx));
      }
      for (int32_t i = valid_rows; i < ChunkSize; ++i) {
        beta_ub_half.SetValue(i, static_cast<half>(0.0f));
      }

      pipe_barrier(PIPE_ALL);

      TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMOV(beta_r_ub, beta_ub);
      pipe_barrier(PIPE_V);
      TCOLEXPAND(beta_2d_ub, beta_r_ub);

      TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
      TMUL(a2_ub, a1_ub, beta_2d_ub);
      TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

      if (!first_iter) wait_flag_dev(3);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      chunk_gdn_pto::copy_ub_to_gm<half, half,
          1, 1, 1, HalfChunk, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          HalfChunk, ChunkSize>(
          workspace_a2_handle +
              static_cast<int64_t>(cid) * WsA2Size +
              static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
          A2HalfUbAddr, 0, HalfChunk, ChunkSize);
      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(2, 2);

      // Load g_sum from BSND [B,S,H]
      chunk_gdn_pto::TileUbDataND<float, ChunkSize, GHeadTileCols,
                                   ChunkSize, GHeadTileCols> g_block_ub;
      TASSIGN(g_block_ub, GBlockUbAddr);
      chunk_gdn_pto::copy_gm_to_ub<float, float,
          1, 1, 1, ChunkSize, GHeadTileCols,
          1, 1, 1, NumHeads, 1,
          ChunkSize, GHeadTileCols, pto::PadValue::Zero>(
          G_handle + chunk_token_start * NumHeads,
          GBlockUbAddr, 0, valid_rows, NumHeads);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

      for (int32_t i = 0; i < valid_rows; ++i) {
        g_ub.SetValue(i,
            g_block_ub.GetValue(i * GHeadTileCols + head_idx));
      }
      for (int32_t i = valid_rows; i < ChunkSize; ++i) {
        g_ub.SetValue(i, 0.0f);
      }

      pipe_barrier(PIPE_ALL);

      TEXP(g_ub, g_ub);
      pipe_barrier(PIPE_V);
      TMUL(g_ub, g_ub, beta_ub);
      pipe_barrier(PIPE_V);
      TMOV(g_r_ub, g_ub);
      pipe_barrier(PIPE_V);
      TCOLEXPAND(g_2d_ub, g_r_ub);
      TMUL(a1_ub, a1_ub, g_2d_ub);
      TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

      if (!first_iter) wait_flag_dev(4);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      chunk_gdn_pto::copy_ub_to_gm<half, half,
          1, 1, 1, HalfChunk, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          HalfChunk, ChunkSize>(
          workspace_a1_handle +
              static_cast<int64_t>(cid) * WsA1Size +
              static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
          A1HalfUbAddr, 0, HalfChunk, ChunkSize);
      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);
      first_iter = false;
    }
  } else {
    int64_t gi = 0;
    bool first_iter_v = true;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;

            chunk_gdn_pto::TileUbDataND<half, ChunkSize, BetaHeadTileCols,
                                         ChunkSize, BetaHeadTileCols>
                beta_block_ub;
            TASSIGN(beta_block_ub, BetaBlockUbAddr);
            chunk_gdn_pto::copy_gm_to_ub<half, half,
                1, 1, 1, ChunkSize, BetaHeadTileCols,
                1, 1, 1, NumHeads, 1,
                ChunkSize, BetaHeadTileCols, pto::PadValue::Zero>(
                Beta_handle + chunk_token_start * NumHeads,
                BetaBlockUbAddr, 0, valid_rows, NumHeads);

            int64_t a_gm_offset =
                ((chunk_token_start +
                  static_cast<int64_t>(vid) * HalfChunk) *
                 NumHeads + head_idx) *
                static_cast<int64_t>(ChunkSize);
            chunk_gdn_pto::copy_gm_to_ub<half, half,
                1, 1, 1, HalfChunk, ChunkSize,
                1, 1, 1, NumHeads * ChunkSize, 1,
                HalfChunk, ChunkSize, pto::PadValue::Zero>(
                A_handle + a_gm_offset,
                A1HalfUbAddr, 0, HalfChunk, ChunkSize);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            for (int32_t i = 0; i < valid_rows; ++i) {
              beta_ub_half.SetValue(i,
                  beta_block_ub.GetValue(
                      i * BetaHeadTileCols + head_idx));
            }
            for (int32_t i = valid_rows; i < ChunkSize; ++i) {
              beta_ub_half.SetValue(i, static_cast<half>(0.0f));
            }

            pipe_barrier(PIPE_ALL);

            TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            TMOV(beta_r_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(beta_2d_ub, beta_r_ub);

            TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
            TMUL(a2_ub, a1_ub, beta_2d_ub);
            TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

            if (!first_iter_v) wait_flag_dev(3);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            chunk_gdn_pto::copy_ub_to_gm<half, half,
                1, 1, 1, HalfChunk, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                HalfChunk, ChunkSize>(
                workspace_a2_handle +
                    static_cast<int64_t>(cid) * WsA2Size +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                A2HalfUbAddr, 0, HalfChunk, ChunkSize);
            chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(2, 2);

            chunk_gdn_pto::TileUbDataND<float, ChunkSize, GHeadTileCols,
                                         ChunkSize, GHeadTileCols>
                g_block_ub;
            TASSIGN(g_block_ub, GBlockUbAddr);
            chunk_gdn_pto::copy_gm_to_ub<float, float,
                1, 1, 1, ChunkSize, GHeadTileCols,
                1, 1, 1, NumHeads, 1,
                ChunkSize, GHeadTileCols, pto::PadValue::Zero>(
                G_handle + chunk_token_start * NumHeads,
                GBlockUbAddr, 0, valid_rows, NumHeads);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            for (int32_t i = 0; i < valid_rows; ++i) {
              g_ub.SetValue(i,
                  g_block_ub.GetValue(
                      i * GHeadTileCols + head_idx));
            }
            for (int32_t i = valid_rows; i < ChunkSize; ++i) {
              g_ub.SetValue(i, 0.0f);
            }

            pipe_barrier(PIPE_ALL);

            TEXP(g_ub, g_ub);
            pipe_barrier(PIPE_V);
            TMUL(g_ub, g_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TMOV(g_r_ub, g_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(g_2d_ub, g_r_ub);
            TMUL(a1_ub, a1_ub, g_2d_ub);
            TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

            if (!first_iter_v) wait_flag_dev(4);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            chunk_gdn_pto::copy_ub_to_gm<half, half,
                1, 1, 1, HalfChunk, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                HalfChunk, ChunkSize>(
                workspace_a1_handle +
                    static_cast<int64_t>(cid) * WsA1Size +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                A1HalfUbAddr, 0, HalfChunk, ChunkSize);
            chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);
            first_iter_v = false;
          }
          gi++;
        }
      }
    }
  }
#endif

#if defined(__DAV_C220_CUBE__)
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;

      int64_t kv_offset =
          (chunk_token_start * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize);

      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          K_handle + kv_offset, 0, 0, valid_rows, HiddenSize);
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          V_handle + kv_offset, 32768, 0, valid_rows, HiddenSize);

      wait_flag_dev(2);
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          ChunkSize, ChunkSize>(
          workspace_a2_handle +
              static_cast<int64_t>(cid) * WsA2Size,
          65536, 0, ChunkSize, ChunkSize);

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      chunk_gdn_pto::gemm_v0<half, float,
          ChunkSize, HiddenSize, ChunkSize,
          ChunkSize, HiddenSize, ChunkSize,
          KTail, false, false>(a2_l1, v_l1, u_l0, true);

      chunk_gdn_pto::copy_l0c_to_gm<half, float,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          U_handle + kv_offset, 0, 0, valid_rows, HiddenSize);
      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(3, 2);

      wait_flag_dev(1);
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          ChunkSize, ChunkSize>(
          workspace_a1_handle +
              static_cast<int64_t>(cid) * WsA1Size,
          98304, 0, ChunkSize, ChunkSize);

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      chunk_gdn_pto::gemm_v0<half, float,
          ChunkSize, HiddenSize, ChunkSize,
          ChunkSize, HiddenSize, ChunkSize,
          KTail, false, false>(a1_l1, k_l1, w_l0, true);

      chunk_gdn_pto::copy_l0c_to_gm<half, float,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          W_handle + kv_offset, 65536, 0, valid_rows, HiddenSize);
      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(4, 2);
    }
  } else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;

            int64_t kv_offset =
                (chunk_token_start * NumHeads + head_idx) *
                static_cast<int64_t>(HiddenSize);

            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                ChunkSize, HiddenSize>(
                K_handle + kv_offset, 0, 0, valid_rows, HiddenSize);
            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                ChunkSize, HiddenSize>(
                V_handle + kv_offset, 32768, 0, valid_rows, HiddenSize);

            wait_flag_dev(2);
            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                ChunkSize, ChunkSize>(
                workspace_a2_handle +
                    static_cast<int64_t>(cid) * WsA2Size,
                65536, 0, ChunkSize, ChunkSize);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            chunk_gdn_pto::gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a2_l1, v_l1, u_l0, true);

            chunk_gdn_pto::copy_l0c_to_gm<half, float,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                ChunkSize, HiddenSize>(
                U_handle + kv_offset, 0, 0, valid_rows, HiddenSize);
            chunk_gdn_pto::set_cross_flag<PIPE_FIX>(3, 2);

            wait_flag_dev(1);
            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                ChunkSize, ChunkSize>(
                workspace_a1_handle +
                    static_cast<int64_t>(cid) * WsA1Size,
                98304, 0, ChunkSize, ChunkSize);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            chunk_gdn_pto::gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a1_l1, k_l1, w_l0, true);

            chunk_gdn_pto::copy_l0c_to_gm<half, float,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                ChunkSize, HiddenSize>(
                W_handle + kv_offset, 65536, 0, valid_rows, HiddenSize);
            chunk_gdn_pto::set_cross_flag<PIPE_FIX>(4, 2);
          }
          gi++;
        }
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_wy_fast(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle,
    __gm__ uint8_t *A_handle,
    __gm__ uint8_t *workspace_a1_handle, __gm__ uint8_t *workspace_a2_handle,
    __gm__ uint8_t *W_handle, __gm__ uint8_t *U_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  wy_fast_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ half *>(workspace_a1_handle),
      reinterpret_cast<__gm__ half *>(workspace_a2_handle),
      reinterpret_cast<__gm__ half *>(W_handle),
      reinterpret_cast<__gm__ half *>(U_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *k, uint8_t *v, uint8_t *beta, uint8_t *g_sum, uint8_t *A,
    uint8_t *workspace_a1, uint8_t *workspace_a2,
    uint8_t *w, uint8_t *u,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_wy_fast<<<block_dim, nullptr, stream>>>(
      k, v, beta, g_sum, A,
      workspace_a1, workspace_a2,
      w, u,
      cu_seqlens,
      batch_size, seq_len, fftsAddr);
}
