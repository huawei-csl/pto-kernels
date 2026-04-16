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
AICORE void chunk_o_kernel(
    __gm__ half *Q_handle, __gm__ half *K_handle, __gm__ half *V_handle,
    __gm__ half *S_handle, __gm__ float *G_handle,
    __gm__ float *Msk_handle,
    __gm__ half *workspace_qk_handle,
    __gm__ half *workspace_qs_qkv_handle,
    __gm__ half *workspace_qk_gated_handle,
    __gm__ half *O_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);
  constexpr uint32_t CTail =
      (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

  constexpr int32_t GHeadTileCols = ((NumHeads + 7) / 8) * 8;

  constexpr int32_t WsQKSize = ChunkSize * ChunkSize;
  constexpr int32_t WsQSSize = ChunkSize * HiddenSize;
  constexpr int32_t WsGatedSize = ChunkSize * ChunkSize;

  constexpr int32_t GUbAddr      = 0;
  constexpr int32_t MskUbAddr    = 512;
  constexpr int32_t QKUbAddr     = 33280;
  constexpr int32_t GvUbAddr     = 66048;
  constexpr int32_t CoeffUbAddr  = 66304;
  constexpr int32_t QKHalfUbAddr = 99072;
  constexpr int32_t QSHalfUbAddr = 115456;
  constexpr int32_t QSUbAddr     = 131840;
  constexpr int32_t OHalfUbAddr  = 164608;
  constexpr int32_t OUbAddr      = 512;

  constexpr int32_t GBlockUbAddr = QKUbAddr;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  chunk_gdn_pto::TileMatL1<half, ChunkSize, HiddenSize,
                            ChunkSize, HiddenSize> q_l1;
  TASSIGN(q_l1, 0);
  chunk_gdn_pto::TileMatL1<half, ChunkSize, HiddenSize,
                            ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 32768);
  TileAcc<float, ChunkSize, ChunkSize,
          ChunkSize, ChunkSize> qk_l0;
  TASSIGN(qk_l0, 0);
  chunk_gdn_pto::TileMatL1<half, HiddenSize, HiddenSize,
                            HiddenSize, HiddenSize> s_l1;
  TASSIGN(s_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> qs_l0;
  TASSIGN(qs_l0, 65536);
  chunk_gdn_pto::TileMatL1<half, ChunkSize, ChunkSize,
                            ChunkSize, ChunkSize> qk_gated_l1;
  TASSIGN(qk_gated_l1, 98304);
  chunk_gdn_pto::TileMatL1<half, ChunkSize, HiddenSize,
                            ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 131072);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> qkv_l0;
  TASSIGN(qkv_l0, 0);

  chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                               1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> qk_ub;
  TASSIGN(qk_ub, QKUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk,
                               1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  chunk_gdn_pto::TileUbDataND<half, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> qk_ub_half;
  TASSIGN(qk_ub_half, QKHalfUbAddr);
  chunk_gdn_pto::TileUbDataND<half, HalfChunk, HiddenSize,
                               HalfChunk, HiddenSize> qs_ub_half;
  TASSIGN(qs_ub_half, QSHalfUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, HiddenSize,
                               HalfChunk, HiddenSize> qs_ub;
  TASSIGN(qs_ub, QSUbAddr);
  chunk_gdn_pto::TileUbDataND<half, HalfChunk, HiddenSize,
                               HalfChunk, HiddenSize> o_ub_half;
  TASSIGN(o_ub_half, OHalfUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, HiddenSize,
                               HalfChunk, HiddenSize> o_ub;
  TASSIGN(o_ub, OUbAddr);

  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * NumHeads;
  }

#if defined(__DAV_C220_CUBE__)
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t global_chunk_base = 0;

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

      int64_t qkv_offset =
          (chunk_token_start * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize);

      int64_t chunk_global_idx = seq_idx * chunks_per_seq + ci;
      int64_t s_offset =
          (chunk_global_idx * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize) *
          static_cast<int64_t>(HiddenSize);

      // Step 1: Q @ K^T -> workspace_qk
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          Q_handle + qkv_offset, 0, 0, valid_rows, HiddenSize);
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          K_handle + qkv_offset, 32768, 0, valid_rows, HiddenSize);

      chunk_gdn_pto::gemm_v0<half, float,
          ChunkSize, ChunkSize, HiddenSize,
          ChunkSize, ChunkSize, HiddenSize,
          KTail, false, true>(q_l1, k_l1, qk_l0, true);

      // Step 2: Q @ S -> workspace_qs
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          Q_handle + qkv_offset, 0, 0, valid_rows, HiddenSize);
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, HiddenSize, HiddenSize,
          1, 1, 1, HiddenSize, 1,
          HiddenSize, HiddenSize>(
          S_handle + s_offset, 65536, 0, HiddenSize, HiddenSize);

      chunk_gdn_pto::gemm_v0<half, float,
          ChunkSize, HiddenSize, HiddenSize,
          ChunkSize, HiddenSize, HiddenSize,
          KTail, false, false>(q_l1, s_l1, qs_l0, true);

      // Store QK and QS to workspace (per-core)
      chunk_gdn_pto::copy_l0c_to_gm<half, float,
          1, 1, 1, ChunkSize, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          ChunkSize, ChunkSize>(
          workspace_qk_handle +
              static_cast<int64_t>(cid) * WsQKSize,
          0, 0, ChunkSize, ChunkSize);

      chunk_gdn_pto::copy_l0c_to_gm<half, float,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, HiddenSize, 1,
          ChunkSize, HiddenSize>(
          workspace_qs_qkv_handle +
              static_cast<int64_t>(cid) * WsQSSize,
          65536, 0, ChunkSize, HiddenSize);

      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(0, 2);

      // Wait for vec to finish gating QK
      chunk_gdn_pto::wait_cross_flag(1);

      // Step 3: gated_QK @ V -> workspace_qkv
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          ChunkSize, ChunkSize>(
          workspace_qk_gated_handle +
              static_cast<int64_t>(cid) * WsGatedSize,
          98304, 0, ChunkSize, ChunkSize);
      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          V_handle + qkv_offset, 131072, 0, valid_rows, HiddenSize);

      chunk_gdn_pto::gemm_v0<half, float,
          ChunkSize, HiddenSize, ChunkSize,
          ChunkSize, HiddenSize, ChunkSize,
          CTail, false, false>(qk_gated_l1, v_l1, qkv_l0, true);

      // Store QKV to workspace (reuse qs_qkv workspace)
      chunk_gdn_pto::copy_l0c_to_gm<half, float,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, HiddenSize, 1,
          ChunkSize, HiddenSize>(
          workspace_qs_qkv_handle +
              static_cast<int64_t>(cid) * WsQSSize,
          0, 0, ChunkSize, HiddenSize);

      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(2, 2);
    }
  } else {
    int64_t gi = 0;
    int64_t chunk_global_idx = 0;
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

            int64_t qkv_offset =
                (chunk_token_start * NumHeads + head_idx) *
                static_cast<int64_t>(HiddenSize);
            int64_t s_offset =
                (chunk_global_idx * NumHeads + head_idx) *
                static_cast<int64_t>(HiddenSize) *
                static_cast<int64_t>(HiddenSize);

            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                ChunkSize, HiddenSize>(
                Q_handle + qkv_offset, 0, 0, valid_rows, HiddenSize);
            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                ChunkSize, HiddenSize>(
                K_handle + qkv_offset, 32768, 0, valid_rows, HiddenSize);

            chunk_gdn_pto::gemm_v0<half, float,
                ChunkSize, ChunkSize, HiddenSize,
                ChunkSize, ChunkSize, HiddenSize,
                KTail, false, true>(q_l1, k_l1, qk_l0, true);

            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                ChunkSize, HiddenSize>(
                Q_handle + qkv_offset, 0, 0, valid_rows, HiddenSize);
            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, HiddenSize, HiddenSize,
                1, 1, 1, HiddenSize, 1,
                HiddenSize, HiddenSize>(
                S_handle + s_offset, 65536, 0, HiddenSize, HiddenSize);

            chunk_gdn_pto::gemm_v0<half, float,
                ChunkSize, HiddenSize, HiddenSize,
                ChunkSize, HiddenSize, HiddenSize,
                KTail, false, false>(q_l1, s_l1, qs_l0, true);

            chunk_gdn_pto::copy_l0c_to_gm<half, float,
                1, 1, 1, ChunkSize, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                ChunkSize, ChunkSize>(
                workspace_qk_handle +
                    static_cast<int64_t>(cid) * WsQKSize,
                0, 0, ChunkSize, ChunkSize);

            chunk_gdn_pto::copy_l0c_to_gm<half, float,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, HiddenSize, 1,
                ChunkSize, HiddenSize>(
                workspace_qs_qkv_handle +
                    static_cast<int64_t>(cid) * WsQSSize,
                65536, 0, ChunkSize, HiddenSize);

            chunk_gdn_pto::set_cross_flag<PIPE_FIX>(0, 2);

            chunk_gdn_pto::wait_cross_flag(1);

            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                ChunkSize, ChunkSize>(
                workspace_qk_gated_handle +
                    static_cast<int64_t>(cid) * WsGatedSize,
                98304, 0, ChunkSize, ChunkSize);
            chunk_gdn_pto::copy_gm_to_l1<half, half,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                ChunkSize, HiddenSize>(
                V_handle + qkv_offset, 131072, 0, valid_rows, HiddenSize);

            chunk_gdn_pto::gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                CTail, false, false>(qk_gated_l1, v_l1, qkv_l0, true);

            chunk_gdn_pto::copy_l0c_to_gm<half, float,
                1, 1, 1, ChunkSize, HiddenSize,
                1, 1, 1, HiddenSize, 1,
                ChunkSize, HiddenSize>(
                workspace_qs_qkv_handle +
                    static_cast<int64_t>(cid) * WsQSSize,
                0, 0, ChunkSize, HiddenSize);

            chunk_gdn_pto::set_cross_flag<PIPE_FIX>(2, 2);
          }
          gi++;
        }
        chunk_global_idx++;
      }
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

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

      // Load g_sum from BSND [B,S,H] into g_ub [1, ChunkSize]
      chunk_gdn_pto::TileUbDataND<float, ChunkSize, GHeadTileCols,
                                   ChunkSize, GHeadTileCols> g_block_ub;
      TASSIGN(g_block_ub, GBlockUbAddr);
      chunk_gdn_pto::copy_gm_to_ub<float, float,
          1, 1, 1, ChunkSize, GHeadTileCols,
          1, 1, 1, NumHeads, 1,
          ChunkSize, GHeadTileCols, pto::PadValue::Zero>(
          G_handle + chunk_token_start * NumHeads,
          GBlockUbAddr, 0, valid_rows, NumHeads);

      // Load mask [HalfChunk, ChunkSize] (vid selects half)
      chunk_gdn_pto::copy_gm_to_ub<float, float,
          1, 1, 1, HalfChunk, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          HalfChunk, ChunkSize, pto::PadValue::Zero>(
          Msk_handle +
              static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
          MskUbAddr, 0, HalfChunk, ChunkSize);

      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);

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

      TEXPANDS(qk_ub, 0.0f);
      chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk,
                                   1, HalfChunk> g_ub_temp_0;
      TASSIGN(g_ub_temp_0,
              GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                            static_cast<int32_t>(sizeof(float)));
      TMOV(g_v_ub, g_ub_temp_0);

      // Build gating coefficient matrix: exp(g_row - g_col)
      for (int32_t i = 0; i < HalfChunk / 4; ++i) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto g_val_0 = g_v_ub.GetValue(i * 4);
        chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                     1, ChunkSize> g_ub_t0;
        TASSIGN(g_ub_t0, GUbAddr);
        chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                     1, ChunkSize> coeff_t0;
        TASSIGN(coeff_t0,
                CoeffUbAddr +
                    (i * 4 * ChunkSize) *
                        static_cast<int32_t>(sizeof(float)));
        TADDS(coeff_t0, g_ub_t0, -g_val_0);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto g_val_1 = g_v_ub.GetValue(i * 4 + 1);
        chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                     1, ChunkSize> g_ub_t1;
        TASSIGN(g_ub_t1, GUbAddr);
        chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                     1, ChunkSize> coeff_t1;
        TASSIGN(coeff_t1,
                CoeffUbAddr +
                    ((i * 4 + 1) * ChunkSize) *
                        static_cast<int32_t>(sizeof(float)));
        TADDS(coeff_t1, g_ub_t1, -g_val_1);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto g_val_2 = g_v_ub.GetValue(i * 4 + 2);
        chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                     1, ChunkSize> g_ub_t2;
        TASSIGN(g_ub_t2, GUbAddr);
        chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                     1, ChunkSize> coeff_t2;
        TASSIGN(coeff_t2,
                CoeffUbAddr +
                    ((i * 4 + 2) * ChunkSize) *
                        static_cast<int32_t>(sizeof(float)));
        TADDS(coeff_t2, g_ub_t2, -g_val_2);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto g_val_3 = g_v_ub.GetValue(i * 4 + 3);
        chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                     1, ChunkSize> g_ub_t3;
        TASSIGN(g_ub_t3, GUbAddr);
        chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                     1, ChunkSize> coeff_t3;
        TASSIGN(coeff_t3,
                CoeffUbAddr +
                    ((i * 4 + 3) * ChunkSize) *
                        static_cast<int32_t>(sizeof(float)));
        TADDS(coeff_t3, g_ub_t3, -g_val_3);
      }

      TSUB(coeff_ub, qk_ub, coeff_ub);
      TMUL(coeff_ub, coeff_ub, msk_ub);
      TEXP(coeff_ub, coeff_ub);
      TEXP(g_v_ub, g_v_ub);

      // Wait for cube to finish QK and QS
      chunk_gdn_pto::wait_cross_flag(0);

      // Load QK from workspace
      chunk_gdn_pto::copy_gm_to_ub<half, half,
          1, 1, 1, HalfChunk, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          HalfChunk, ChunkSize, pto::PadValue::Zero>(
          workspace_qk_handle +
              static_cast<int64_t>(cid) * WsQKSize +
              static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
          QKHalfUbAddr, 0, HalfChunk, ChunkSize);

      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE2>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE2>(0);

      // Load QS from workspace
      chunk_gdn_pto::copy_gm_to_ub<half, half,
          1, 1, 1, HalfChunk, HiddenSize,
          1, 1, 1, HiddenSize, 1,
          HalfChunk, HiddenSize, pto::PadValue::Zero>(
          workspace_qs_qkv_handle +
              static_cast<int64_t>(cid) * WsQSSize +
              static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
          QSHalfUbAddr, 0, HalfChunk, HiddenSize);

      // Apply gating: QK * coeff * mask
      TMUL(qk_ub, qk_ub, coeff_ub);
      TMUL(qk_ub, qk_ub, msk_ub);
      TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

      // Store gated QK to workspace for cube
      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
      chunk_gdn_pto::copy_ub_to_gm<half, half,
          1, 1, 1, HalfChunk, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          HalfChunk, ChunkSize>(
          workspace_qk_gated_handle +
              static_cast<int64_t>(cid) * WsGatedSize +
              static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
          QKHalfUbAddr, 0, HalfChunk, ChunkSize);
      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);

      // Convert QS to float
      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);

      // Apply exp(g) row-wise scaling to QS
      for (int32_t i = 0; i < HalfChunk / 4; ++i) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto gv0 = g_v_ub.GetValue(i * 4);
        chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                     1, HiddenSize> qs_r0;
        TASSIGN(qs_r0,
                QSUbAddr +
                    (i * 4 * HiddenSize) *
                        static_cast<int32_t>(sizeof(float)));
        chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                     1, HiddenSize> qs_d0;
        TASSIGN(qs_d0,
                QSUbAddr +
                    (i * 4 * HiddenSize) *
                        static_cast<int32_t>(sizeof(float)));
        TMULS(qs_d0, qs_r0, gv0);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto gv1 = g_v_ub.GetValue(i * 4 + 1);
        chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                     1, HiddenSize> qs_r1;
        TASSIGN(qs_r1,
                QSUbAddr +
                    ((i * 4 + 1) * HiddenSize) *
                        static_cast<int32_t>(sizeof(float)));
        chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                     1, HiddenSize> qs_d1;
        TASSIGN(qs_d1,
                QSUbAddr +
                    ((i * 4 + 1) * HiddenSize) *
                        static_cast<int32_t>(sizeof(float)));
        TMULS(qs_d1, qs_r1, gv1);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto gv2 = g_v_ub.GetValue(i * 4 + 2);
        chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                     1, HiddenSize> qs_r2;
        TASSIGN(qs_r2,
                QSUbAddr +
                    ((i * 4 + 2) * HiddenSize) *
                        static_cast<int32_t>(sizeof(float)));
        chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                     1, HiddenSize> qs_d2;
        TASSIGN(qs_d2,
                QSUbAddr +
                    ((i * 4 + 2) * HiddenSize) *
                        static_cast<int32_t>(sizeof(float)));
        TMULS(qs_d2, qs_r2, gv2);

        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        auto gv3 = g_v_ub.GetValue(i * 4 + 3);
        chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                     1, HiddenSize> qs_r3;
        TASSIGN(qs_r3,
                QSUbAddr +
                    ((i * 4 + 3) * HiddenSize) *
                        static_cast<int32_t>(sizeof(float)));
        chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                     1, HiddenSize> qs_d3;
        TASSIGN(qs_d3,
                QSUbAddr +
                    ((i * 4 + 3) * HiddenSize) *
                        static_cast<int32_t>(sizeof(float)));
        TMULS(qs_d3, qs_r3, gv3);
      }

      // Wait for cube to finish QKV
      chunk_gdn_pto::wait_cross_flag(2);

      // Load QKV from workspace
      chunk_gdn_pto::copy_gm_to_ub<half, half,
          1, 1, 1, HalfChunk, HiddenSize,
          1, 1, 1, HiddenSize, 1,
          HalfChunk, HiddenSize, pto::PadValue::Zero>(
          workspace_qs_qkv_handle +
              static_cast<int64_t>(cid) * WsQSSize +
              static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
          OHalfUbAddr, 0, HalfChunk, HiddenSize);

      chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);

      // O = QS_gated + QKV
      TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
      TADD(o_ub, qs_ub, o_ub);
      TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

      // Store O to BSND
      chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
      chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3>(0);

      int64_t o_offset =
          (chunk_token_start * NumHeads + head_idx) *
              static_cast<int64_t>(HiddenSize) +
          static_cast<int64_t>(vid) * HalfChunk * NumHeads * HiddenSize;

      chunk_gdn_pto::copy_ub_to_gm<half, half,
          1, 1, 1, HalfChunk, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          HalfChunk, HiddenSize>(
          O_handle + o_offset,
          OHalfUbAddr, 0, HalfChunk, HiddenSize);
    }
  } else {
    int64_t gi = 0;
    int64_t chunk_global_idx = 0;
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

            chunk_gdn_pto::copy_gm_to_ub<float, float,
                1, 1, 1, HalfChunk, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                HalfChunk, ChunkSize, pto::PadValue::Zero>(
                Msk_handle +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                MskUbAddr, 0, HalfChunk, ChunkSize);

            chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
            chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);

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

            TEXPANDS(qk_ub, 0.0f);
            chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk,
                                         1, HalfChunk> g_ub_temp_v;
            TASSIGN(g_ub_temp_v,
                    GUbAddr +
                        static_cast<int32_t>(vid) * HalfChunk *
                            static_cast<int32_t>(sizeof(float)));
            TMOV(g_v_ub, g_ub_temp_v);

            for (int32_t i = 0; i < HalfChunk / 4; ++i) {
              set_flag(PIPE_V, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
              auto gv0 = g_v_ub.GetValue(i * 4);
              chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                           1, ChunkSize> gt0;
              TASSIGN(gt0, GUbAddr);
              chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                           1, ChunkSize> ct0;
              TASSIGN(ct0,
                      CoeffUbAddr +
                          (i * 4 * ChunkSize) *
                              static_cast<int32_t>(sizeof(float)));
              TADDS(ct0, gt0, -gv0);

              set_flag(PIPE_V, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
              auto gv1 = g_v_ub.GetValue(i * 4 + 1);
              chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                           1, ChunkSize> gt1;
              TASSIGN(gt1, GUbAddr);
              chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                           1, ChunkSize> ct1;
              TASSIGN(ct1,
                      CoeffUbAddr +
                          ((i * 4 + 1) * ChunkSize) *
                              static_cast<int32_t>(sizeof(float)));
              TADDS(ct1, gt1, -gv1);

              set_flag(PIPE_V, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
              auto gv2 = g_v_ub.GetValue(i * 4 + 2);
              chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                           1, ChunkSize> gt2;
              TASSIGN(gt2, GUbAddr);
              chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                           1, ChunkSize> ct2;
              TASSIGN(ct2,
                      CoeffUbAddr +
                          ((i * 4 + 2) * ChunkSize) *
                              static_cast<int32_t>(sizeof(float)));
              TADDS(ct2, gt2, -gv2);

              set_flag(PIPE_V, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
              auto gv3 = g_v_ub.GetValue(i * 4 + 3);
              chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                           1, ChunkSize> gt3;
              TASSIGN(gt3, GUbAddr);
              chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize,
                                           1, ChunkSize> ct3;
              TASSIGN(ct3,
                      CoeffUbAddr +
                          ((i * 4 + 3) * ChunkSize) *
                              static_cast<int32_t>(sizeof(float)));
              TADDS(ct3, gt3, -gv3);
            }

            TSUB(coeff_ub, qk_ub, coeff_ub);
            TMUL(coeff_ub, coeff_ub, msk_ub);
            TEXP(coeff_ub, coeff_ub);
            TEXP(g_v_ub, g_v_ub);

            chunk_gdn_pto::wait_cross_flag(0);

            chunk_gdn_pto::copy_gm_to_ub<half, half,
                1, 1, 1, HalfChunk, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                HalfChunk, ChunkSize, pto::PadValue::Zero>(
                workspace_qk_handle +
                    static_cast<int64_t>(cid) * WsQKSize +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                QKHalfUbAddr, 0, HalfChunk, ChunkSize);

            chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
            chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
            TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

            chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE2>(0);
            chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE2>(0);

            chunk_gdn_pto::copy_gm_to_ub<half, half,
                1, 1, 1, HalfChunk, HiddenSize,
                1, 1, 1, HiddenSize, 1,
                HalfChunk, HiddenSize, pto::PadValue::Zero>(
                workspace_qs_qkv_handle +
                    static_cast<int64_t>(cid) * WsQSSize +
                    static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                QSHalfUbAddr, 0, HalfChunk, HiddenSize);

            TMUL(qk_ub, qk_ub, coeff_ub);
            TMUL(qk_ub, qk_ub, msk_ub);
            TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

            chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
            chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
            chunk_gdn_pto::copy_ub_to_gm<half, half,
                1, 1, 1, HalfChunk, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                HalfChunk, ChunkSize>(
                workspace_qk_gated_handle +
                    static_cast<int64_t>(cid) * WsGatedSize +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                QKHalfUbAddr, 0, HalfChunk, ChunkSize);
            chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);

            chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
            chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
            TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);

            for (int32_t i = 0; i < HalfChunk / 4; ++i) {
              set_flag(PIPE_V, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
              auto gv0 = g_v_ub.GetValue(i * 4);
              chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                           1, HiddenSize> qsr0;
              TASSIGN(qsr0,
                      QSUbAddr +
                          (i * 4 * HiddenSize) *
                              static_cast<int32_t>(sizeof(float)));
              chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                           1, HiddenSize> qsd0;
              TASSIGN(qsd0,
                      QSUbAddr +
                          (i * 4 * HiddenSize) *
                              static_cast<int32_t>(sizeof(float)));
              TMULS(qsd0, qsr0, gv0);

              set_flag(PIPE_V, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
              auto gv1 = g_v_ub.GetValue(i * 4 + 1);
              chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                           1, HiddenSize> qsr1;
              TASSIGN(qsr1,
                      QSUbAddr +
                          ((i * 4 + 1) * HiddenSize) *
                              static_cast<int32_t>(sizeof(float)));
              chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                           1, HiddenSize> qsd1;
              TASSIGN(qsd1,
                      QSUbAddr +
                          ((i * 4 + 1) * HiddenSize) *
                              static_cast<int32_t>(sizeof(float)));
              TMULS(qsd1, qsr1, gv1);

              set_flag(PIPE_V, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
              auto gv2 = g_v_ub.GetValue(i * 4 + 2);
              chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                           1, HiddenSize> qsr2;
              TASSIGN(qsr2,
                      QSUbAddr +
                          ((i * 4 + 2) * HiddenSize) *
                              static_cast<int32_t>(sizeof(float)));
              chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                           1, HiddenSize> qsd2;
              TASSIGN(qsd2,
                      QSUbAddr +
                          ((i * 4 + 2) * HiddenSize) *
                              static_cast<int32_t>(sizeof(float)));
              TMULS(qsd2, qsr2, gv2);

              set_flag(PIPE_V, PIPE_S, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
              auto gv3 = g_v_ub.GetValue(i * 4 + 3);
              chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                           1, HiddenSize> qsr3;
              TASSIGN(qsr3,
                      QSUbAddr +
                          ((i * 4 + 3) * HiddenSize) *
                              static_cast<int32_t>(sizeof(float)));
              chunk_gdn_pto::TileUbDataND<float, 1, HiddenSize,
                                           1, HiddenSize> qsd3;
              TASSIGN(qsd3,
                      QSUbAddr +
                          ((i * 4 + 3) * HiddenSize) *
                              static_cast<int32_t>(sizeof(float)));
              TMULS(qsd3, qsr3, gv3);
            }

            chunk_gdn_pto::wait_cross_flag(2);

            chunk_gdn_pto::copy_gm_to_ub<half, half,
                1, 1, 1, HalfChunk, HiddenSize,
                1, 1, 1, HiddenSize, 1,
                HalfChunk, HiddenSize, pto::PadValue::Zero>(
                workspace_qs_qkv_handle +
                    static_cast<int64_t>(cid) * WsQSSize +
                    static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                OHalfUbAddr, 0, HalfChunk, HiddenSize);

            chunk_gdn_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V>(0);
            chunk_gdn_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V>(0);

            TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
            TADD(o_ub, qs_ub, o_ub);
            TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

            chunk_gdn_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3>(0);
            chunk_gdn_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3>(0);

            int64_t o_offset =
                (chunk_token_start * NumHeads + head_idx) *
                    static_cast<int64_t>(HiddenSize) +
                static_cast<int64_t>(vid) * HalfChunk *
                    NumHeads * HiddenSize;

            chunk_gdn_pto::copy_ub_to_gm<half, half,
                1, 1, 1, HalfChunk, HiddenSize,
                1, 1, 1, NumHeads * HiddenSize, 1,
                HalfChunk, HiddenSize>(
                O_handle + o_offset,
                OHalfUbAddr, 0, HalfChunk, HiddenSize);
          }
          gi++;
        }
        chunk_global_idx++;
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_chunk_o(
    __gm__ uint8_t *Q_handle, __gm__ uint8_t *K_handle,
    __gm__ uint8_t *V_handle, __gm__ uint8_t *S_handle,
    __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle,
    __gm__ uint8_t *workspace_qk, __gm__ uint8_t *workspace_qs_qkv,
    __gm__ uint8_t *workspace_qk_gated,
    __gm__ uint8_t *O_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  chunk_o_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(Q_handle),
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(S_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ float *>(Msk_handle),
      reinterpret_cast<__gm__ half *>(workspace_qk),
      reinterpret_cast<__gm__ half *>(workspace_qs_qkv),
      reinterpret_cast<__gm__ half *>(workspace_qk_gated),
      reinterpret_cast<__gm__ half *>(O_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *q, uint8_t *k, uint8_t *v, uint8_t *s, uint8_t *g_sum,
    uint8_t *mask,
    uint8_t *workspace_qk, uint8_t *workspace_qs_qkv,
    uint8_t *workspace_qk_gated,
    uint8_t *o,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_o<<<block_dim, nullptr, stream>>>(
      q, k, v, s, g_sum, mask,
      workspace_qk, workspace_qs_qkv, workspace_qk_gated,
      o,
      cu_seqlens,
      batch_size, seq_len, fftsAddr);
}
