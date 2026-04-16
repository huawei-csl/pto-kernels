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
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);
  constexpr uint32_t CTail =
      (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

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
  constexpr int32_t OUbAddr      = QKUbAddr;

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
    bool first_cube_iter = true;

    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      if (!first_cube_iter) wait_flag_dev(3);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

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

      wait_flag_dev(1);

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

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
      first_cube_iter = false;
    }
  } else {
    int64_t gi = 0;
    int64_t chunk_global_idx = 0;
    bool first_cube_iter_v = true;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            if (!first_cube_iter_v) wait_flag_dev(3);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

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

            wait_flag_dev(1);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

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
            first_cube_iter_v = false;
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

  chunk_gdn_pto::copy_gm_to_ub<float, float,
      1, 1, 1, HalfChunk, ChunkSize,
      1, 1, 1, ChunkSize, 1,
      HalfChunk, ChunkSize, pto::PadValue::Zero>(
      Msk_handle +
          static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
      MskUbAddr, 0, HalfChunk, ChunkSize);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

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

      // G is pre-transposed to [H, total_tokens] float — contiguous per head
      chunk_gdn_pto::copy_gm_to_ub<float, float,
          1, 1, 1, 1, ChunkSize,
          1, 1, 1, 1, 1,
          1, ChunkSize, pto::PadValue::Zero>(
          G_handle + static_cast<int64_t>(head_idx) * total_tokens
                   + chunk_token_start,
          GUbAddr, 0, 1, valid_rows);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk,
                                   1, HalfChunk> g_ub_temp_0;
      TASSIGN(g_ub_temp_0,
              GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                            static_cast<int32_t>(sizeof(float)));
      TMOV(g_v_ub, g_ub_temp_0);

      chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                                   HalfChunk, ChunkSize> g_r_2d;
      TASSIGN(g_r_2d, QSUbAddr);
      chunk_gdn_pto::TileUbDataDN<float, HalfChunk, 1,
                                   HalfChunk, 1> g_v_col;
      TASSIGN(g_v_col, GvUbAddr);
      TROWEXPAND(g_r_2d, g_v_col);
      TCOLEXPAND(coeff_ub, g_ub);
      TSUB(coeff_ub, g_r_2d, coeff_ub);
      pipe_barrier(PIPE_V);
      TMINS(coeff_ub, coeff_ub, 0.0f);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TMUL(coeff_ub, coeff_ub, msk_ub);
      TEXP(g_v_ub, g_v_ub);

      wait_flag_dev(0);

      chunk_gdn_pto::copy_gm_to_ub<half, half,
          1, 1, 1, HalfChunk, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          HalfChunk, ChunkSize, pto::PadValue::Zero>(
          workspace_qk_handle +
              static_cast<int64_t>(cid) * WsQKSize +
              static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
          QKHalfUbAddr, 0, HalfChunk, ChunkSize);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

      chunk_gdn_pto::copy_gm_to_ub<half, half,
          1, 1, 1, HalfChunk, HiddenSize,
          1, 1, 1, HiddenSize, 1,
          HalfChunk, HiddenSize, pto::PadValue::Zero>(
          workspace_qs_qkv_handle +
              static_cast<int64_t>(cid) * WsQSSize +
              static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
          QSHalfUbAddr, 0, HalfChunk, HiddenSize);

      TMUL(qk_ub, qk_ub, coeff_ub);
      TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      chunk_gdn_pto::copy_ub_to_gm<half, half,
          1, 1, 1, HalfChunk, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          HalfChunk, ChunkSize>(
          workspace_qk_gated_handle +
              static_cast<int64_t>(cid) * WsGatedSize +
              static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
          QKHalfUbAddr, 0, HalfChunk, ChunkSize);
      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);
      chunk_gdn_pto::TileUbDataND<float, HalfChunk, HiddenSize,
                                   HalfChunk, HiddenSize> g_exp_2d;
      TASSIGN(g_exp_2d, CoeffUbAddr);
      chunk_gdn_pto::TileUbDataDN<float, HalfChunk, 1,
                                   HalfChunk, 1> g_v_col2;
      TASSIGN(g_v_col2, GvUbAddr);
      TROWEXPAND(g_exp_2d, g_v_col2);
      pipe_barrier(PIPE_V);
      TMUL(qs_ub, qs_ub, g_exp_2d);

      wait_flag_dev(2);

      chunk_gdn_pto::copy_gm_to_ub<half, half,
          1, 1, 1, HalfChunk, HiddenSize,
          1, 1, 1, HiddenSize, 1,
          HalfChunk, HiddenSize, pto::PadValue::Zero>(
          workspace_qs_qkv_handle +
              static_cast<int64_t>(cid) * WsQSSize +
              static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
          OHalfUbAddr, 0, HalfChunk, HiddenSize);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
      TADD(o_ub, qs_ub, o_ub);
      TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

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

      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(3, 2);
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

            // G is pre-transposed to [H, total_tokens] float
            chunk_gdn_pto::copy_gm_to_ub<float, float,
                1, 1, 1, 1, ChunkSize,
                1, 1, 1, 1, 1,
                1, ChunkSize, pto::PadValue::Zero>(
                G_handle + static_cast<int64_t>(head_idx) * total_tokens
                         + chunk_token_start,
                GUbAddr, 0, 1, valid_rows);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk,
                                         1, HalfChunk> g_ub_temp_v;
            TASSIGN(g_ub_temp_v,
                    GUbAddr +
                        static_cast<int32_t>(vid) * HalfChunk *
                            static_cast<int32_t>(sizeof(float)));
            TMOV(g_v_ub, g_ub_temp_v);

            chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                                         HalfChunk, ChunkSize> g_r_2d_v;
            TASSIGN(g_r_2d_v, QSUbAddr);
            chunk_gdn_pto::TileUbDataDN<float, HalfChunk, 1,
                                         HalfChunk, 1> g_v_col_v;
            TASSIGN(g_v_col_v, GvUbAddr);
            TROWEXPAND(g_r_2d_v, g_v_col_v);
            TCOLEXPAND(coeff_ub, g_ub);
            TSUB(coeff_ub, g_r_2d_v, coeff_ub);
            pipe_barrier(PIPE_V);
            TMINS(coeff_ub, coeff_ub, 0.0f);
            pipe_barrier(PIPE_V);
            TEXP(coeff_ub, coeff_ub);
            pipe_barrier(PIPE_V);
            TMUL(coeff_ub, coeff_ub, msk_ub);
            TEXP(g_v_ub, g_v_ub);

            wait_flag_dev(0);

            chunk_gdn_pto::copy_gm_to_ub<half, half,
                1, 1, 1, HalfChunk, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                HalfChunk, ChunkSize, pto::PadValue::Zero>(
                workspace_qk_handle +
                    static_cast<int64_t>(cid) * WsQKSize +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                QKHalfUbAddr, 0, HalfChunk, ChunkSize);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            chunk_gdn_pto::copy_gm_to_ub<half, half,
                1, 1, 1, HalfChunk, HiddenSize,
                1, 1, 1, HiddenSize, 1,
                HalfChunk, HiddenSize, pto::PadValue::Zero>(
                workspace_qs_qkv_handle +
                    static_cast<int64_t>(cid) * WsQSSize +
                    static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                QSHalfUbAddr, 0, HalfChunk, HiddenSize);

            TMUL(qk_ub, qk_ub, coeff_ub);
            TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            chunk_gdn_pto::copy_ub_to_gm<half, half,
                1, 1, 1, HalfChunk, ChunkSize,
                1, 1, 1, ChunkSize, 1,
                HalfChunk, ChunkSize>(
                workspace_qk_gated_handle +
                    static_cast<int64_t>(cid) * WsGatedSize +
                    static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                QKHalfUbAddr, 0, HalfChunk, ChunkSize);
            chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(1, 2);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);

            chunk_gdn_pto::TileUbDataND<float, HalfChunk, HiddenSize,
                                         HalfChunk, HiddenSize> g_exp_2d_v;
            TASSIGN(g_exp_2d_v, CoeffUbAddr);
            chunk_gdn_pto::TileUbDataDN<float, HalfChunk, 1,
                                         HalfChunk, 1> g_v_col2_v;
            TASSIGN(g_v_col2_v, GvUbAddr);
            TROWEXPAND(g_exp_2d_v, g_v_col2_v);
            pipe_barrier(PIPE_V);
            TMUL(qs_ub, qs_ub, g_exp_2d_v);

            wait_flag_dev(2);

            chunk_gdn_pto::copy_gm_to_ub<half, half,
                1, 1, 1, HalfChunk, HiddenSize,
                1, 1, 1, HiddenSize, 1,
                HalfChunk, HiddenSize, pto::PadValue::Zero>(
                workspace_qs_qkv_handle +
                    static_cast<int64_t>(cid) * WsQSSize +
                    static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                OHalfUbAddr, 0, HalfChunk, HiddenSize);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
            TADD(o_ub, qs_ub, o_ub);
            TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

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

            chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(3, 2);
          }
          gi++;
        }
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
    int64_t total_tokens,
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
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *q, uint8_t *k, uint8_t *v, uint8_t *s, uint8_t *g_sum,
    uint8_t *mask,
    uint8_t *workspace_qk, uint8_t *workspace_qs_qkv,
    uint8_t *workspace_qk_gated,
    uint8_t *o,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_o<<<block_dim, nullptr, stream>>>(
      q, k, v, s, g_sum, mask,
      workspace_qk, workspace_qs_qkv, workspace_qk_gated,
      o,
      cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
