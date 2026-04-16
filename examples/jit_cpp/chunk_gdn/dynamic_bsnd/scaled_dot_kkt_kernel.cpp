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
AICORE void kkt_kernel(
    __gm__ half *K_handle, __gm__ half *Beta_handle,
    __gm__ float *G_handle, __gm__ float *Msk_handle,
    __gm__ half *workspace_handle, __gm__ half *A_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t ChunkSquare = ChunkSize * ChunkSize;
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);

  constexpr int32_t GUbAddr      = 0;
  constexpr int32_t BetaHalfUbAddr = 512;
  constexpr int32_t BetaUbAddr   = 640;
  constexpr int32_t GvUbAddr     = 896;
  constexpr int32_t AUbAddr      = 1152;
  constexpr int32_t GRUbAddr     = 33920;
  constexpr int32_t GCUbAddr     = 34176;
  constexpr int32_t MskUbAddr    = 34688;
  constexpr int32_t GR2dUbAddr   = 67456;
  constexpr int32_t GC2dUbAddr   = 124800;
  constexpr int32_t CoeffUbAddr  = 157568;
  constexpr int32_t AUbHalfAddr  = GR2dUbAddr;
  constexpr int32_t GBlockUbAddr = AUbAddr;
  constexpr int32_t BetaBlockUbAddr = CoeffUbAddr;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * NumHeads;

  chunk_gdn_pto::TileMatL1<half, ChunkSize, HiddenSize,
                            ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a_l0;
  TASSIGN(a_l0, 0);

  chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  chunk_gdn_pto::TileUbDataND<half, 1, HalfChunk, 1, HalfChunk> beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk, 1, HalfChunk> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk, 1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> a_ub;
  TASSIGN(a_ub, AUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk, 1, HalfChunk> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  chunk_gdn_pto::TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> g_c_ub;
  TASSIGN(g_c_ub, GCUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> g_r_2d_ub;
  TASSIGN(g_r_2d_ub, GR2dUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> g_c_2d_ub;
  TASSIGN(g_c_2d_ub, GC2dUbAddr);
  chunk_gdn_pto::TileUbDataND<float, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  chunk_gdn_pto::TileUbDataND<half, HalfChunk, ChunkSize,
                               HalfChunk, ChunkSize> a_ub_half;
  TASSIGN(a_ub_half, AUbHalfAddr);

#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0;
       work_idx < (total_work + block_num - 1) / block_num; ++work_idx) {
    int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                  static_cast<int64_t>(cid);
    if (pid >= total_work) continue;

    int32_t head_idx = static_cast<int32_t>(pid % NumHeads);
    int64_t seq_idx = pid / NumHeads;

    int64_t bos, slen;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
    }
    int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

    for (int64_t ci = 0; ci < num_chunks; ++ci) {
      int32_t slot = static_cast<int32_t>(ci & 1);
      wait_flag_dev(2 + slot);
      pipe_barrier(PIPE_ALL);

      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      int64_t k_offset =
          ((bos + chunk_start) * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize);

      chunk_gdn_pto::copy_gm_to_l1<half, half,
          1, 1, 1, ChunkSize, HiddenSize,
          1, 1, 1, NumHeads * HiddenSize, 1,
          ChunkSize, HiddenSize>(
          K_handle + k_offset, 0, 0, valid_rows, HiddenSize);

      chunk_gdn_pto::gemm_v0<half, float,
          ChunkSize, ChunkSize, HiddenSize,
          ChunkSize, ChunkSize, HiddenSize,
          KTail, false, true>(k_l1, k_l1, a_l0, true);

      chunk_gdn_pto::copy_l0c_to_gm<half, float,
          1, 1, 1, ChunkSize, ChunkSize,
          1, 1, 1, ChunkSize, 1,
          ChunkSize, ChunkSize>(
          workspace_handle +
              (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare,
          0, 0, ChunkSize, ChunkSize);

      chunk_gdn_pto::set_cross_flag<PIPE_FIX>(slot, 2);
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

  chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(2, 2);
  chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(3, 2);

  for (int64_t work_idx = 0;
       work_idx < (total_work + block_num - 1) / block_num; ++work_idx) {
    int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                  static_cast<int64_t>(cid);
    if (pid >= total_work) continue;

    int32_t head_idx = static_cast<int32_t>(pid % NumHeads);
    int64_t seq_idx = pid / NumHeads;

    int64_t bos, slen;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
    }
    int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

    for (int64_t ci = 0; ci < num_chunks; ++ci) {
      int32_t slot = static_cast<int32_t>(ci & 1);
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_valid =
          valid_rows > row_offset
              ? (valid_rows - row_offset < HalfChunk
                     ? valid_rows - row_offset
                     : HalfChunk)
              : 0;

      if (local_valid > 0) {
        chunk_gdn_pto::copy_gm_to_ub<float, float,
            1, 1, 1, 1, ChunkSize,
            1, 1, 1, 1, 1,
            1, ChunkSize, pto::PadValue::Zero>(
            G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                (bos + chunk_start),
            GUbAddr, 0, 1, valid_rows);

        chunk_gdn_pto::copy_gm_to_ub<half, half,
            1, 1, 1, 1, HalfChunk,
            1, 1, 1, 1, 1,
            1, HalfChunk, pto::PadValue::Zero>(
            Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                (bos + chunk_start + row_offset),
            BetaHalfUbAddr, 0, 1, local_valid);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
        chunk_gdn_pto::TileUbDataND<float, 1, HalfChunk, 1, HalfChunk>
            g_ub_temp;
        TASSIGN(g_ub_temp,
                GUbAddr + row_offset *
                              static_cast<int32_t>(sizeof(float)));
        TMOV(g_v_ub, g_ub_temp);
        pipe_barrier(PIPE_V);

        TLOG(beta_ub, beta_ub);
        pipe_barrier(PIPE_V);
        TADD(g_v_ub, g_v_ub, beta_ub);
        pipe_barrier(PIPE_V);
        TMOV(g_r_ub, g_v_ub);
        TMOV(g_c_ub, g_ub);
        pipe_barrier(PIPE_V);

        chunk_gdn_pto::TileUbDataDN<float, HalfChunk, 1,
                                     HalfChunk, 1> g_r_ub_temp;
        TASSIGN(g_r_ub_temp, GRUbAddr);
        TROWEXPAND(g_r_2d_ub, g_r_ub_temp);
        TCOLEXPAND(g_c_2d_ub, g_c_ub);
        pipe_barrier(PIPE_V);
        TSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);
        pipe_barrier(PIPE_V);
        TMINS(coeff_ub, coeff_ub, 0.0f);
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);
      }

      wait_flag_dev(slot);
      pipe_barrier(PIPE_ALL);

      if (local_valid > 0) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        chunk_gdn_pto::copy_gm_to_ub<half, half,
            1, 1, 1, HalfChunk, ChunkSize,
            1, 1, 1, ChunkSize, 1,
            HalfChunk, ChunkSize, pto::PadValue::Zero>(
            workspace_handle +
                (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
            AUbHalfAddr, 0, HalfChunk, ChunkSize);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TCVT(a_ub, a_ub_half, pto::RoundMode::CAST_NONE);
        TMUL(a_ub, a_ub, coeff_ub);
        TMUL(a_ub, a_ub, msk_ub);
        TCVT(a_ub_half, a_ub, pto::RoundMode::CAST_NONE);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        int64_t a_gm_offset =
            ((bos + chunk_start + row_offset) * NumHeads +
             head_idx) *
            static_cast<int64_t>(ChunkSize);

        chunk_gdn_pto::copy_ub_to_gm<half, half,
            1, 1, 1, HalfChunk, ChunkSize,
            1, 1, 1, NumHeads * ChunkSize, 1,
            HalfChunk, ChunkSize>(
            A_handle + a_gm_offset, AUbHalfAddr, 0,
            local_valid, ChunkSize);
      }

      pipe_barrier(PIPE_ALL);
      chunk_gdn_pto::set_cross_flag<PIPE_MTE3>(2 + slot, 2);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_scaled_dot_kkt(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *Beta_handle,
    __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle,
    __gm__ uint8_t *workspace_handle, __gm__ uint8_t *A_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  kkt_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ float *>(Msk_handle),
      reinterpret_cast<__gm__ half *>(workspace_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *K_handle, uint8_t *Beta_handle,
    uint8_t *G_handle, uint8_t *Msk_handle,
    uint8_t *workspace_handle, uint8_t *A_handle,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_scaled_dot_kkt<<<block_dim, nullptr, stream>>>(
      K_handle, Beta_handle, G_handle, Msk_handle,
      workspace_handle, A_handle, cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
