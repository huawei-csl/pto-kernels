#include "common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

template <int32_t NumHeads, int32_t ChunkSize>
AICORE void cumsum_kernel(
    __gm__ float *g_ptr, __gm__ float *g_sum_ptr,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (vid != 0) return;

  constexpr int32_t HeadTileCols = ((NumHeads + 7) / 8) * 8;
  constexpr int32_t BlockBytes = ChunkSize * HeadTileCols *
                                 static_cast<int32_t>(sizeof(float));
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t SUbAddr = BlockBytes;

  chunk_gdn_pto::TileUbDataND<float, ChunkSize, HeadTileCols,
                               ChunkSize, HeadTileCols> g_block_ub;
  TASSIGN(g_block_ub, GUbAddr);
  chunk_gdn_pto::TileUbDataND<float, ChunkSize, HeadTileCols,
                               ChunkSize, HeadTileCols> s_block_ub;
  TASSIGN(s_block_ub, SUbAddr);

  int64_t num_seqs = batch_size;

  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t total_chunks = num_seqs * chunks_per_seq;

    for (int64_t gi = static_cast<int64_t>(cid); gi < total_chunks;
         gi += static_cast<int64_t>(block_num)) {
      int64_t seq_idx = gi / chunks_per_seq;
      int64_t local_chunk = gi % chunks_per_seq;
      int64_t bos = seq_idx * seq_len;
      int64_t chunk_start = bos + local_chunk * ChunkSize;
      int64_t remaining = seq_len - local_chunk * ChunkSize;
      int32_t valid = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      chunk_gdn_pto::copy_gm_to_ub<float, float,
          1, 1, 1, ChunkSize, HeadTileCols,
          1, 1, 1, NumHeads, 1,
          ChunkSize, HeadTileCols, pto::PadValue::Zero>(
          g_ptr + chunk_start * NumHeads, GUbAddr, 0, valid, NumHeads);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

      for (int32_t h = 0; h < NumHeads; ++h) {
        float acc = g_block_ub.GetValue(h);
        s_block_ub.SetValue(h, acc);
        for (int32_t i = 1; i < valid; ++i) {
          acc += g_block_ub.GetValue(i * HeadTileCols + h);
          s_block_ub.SetValue(i * HeadTileCols + h, acc);
        }
        for (int32_t i = valid; i < ChunkSize; ++i) {
          s_block_ub.SetValue(i * HeadTileCols + h, 0.0f);
        }
      }

      pipe_barrier(PIPE_ALL);

      chunk_gdn_pto::copy_ub_to_gm<float, float,
          1, 1, 1, ChunkSize, HeadTileCols,
          1, 1, 1, NumHeads, 1,
          ChunkSize, HeadTileCols>(
          g_sum_ptr + chunk_start * NumHeads, SUbAddr, 0, valid, NumHeads);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
  } else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t c = 0; c < nc; ++c) {
        if (gi % static_cast<int64_t>(block_num) ==
            static_cast<int64_t>(cid)) {
          int64_t chunk_start = bos + c * ChunkSize;
          int64_t remaining = slen - c * ChunkSize;
          int32_t valid = static_cast<int32_t>(
              remaining < ChunkSize ? remaining : ChunkSize);

          chunk_gdn_pto::copy_gm_to_ub<float, float,
              1, 1, 1, ChunkSize, HeadTileCols,
              1, 1, 1, NumHeads, 1,
              ChunkSize, HeadTileCols, pto::PadValue::Zero>(
              g_ptr + chunk_start * NumHeads,
              GUbAddr, 0, valid, NumHeads);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

          for (int32_t h = 0; h < NumHeads; ++h) {
            float acc = g_block_ub.GetValue(h);
            s_block_ub.SetValue(h, acc);
            for (int32_t i = 1; i < valid; ++i) {
              acc += g_block_ub.GetValue(i * HeadTileCols + h);
              s_block_ub.SetValue(i * HeadTileCols + h, acc);
            }
            for (int32_t i = valid; i < ChunkSize; ++i) {
              s_block_ub.SetValue(i * HeadTileCols + h, 0.0f);
            }
          }

          pipe_barrier(PIPE_ALL);

          chunk_gdn_pto::copy_ub_to_gm<float, float,
              1, 1, 1, ChunkSize, HeadTileCols,
              1, 1, 1, NumHeads, 1,
              ChunkSize, HeadTileCols>(
              g_sum_ptr + chunk_start * NumHeads,
              SUbAddr, 0, valid, NumHeads);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        gi++;
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_cumsum(
    __gm__ uint8_t *g_ptr, __gm__ uint8_t *g_sum_ptr,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  cumsum_kernel<GDN_H, GDN_C>(
      reinterpret_cast<__gm__ float *>(g_ptr),
      reinterpret_cast<__gm__ float *>(g_sum_ptr),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *g_ptr, uint8_t *g_sum_ptr, uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_cumsum<<<block_dim, nullptr, stream>>>(
      g_ptr, g_sum_ptr, cu_seqlens, batch_size, seq_len, fftsAddr);
}
