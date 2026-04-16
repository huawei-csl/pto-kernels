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

#ifdef __CCE_AICORE__

namespace {

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T, int32_t Rows, int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using VecTile =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor, Rows,
              Cols, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                             pto::BLayout::RowMajor, pto::DYNAMIC,
                             pto::DYNAMIC, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

} // namespace

#endif

template <int32_t NumHeads, int32_t ChunkSize>
AICORE void cumsum_kernel(
    __gm__ float *g_ptr, __gm__ float *g_sum_ptr, __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, uint64_t ffts_addr) {
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();
  set_ffts_base_addr(ffts_addr);

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t subblock_num = static_cast<int64_t>(get_subblockdim());
  const int64_t num_vec_workers =
      static_cast<int64_t>(block_num) * subblock_num;
  const int64_t worker_id =
      static_cast<int64_t>(cid) * subblock_num + static_cast<int64_t>(vid);

  constexpr int32_t HeadTileCols = ((NumHeads + 7) / 8) * 8;
  constexpr int32_t BlockBytes =
      ChunkSize * HeadTileCols * static_cast<int32_t>(sizeof(float));
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t SUbAddr = BlockBytes;
  constexpr int32_t AccUbAddr = SUbAddr + BlockBytes;

  using BlockTile =
      VecTile<float, ChunkSize, HeadTileCols, pto::PadValue::Zero>;
  using BlockLoadTile =
      DynVecTile<float, ChunkSize, HeadTileCols, pto::PadValue::Zero>;
  using RowTile = VecTile<float, 1, HeadTileCols, pto::PadValue::Zero>;

  BlockTile g_block_ub;
  TASSIGN(g_block_ub, GUbAddr);
  BlockTile s_block_ub;
  TASSIGN(s_block_ub, SUbAddr);
  RowTile acc_ub;
  TASSIGN(acc_ub, AccUbAddr);

  int64_t num_seqs = batch_size;

  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t total_chunks = num_seqs * chunks_per_seq;

    for (int64_t gi = worker_id; gi < total_chunks; gi += num_vec_workers) {
      int64_t seq_idx = gi / chunks_per_seq;
      int64_t local_chunk = gi % chunks_per_seq;
      int64_t bos = seq_idx * seq_len;
      int64_t chunk_start = bos + local_chunk * ChunkSize;
      int64_t remaining = seq_len - local_chunk * ChunkSize;
      int32_t valid =
          static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);

      GmShape2D shape(valid, NumHeads);
      GmStride2D stride(NumHeads);
      GmTensor2D<float> g_global(g_ptr + chunk_start * NumHeads, shape, stride);
      BlockLoadTile g_block_load(valid, NumHeads);
      TASSIGN(g_block_load, GUbAddr);
      TLOAD(g_block_load, g_global);
      if (valid != ChunkSize || NumHeads != HeadTileCols) {
        TFILLPAD_INPLACE(g_block_ub, g_block_load);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // The original algorithm computes a per-head prefix sum over the chunk
      // dimension. Each row is one token and each column is one attention head.
      RowTile g_row_0;
      TASSIGN(g_row_0, GUbAddr);
      RowTile s_row_0;
      TASSIGN(s_row_0, SUbAddr);
      TMOV(acc_ub, g_row_0);
      TMOV(s_row_0, g_row_0);
      pipe_barrier(PIPE_V);

      for (int32_t i = 1; i < valid; ++i) {
        RowTile g_row_i;
        TASSIGN(
            g_row_i,
            GUbAddr + i * HeadTileCols * static_cast<int32_t>(sizeof(float)));
        RowTile s_row_i;
        TASSIGN(
            s_row_i,
            SUbAddr + i * HeadTileCols * static_cast<int32_t>(sizeof(float)));
        TADD(s_row_i, acc_ub, g_row_i);
        pipe_barrier(PIPE_V);
        TMOV(acc_ub, s_row_i);
        pipe_barrier(PIPE_V);
      }

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      GmTensor2D<float> g_sum_global(g_sum_ptr + chunk_start * NumHeads, shape,
                                     stride);
      BlockLoadTile s_block_store(valid, NumHeads);
      TASSIGN(s_block_store, SUbAddr);
      TSTORE(g_sum_global, s_block_store);
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
        if (gi % num_vec_workers == worker_id) {
          int64_t chunk_start = bos + c * ChunkSize;
          int64_t remaining = slen - c * ChunkSize;
          int32_t valid =
              static_cast<int32_t>(remaining < ChunkSize ? remaining : ChunkSize);

          GmShape2D shape(valid, NumHeads);
          GmStride2D stride(NumHeads);
          GmTensor2D<float> g_global(g_ptr + chunk_start * NumHeads, shape,
                                     stride);
          BlockLoadTile g_block_load(valid, NumHeads);
          TASSIGN(g_block_load, GUbAddr);
          TLOAD(g_block_load, g_global);
          if (valid != ChunkSize || NumHeads != HeadTileCols) {
            TFILLPAD_INPLACE(g_block_ub, g_block_load);
          }

          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

          RowTile g_row_0;
          TASSIGN(g_row_0, GUbAddr);
          RowTile s_row_0;
          TASSIGN(s_row_0, SUbAddr);
          TMOV(acc_ub, g_row_0);
          TMOV(s_row_0, g_row_0);
          pipe_barrier(PIPE_V);

          for (int32_t i = 1; i < valid; ++i) {
            RowTile g_row_i;
            TASSIGN(
                g_row_i,
                GUbAddr + i * HeadTileCols * static_cast<int32_t>(sizeof(float)));
            RowTile s_row_i;
            TASSIGN(
                s_row_i,
                SUbAddr + i * HeadTileCols * static_cast<int32_t>(sizeof(float)));
            TADD(s_row_i, acc_ub, g_row_i);
            pipe_barrier(PIPE_V);
            TMOV(acc_ub, s_row_i);
            pipe_barrier(PIPE_V);
          }

          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

          GmTensor2D<float> g_sum_global(g_sum_ptr + chunk_start * NumHeads,
                                         shape, stride);
          BlockLoadTile s_block_store(valid, NumHeads);
          TASSIGN(s_block_store, SUbAddr);
          TSTORE(g_sum_global, s_block_store);
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
