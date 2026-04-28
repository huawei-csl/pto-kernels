#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

#include "gdn_pto_shared.h"
#include "gdn_seq_info.h"

using namespace pto;

#ifndef GDN_H
#define GDN_H 2
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

AICORE inline uint32_t GdnMinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void main_kernel(__gm__ half *q, __gm__ half *k, __gm__ half *v,
                        __gm__ half *s_packed, __gm__ float *g_packed,
                        __gm__ half *workspace_qk, __gm__ half *workspace_qs_qkv,
                        __gm__ half *workspace_qk_gated, __gm__ half *o,
                        __gm__ int32_t *cu_seqlens, int64_t batch_size,
                        int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;

  constexpr int32_t QL1Addr = 0;
  constexpr int32_t KL1Addr = 32768;
  constexpr int32_t SL1Addr = 65536;
  constexpr int32_t QKL1Addr = 98304;
  constexpr int32_t VL1Addr = 131072;

  constexpr int32_t GUbAddr = 0;
  constexpr int32_t MaskUbAddr = GUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t QKUbAddr = MaskUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t GvUbAddr = QKUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t CoeffUbAddr = GvUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t QKHalfUbAddr = CoeffUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t QSHalfUbAddr = QKHalfUbAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t QSUbAddr = QSHalfUbAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t OHalfUbAddr = QSUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t OUbAddr = MaskUbAddr;

  using ChunkGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using ChunkGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using ChunkGlobalDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using PackedSquareDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using PackedState =
      GlobalTensor<half, TileShape2D<half, HiddenSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, HiddenSize, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedHiddenHalf =
      GlobalTensor<half, TileShape2D<half, HalfChunk, HiddenSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedSquareHalf =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using PackedGGlobal =
      GlobalTensor<float, TileShape2D<float, 1, ChunkSize, Layout::ND>,
                   BaseShape2D<float, 1, ChunkSize, Layout::ND>, Layout::ND>;
  using OutGlobalDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;

  using ChunkL1Dyn = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                          DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  using SquareL1Dyn = Tile<TileType::Mat, half, ChunkSize, ChunkSize, BLayout::ColMajor,
                           DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  using GUb = Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, DYNAMIC,
                   DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using GHalfUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor, DYNAMIC,
                       DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using QKUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using QKHalfUb = GdnUbND<half, HalfChunk, ChunkSize>;
  using QSHalfUb = GdnUbND<half, HalfChunk, HiddenSize>;
  using QSUb = GdnUbND<float, HalfChunk, HiddenSize>;
  using OHalfUb = GdnUbND<half, HalfChunk, HiddenSize>;
  using OUb = GdnUbND<float, HalfChunk, HiddenSize>;
  using CoeffUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using MaskUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using GColUb = GdnUbDN<float, HalfChunk, 1>;
  using GRowUb = GdnUbND<float, 1, ChunkSize>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * NumHeads;

  GdnL1Mat<half, ChunkSize, HiddenSize> q_l1;
  GdnL1Mat<half, ChunkSize, HiddenSize> k_l1;
  GdnL1Mat<half, HiddenSize, HiddenSize> s_l1;
  GdnL1Mat<half, ChunkSize, ChunkSize> qk_l1;
  GdnL1Mat<half, ChunkSize, HiddenSize> v_l1;
  TASSIGN(q_l1, QL1Addr);
  TASSIGN(k_l1, KL1Addr);
  TASSIGN(s_l1, SL1Addr);
  TASSIGN(qk_l1, QKL1Addr);
  TASSIGN(v_l1, VL1Addr);

  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> qk_l0;
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qs_l0;
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qkv_l0;
  TASSIGN(qk_l0, 0);
  TASSIGN(qs_l0, 65536);
  TASSIGN(qkv_l0, 0);

  GUb g_ub(1, ChunkSize);
  MaskUb msk_ub;
  QKUb qk_ub;
  GHalfUb g_v_ub(1, HalfChunk);
  CoeffUb coeff_ub;
  QKHalfUb qk_half_ub;
  QSHalfUb qs_half_ub;
  QSUb qs_ub;
  OHalfUb o_half_ub;
  OUb o_ub;
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(msk_ub, MaskUbAddr);
  TASSIGN(qk_ub, QKUbAddr);
  TASSIGN(g_v_ub, GvUbAddr);
  TASSIGN(coeff_ub, CoeffUbAddr);
  TASSIGN(qk_half_ub, QKHalfUbAddr);
  TASSIGN(qs_half_ub, QSHalfUbAddr);
  TASSIGN(qs_ub, QSUbAddr);
  TASSIGN(o_half_ub, OHalfUbAddr);
  TASSIGN(o_ub, OUbAddr);

#if defined(__DAV_C220_CUBE__)
  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }
    const uint32_t head_idx = static_cast<uint32_t>(pid % NumHeads);
    const uint32_t seq_idx = static_cast<uint32_t>(pid / NumHeads);
    const GdnBsndSeqInfo seq = GetGdnBsndSeqInfo(
        seq_idx, head_idx, NumHeads, HiddenSize, ChunkSize,
        static_cast<uint32_t>(fixed_seq_len), cu_seqlens);
    const uint32_t chunk_num = GdnDivCeilU32(seq.seq_len, ChunkSize);

    for (uint32_t chunk_idx = 0; chunk_idx < chunk_num; ++chunk_idx) {
      const uint32_t row_start = chunk_idx * ChunkSize;
      const uint32_t valid_rows = GdnMinU32(
          static_cast<uint32_t>(seq.seq_len - row_start),
          static_cast<uint32_t>(ChunkSize));
      const int32_t token_offset =
          static_cast<int32_t>(seq.token_base_offset + row_start * seq.row_stride);
      const int32_t chunk_base =
          static_cast<int32_t>((seq.chunk_offset + chunk_idx) * NumHeads + head_idx);
      const int32_t square_offset = chunk_base * ChunkSquareElems;
      const int32_t hidden_offset = chunk_base * ChunkHiddenElems;

      {
        ChunkL1Dyn q_dyn(valid_rows, HiddenSize);
        ChunkL1Dyn k_dyn(valid_rows, HiddenSize);
        TASSIGN(q_dyn, QL1Addr);
        TASSIGN(k_dyn, KL1Addr);
        ChunkGlobalDyn q_global(
            q + token_offset,
            {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
            {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
        ChunkGlobalDyn k_global(
            k + token_offset,
            {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
            {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
        TLOAD(q_dyn, q_global);
        TLOAD(k_dyn, k_global);
        pipe_barrier(PIPE_ALL);
        GdnMatmulL1<ChunkSize, ChunkSize, HiddenSize, false, true>(qk_l0, q_l1, k_l1,
                                                                   true);
        PackedSquareDyn qk_global(
            workspace_qk + square_offset,
            {1, 1, 1, static_cast<int32_t>(valid_rows), ChunkSize},
            {1, 1, 1, ChunkSize, 1});
        TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> qk_tail(valid_rows,
                                                                       ChunkSize);
        TASSIGN(qk_tail, 0);
        TSTORE(qk_global, qk_tail);
        pipe_barrier(PIPE_ALL);
      }

      {
        ChunkL1Dyn q_dyn(valid_rows, HiddenSize);
        TASSIGN(q_dyn, QL1Addr);
        ChunkGlobalDyn q_global(
            q + token_offset,
            {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
            {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
        PackedState s_global(s_packed + chunk_base * HiddenSize * HiddenSize);
        TLOAD(q_dyn, q_global);
        TLOAD(s_l1, s_global);
        pipe_barrier(PIPE_ALL);
        GdnMatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(qs_l0, q_l1,
                                                                     s_l1, true);
        ChunkGlobalDyn qs_global(
            workspace_qs_qkv + hidden_offset,
            {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
            {1, 1, 1, HiddenSize, 1});
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> qs_tail(valid_rows,
                                                                         HiddenSize);
        TASSIGN(qs_tail, 65536);
        TSTORE(qs_global, qs_tail);
        pipe_barrier(PIPE_ALL);
      }

      pipe_barrier(PIPE_ALL);
      GdnSetCrossFlag<PIPE_FIX>(0, 2);
      GdnWaitCrossFlag(1);
      pipe_barrier(PIPE_ALL);

      {
        SquareL1Dyn qk_dyn(valid_rows, ChunkSize);
        ChunkL1Dyn v_dyn(valid_rows, HiddenSize);
        TASSIGN(qk_dyn, QKL1Addr);
        TASSIGN(v_dyn, VL1Addr);
        PackedSquareDyn qk_global(
            workspace_qk_gated + square_offset,
            {1, 1, 1, static_cast<int32_t>(valid_rows), ChunkSize},
            {1, 1, 1, ChunkSize, 1});
        ChunkGlobalDyn v_global(
            v + token_offset,
            {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
            {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
        TLOAD(qk_dyn, qk_global);
        TLOAD(v_dyn, v_global);
        pipe_barrier(PIPE_ALL);
        GdnMatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(qkv_l0, qk_l1,
                                                                    v_l1, true);
        ChunkGlobalDyn qkv_global(
            workspace_qs_qkv + hidden_offset,
            {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
            {1, 1, 1, HiddenSize, 1});
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> qkv_tail(valid_rows,
                                                                          HiddenSize);
        TASSIGN(qkv_tail, 0);
        TSTORE(qkv_global, qkv_tail);
        pipe_barrier(PIPE_ALL);
      }

      pipe_barrier(PIPE_ALL);
      GdnSetCrossFlag<PIPE_FIX>(2, 2);
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }
    const uint32_t head_idx = static_cast<uint32_t>(pid % NumHeads);
    const uint32_t seq_idx = static_cast<uint32_t>(pid / NumHeads);
    const GdnBsndSeqInfo seq = GetGdnBsndSeqInfo(
        seq_idx, head_idx, NumHeads, HiddenSize, ChunkSize,
        static_cast<uint32_t>(fixed_seq_len), cu_seqlens);
    const uint32_t chunk_num = GdnDivCeilU32(seq.seq_len, ChunkSize);

    for (uint32_t chunk_idx = 0; chunk_idx < chunk_num; ++chunk_idx) {
      const uint32_t row_start = chunk_idx * ChunkSize;
      const uint32_t valid_rows = GdnMinU32(
          static_cast<uint32_t>(seq.seq_len - row_start),
          static_cast<uint32_t>(ChunkSize));
      const uint32_t row_offset = static_cast<uint32_t>(vid) * HalfChunk;
      const uint32_t local_rows =
          valid_rows > row_offset
              ? GdnMinU32(static_cast<uint32_t>(valid_rows - row_offset),
                          static_cast<uint32_t>(HalfChunk))
              : 0;
      const int32_t chunk_base =
          static_cast<int32_t>((seq.chunk_offset + chunk_idx) * NumHeads + head_idx);
      const int32_t square_offset = chunk_base * ChunkSquareElems;
      const int32_t hidden_offset = chunk_base * ChunkHiddenElems;

      if (local_rows == 0) {
        GdnWaitCrossFlag(0);
        pipe_barrier(PIPE_ALL);
        GdnSetCrossFlag<PIPE_MTE3>(1, 2);
        GdnWaitCrossFlag(2);
        pipe_barrier(PIPE_ALL);
        continue;
      }

      PackedGGlobal g_global(g_packed + chunk_base * ChunkSize);
      TLOAD(g_ub, g_global);
      pipe_barrier(PIPE_ALL);

      for (uint32_t r = 0; r < HalfChunk; ++r) {
        const uint32_t global_r = row_offset + r;
        for (uint32_t c = 0; c < static_cast<uint32_t>(ChunkSize); ++c) {
          const bool keep = (global_r < valid_rows) && (c < valid_rows) &&
                            (global_r >= c);
          qk_half_ub.SetValue(r * ChunkSize + c,
                              keep ? static_cast<half>(1.0f)
                                   : static_cast<half>(0.0f));
        }
      }
      TCVT(msk_ub, qk_half_ub, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_ALL);

      GHalfUb g_slice(1, local_rows);
      TASSIGN(g_slice, GUbAddr + row_offset * sizeof(float));
      TMOV(g_v_ub, g_slice);
      pipe_barrier(PIPE_V);

      TEXPANDS(qk_ub, 0.0f);
      pipe_barrier(PIPE_V);
      for (uint32_t row = 0; row < local_rows; ++row) {
        GRowUb coeff_row;
        TASSIGN(coeff_row, CoeffUbAddr + row * ChunkSize * sizeof(float));
        TADDS(coeff_row, g_ub, -g_v_ub.GetValue(row));
        pipe_barrier(PIPE_V);
      }
      TSUB(coeff_ub, qk_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TMUL(coeff_ub, coeff_ub, msk_ub);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);
      TEXP(g_v_ub, g_v_ub);
      pipe_barrier(PIPE_V);

      GdnWaitCrossFlag(0);
      pipe_barrier(PIPE_ALL);
      PackedSquareHalf qk_global(workspace_qk + square_offset + row_offset * ChunkSize);
      PackedHiddenHalf qs_global(workspace_qs_qkv + hidden_offset +
                                 row_offset * HiddenSize);
      TLOAD(qk_half_ub, qk_global);
      TLOAD(qs_half_ub, qs_global);
      pipe_barrier(PIPE_ALL);
      GdnSetFlag<PIPE_MTE2, PIPE_V>(0);
      GdnWaitFlag<PIPE_MTE2, PIPE_V>(0);

      TCVT(qk_ub, qk_half_ub, pto::RoundMode::CAST_NONE);
      TCVT(qs_ub, qs_half_ub, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_ALL);
      TMUL(qk_ub, qk_ub, coeff_ub);
      TMUL(qk_ub, qk_ub, msk_ub);
      pipe_barrier(PIPE_V);
      TCVT(qk_half_ub, qk_ub, pto::RoundMode::CAST_NONE);
      GdnSetFlag<PIPE_V, PIPE_MTE3>(0);
      GdnWaitFlag<PIPE_V, PIPE_MTE3>(0);
      PackedSquareHalf qk_gated_global(workspace_qk_gated + square_offset +
                                       row_offset * ChunkSize);
      TSTORE(qk_gated_global, qk_half_ub);
      pipe_barrier(PIPE_ALL);
      GdnSetCrossFlag<PIPE_MTE3>(1, 2);

      GColUb g_col_ub;
      TASSIGN(g_col_ub, GvUbAddr);
      TROWEXPAND(coeff_ub, g_col_ub);
      pipe_barrier(PIPE_V);
      TMUL(qs_ub, qs_ub, coeff_ub);
      pipe_barrier(PIPE_V);

      GdnWaitCrossFlag(2);
      pipe_barrier(PIPE_ALL);
      PackedHiddenHalf qkv_global(workspace_qs_qkv + hidden_offset +
                                  row_offset * HiddenSize);
      TLOAD(o_half_ub, qkv_global);
      pipe_barrier(PIPE_ALL);
      GdnSetFlag<PIPE_MTE2, PIPE_V>(1);
      GdnWaitFlag<PIPE_MTE2, PIPE_V>(1);
      TCVT(o_ub, o_half_ub, pto::RoundMode::CAST_NONE);
      TADD(o_ub, qs_ub, o_ub);
      pipe_barrier(PIPE_V);
      TCVT(o_half_ub, o_ub, pto::RoundMode::CAST_NONE);
      GdnSetFlag<PIPE_V, PIPE_MTE3>(1);
      GdnWaitFlag<PIPE_V, PIPE_MTE3>(1);

      const int32_t token_offset = static_cast<int32_t>(
          seq.token_base_offset + (row_start + row_offset) * seq.row_stride);
      OutGlobalDyn o_global(
          o + token_offset,
          {1, 1, 1, static_cast<int32_t>(local_rows), HiddenSize},
          {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
      TSTORE(o_global, o_half_ub);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_chunk_o(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v,
    __gm__ uint8_t *s_packed, __gm__ uint8_t *g_packed,
    __gm__ uint8_t *workspace_qk, __gm__ uint8_t *workspace_qs_qkv,
    __gm__ uint8_t *workspace_qk_gated, __gm__ uint8_t *o,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t fixed_seq_len,
    uint64_t ffts_addr) {
  main_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(q), reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(v),
      reinterpret_cast<__gm__ half *>(s_packed),
      reinterpret_cast<__gm__ float *>(g_packed),
      reinterpret_cast<__gm__ half *>(workspace_qk),
      reinterpret_cast<__gm__ half *>(workspace_qs_qkv),
      reinterpret_cast<__gm__ half *>(workspace_qk_gated),
      reinterpret_cast<__gm__ half *>(o), cu_seqlens, batch_size, fixed_seq_len,
      ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *q,
                            uint8_t *k, uint8_t *v, uint8_t *s_packed,
                            uint8_t *g_packed, uint8_t *workspace_qk,
                            uint8_t *workspace_qs_qkv,
                            uint8_t *workspace_qk_gated, uint8_t *o,
                            int32_t *cu_seqlens, int64_t batch_size,
                            int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_o<<<blockDim, nullptr, stream>>>(
      q, k, v, s_packed, g_packed, workspace_qk, workspace_qs_qkv,
      workspace_qk_gated, o, cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}
