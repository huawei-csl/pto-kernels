#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

#include "gdn_pto_shared.h"
#include "gdn_seq_info.h"

using namespace pto;

AICORE inline uint32_t GdnMinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

#ifndef GDN_H
#define GDN_H 2
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void qk_cube_kernel(__gm__ half *q, __gm__ half *k, __gm__ half *workspace_qk,
                           __gm__ int32_t *cu_seqlens, int64_t batch_size,
                           int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t QL1Addr = 0;
  constexpr int32_t KL1Addr = 32768;

  using ChunkGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using ChunkGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using ChunkGlobalDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using PackedOutDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using ChunkL1Dyn = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                          DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t total_work = batch_size * NumHeads;

  GdnL1Mat<half, ChunkSize, HiddenSize> q_l1;
  GdnL1Mat<half, ChunkSize, HiddenSize> k_l1;
  TASSIGN(q_l1, QL1Addr);
  TASSIGN(k_l1, KL1Addr);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> qk_l0;
  TASSIGN(qk_l0, 0);

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
      const int32_t out_offset = static_cast<int32_t>(
          ((seq.chunk_offset + chunk_idx) * NumHeads + head_idx) * ChunkSquareElems);

      ChunkL1Dyn q_dyn(valid_rows, HiddenSize);
      ChunkL1Dyn k_dyn(valid_rows, HiddenSize);
      TASSIGN(q_dyn, QL1Addr);
      TASSIGN(k_dyn, KL1Addr);
      ChunkGlobalDyn q_global(
          q + token_offset, {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
      ChunkGlobalDyn k_global(
          k + token_offset, {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
      TLOAD(q_dyn, q_global);
      TLOAD(k_dyn, k_global);
      pipe_barrier(PIPE_ALL);

      GdnMatmulL1<ChunkSize, ChunkSize, HiddenSize, false, true>(qk_l0, q_l1, k_l1,
                                                                 true);
      PackedOutDyn out_global(
          workspace_qk + out_offset,
          {1, 1, 1, static_cast<int32_t>(valid_rows), ChunkSize},
          {1, 1, 1, ChunkSize, 1});
      TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> qk_tail(valid_rows,
                                                                     ChunkSize);
      TASSIGN(qk_tail, 0);
      TSTORE(out_global, qk_tail);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void qs_cube_kernel(__gm__ half *q, __gm__ half *s_packed,
                           __gm__ half *workspace_qs, __gm__ int32_t *cu_seqlens,
                           int64_t batch_size, int64_t fixed_seq_len,
                           uint64_t ffts_addr) {
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t HiddenSquareElems = HiddenSize * HiddenSize;
  constexpr int32_t QL1Addr = 0;
  constexpr int32_t SL1Addr = 32768;

  using ChunkGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using ChunkGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using ChunkGlobalDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using PackedState =
      GlobalTensor<half, TileShape2D<half, HiddenSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, HiddenSize, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedOutDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using ChunkL1Dyn = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                          DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t total_work = batch_size * NumHeads;

  GdnL1Mat<half, ChunkSize, HiddenSize> q_l1;
  GdnL1Mat<half, HiddenSize, HiddenSize> s_l1;
  TASSIGN(q_l1, QL1Addr);
  TASSIGN(s_l1, SL1Addr);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qs_l0;
  TASSIGN(qs_l0, 0);

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

      ChunkL1Dyn q_dyn(valid_rows, HiddenSize);
      TASSIGN(q_dyn, QL1Addr);
      ChunkGlobalDyn q_global(
          q + token_offset, {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
      PackedState s_global(s_packed + chunk_base * HiddenSquareElems);
      TLOAD(q_dyn, q_global);
      TLOAD(s_l1, s_global);
      pipe_barrier(PIPE_ALL);

      GdnMatmulL1<ChunkSize, HiddenSize, HiddenSize, false, false>(qs_l0, q_l1,
                                                                   s_l1, true);
      PackedOutDyn out_global(
          workspace_qs + chunk_base * ChunkHiddenElems,
          {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, HiddenSize, 1});
      TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> qs_tail(valid_rows,
                                                                       HiddenSize);
      TASSIGN(qs_tail, 0);
      TSTORE(out_global, qs_tail);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void qkv_cube_kernel(__gm__ half *qk_packed, __gm__ half *v,
                            __gm__ half *workspace_qkv, __gm__ int32_t *cu_seqlens,
                            int64_t batch_size, int64_t fixed_seq_len,
                            uint64_t ffts_addr) {
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t QKL1Addr = 0;
  constexpr int32_t VL1Addr = 32768;

  using ChunkGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using ChunkGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using ChunkGlobalDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using PackedQKDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using PackedOutDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using QKL1Dyn = Tile<TileType::Mat, half, ChunkSize, ChunkSize, BLayout::ColMajor,
                       DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  using VL1Dyn = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                      DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t total_work = batch_size * NumHeads;

  GdnL1Mat<half, ChunkSize, ChunkSize> qk_l1;
  GdnL1Mat<half, ChunkSize, HiddenSize> v_l1;
  TASSIGN(qk_l1, QKL1Addr);
  TASSIGN(v_l1, VL1Addr);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> qkv_l0;
  TASSIGN(qkv_l0, 0);

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

      QKL1Dyn qk_dyn(valid_rows, ChunkSize);
      VL1Dyn v_dyn(valid_rows, HiddenSize);
      TASSIGN(qk_dyn, QKL1Addr);
      TASSIGN(v_dyn, VL1Addr);
      PackedQKDyn qk_global(
          qk_packed + chunk_base * ChunkSquareElems,
          {1, 1, 1, static_cast<int32_t>(valid_rows), ChunkSize},
          {1, 1, 1, ChunkSize, 1});
      ChunkGlobalDyn v_global(
          v + token_offset, {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
      TLOAD(qk_dyn, qk_global);
      TLOAD(v_dyn, v_global);
      pipe_barrier(PIPE_ALL);

      GdnMatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(qkv_l0, qk_l1,
                                                                  v_l1, true);
      PackedOutDyn out_global(
          workspace_qkv + chunk_base * ChunkHiddenElems,
          {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, HiddenSize, 1});
      TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> qkv_tail(valid_rows,
                                                                        HiddenSize);
      TASSIGN(qkv_tail, 0);
      TSTORE(out_global, qkv_tail);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void gate_qk_vec_kernel(__gm__ half *workspace_qk, __gm__ float *g_packed,
                               __gm__ int32_t *cu_seqlens, int64_t batch_size,
                               int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t GVUbAddr = GUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t QKHalfUbAddr = GVUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t QKUbAddr = QKHalfUbAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t CoeffUbAddr = QKUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t TmpUbAddr = CoeffUbAddr + HalfChunk * ChunkSize * sizeof(float);

  using PackedGGlobal =
      GlobalTensor<float, TileShape2D<float, 1, ChunkSize, Layout::ND>,
                   BaseShape2D<float, 1, ChunkSize, Layout::ND>, Layout::ND>;
  using PackedGHalfShape = Shape<1, 1, 1, 1, DYNAMIC>;
  using PackedGHalfStride = Stride<1, 1, 1, 1, 1>;
  using PackedGHalfGlobal =
      GlobalTensor<float, PackedGHalfShape, PackedGHalfStride, Layout::ND>;
  using HalfQKGlobal =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using GUb = Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, DYNAMIC,
                   DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using GHalfUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor, DYNAMIC,
                       DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using QKHalfUb = GdnUbND<half, HalfChunk, ChunkSize>;
  using QKUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using GRowUb = GdnUbND<float, 1, ChunkSize>;
  using MaskUb = GdnUbND<float, HalfChunk, ChunkSize>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * NumHeads;

  GUb g_ub(1, ChunkSize);
  QKHalfUb qk_half_ub;
  QKUb qk_ub;
  MaskUb mask_ub;
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(qk_half_ub, QKHalfUbAddr);
  TASSIGN(qk_ub, QKUbAddr);
  TASSIGN(mask_ub, TmpUbAddr);

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  GdnBuildLowerTriMask(mask_ub, vid, true);
  pipe_barrier(PIPE_ALL);
  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }
    const uint32_t head_idx = static_cast<uint32_t>(pid % NumHeads);
    const uint32_t seq_idx = static_cast<uint32_t>(pid / NumHeads);
    const GdnSeqInfo seq =
        GetGdnSeqInfo(seq_idx, ChunkSize, static_cast<uint32_t>(fixed_seq_len),
                      cu_seqlens);
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
      if (local_rows == 0) {
        continue;
      }
      const int32_t chunk_base =
          static_cast<int32_t>((seq.chunk_offset + chunk_idx) * NumHeads + head_idx);
      PackedGGlobal g_global(g_packed + chunk_base * ChunkSize);
      PackedGHalfGlobal g_half_global(
          g_packed + chunk_base * ChunkSize + row_offset,
          {1, 1, 1, 1, static_cast<int32_t>(local_rows)},
          {1, 1, 1, 1, 1});
      HalfQKGlobal qk_global(workspace_qk + chunk_base * ChunkSquareElems +
                             row_offset * ChunkSize);
      GHalfUb g_local_ub(1, local_rows);
      TASSIGN(g_local_ub, GVUbAddr);
      TLOAD(g_ub, g_global);
      TLOAD(g_local_ub, g_half_global);
      TLOAD(qk_half_ub, qk_global);
      pipe_barrier(PIPE_ALL);
      TCVT(qk_ub, qk_half_ub, pto::RoundMode::CAST_NONE);
      for (uint32_t row = 0; row < local_rows; ++row) {
        GRowUb coeff_row;
        GRowUb qk_row;
        TASSIGN(coeff_row, CoeffUbAddr);
        TASSIGN(qk_row, QKUbAddr + row * ChunkSize * sizeof(float));
        TEXPANDS(coeff_row, g_local_ub.GetValue(row));
        TSUB(coeff_row, coeff_row, g_ub);
        TEXP(coeff_row, coeff_row);
        pipe_barrier(PIPE_V);
        TMUL(qk_row, qk_row, coeff_row);
        pipe_barrier(PIPE_V);
      }
      TMUL(qk_ub, qk_ub, mask_ub);
      pipe_barrier(PIPE_ALL);
      TCVT(qk_half_ub, qk_ub, pto::RoundMode::CAST_NONE);
      TSTORE(qk_global, qk_half_ub);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void add_store_vec_kernel(__gm__ half *workspace_qs, __gm__ half *workspace_qkv,
                                 __gm__ float *g_packed, __gm__ half *o,
                                 __gm__ int32_t *cu_seqlens, int64_t batch_size,
                                 int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t QSHalfUbAddr = GUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t QSUbAddr = QSHalfUbAddr + HalfChunk * HiddenSize * sizeof(half);
  constexpr int32_t QKVHalfUbAddr = QSUbAddr + HalfChunk * HiddenSize * sizeof(float);
  constexpr int32_t QKVUbAddr = QKVHalfUbAddr + HalfChunk * HiddenSize * sizeof(half);
  constexpr int32_t ScaleUbAddr = QKVUbAddr + HalfChunk * HiddenSize * sizeof(float);

  using PackedGHalfShape = Shape<1, 1, 1, 1, DYNAMIC>;
  using PackedGHalfStride = Stride<1, 1, 1, 1, 1>;
  using PackedGHalfGlobal =
      GlobalTensor<float, PackedGHalfShape, PackedGHalfStride, Layout::ND>;
  using HalfChunkGlobal =
      GlobalTensor<half, TileShape2D<half, HalfChunk, HiddenSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, HiddenSize, Layout::ND>, Layout::ND>;
  using OutGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using OutGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using OutGlobalDyn =
      GlobalTensor<half, OutGlobalDynShape, OutGlobalDynStride, Layout::ND>;
  using GHalfUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor, DYNAMIC,
                       DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using QSHalfUb = GdnUbND<half, HalfChunk, HiddenSize>;
  using QSUb = GdnUbND<float, HalfChunk, HiddenSize>;
  using GColUb = GdnUbDN<float, HalfChunk, 1>;
  using ScaleUb = GdnUbND<float, HalfChunk, HiddenSize>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * NumHeads;

  GColUb g_col_ub;
  QSHalfUb qs_half_ub;
  QSUb qs_ub;
  QSHalfUb qkv_half_ub;
  QSUb qkv_ub;
  ScaleUb scale_ub;
  TASSIGN(g_col_ub, GUbAddr);
  TASSIGN(qs_half_ub, QSHalfUbAddr);
  TASSIGN(qs_ub, QSUbAddr);
  TASSIGN(qkv_half_ub, QKVHalfUbAddr);
  TASSIGN(qkv_ub, QKVUbAddr);
  TASSIGN(scale_ub, ScaleUbAddr);

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
      if (local_rows == 0) {
        continue;
      }
      const int32_t chunk_base =
          static_cast<int32_t>((seq.chunk_offset + chunk_idx) * NumHeads + head_idx);
      PackedGHalfGlobal g_half_global(
          g_packed + chunk_base * ChunkSize + row_offset,
          {1, 1, 1, 1, static_cast<int32_t>(local_rows)},
          {1, 1, 1, 1, 1});
      HalfChunkGlobal qs_global(workspace_qs + chunk_base * ChunkHiddenElems +
                                row_offset * HiddenSize);
      HalfChunkGlobal qkv_global(workspace_qkv + chunk_base * ChunkHiddenElems +
                                 row_offset * HiddenSize);
      GHalfUb g_local_ub(1, local_rows);
      TASSIGN(g_local_ub, GUbAddr);
      TLOAD(g_local_ub, g_half_global);
      TLOAD(qs_half_ub, qs_global);
      TLOAD(qkv_half_ub, qkv_global);
      pipe_barrier(PIPE_ALL);
      TEXP(g_local_ub, g_local_ub);
      pipe_barrier(PIPE_V);
      TROWEXPAND(scale_ub, g_col_ub);
      TCVT(qs_ub, qs_half_ub, pto::RoundMode::CAST_NONE);
      TCVT(qkv_ub, qkv_half_ub, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TMUL(qs_ub, qs_ub, scale_ub);
      TADD(qs_ub, qs_ub, qkv_ub);
      pipe_barrier(PIPE_V);
      TCVT(qs_half_ub, qs_ub, pto::RoundMode::CAST_NONE);
      const int32_t token_offset = static_cast<int32_t>(
          seq.token_base_offset + (row_start + row_offset) * seq.row_stride);
      OutGlobalDyn o_global(
          o + token_offset,
          {1, 1, 1, static_cast<int32_t>(local_rows), HiddenSize},
          {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
      TSTORE(o_global, qs_half_ub);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_chunk_o_qk(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *workspace_qk,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t fixed_seq_len,
    uint64_t ffts_addr) {
  qk_cube_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(q), reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(workspace_qk), cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" __global__ AICORE void launch_chunk_o_qs(
    __gm__ uint8_t *q, __gm__ uint8_t *s_packed, __gm__ uint8_t *workspace_qs,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t fixed_seq_len,
    uint64_t ffts_addr) {
  qs_cube_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(q),
      reinterpret_cast<__gm__ half *>(s_packed),
      reinterpret_cast<__gm__ half *>(workspace_qs), cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" __global__ AICORE void launch_chunk_o_qkv(
    __gm__ uint8_t *workspace_qk, __gm__ uint8_t *v, __gm__ uint8_t *workspace_qkv,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t fixed_seq_len,
    uint64_t ffts_addr) {
  qkv_cube_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(workspace_qk),
      reinterpret_cast<__gm__ half *>(v),
      reinterpret_cast<__gm__ half *>(workspace_qkv), cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" __global__ AICORE void launch_chunk_o_gate_qk(
    __gm__ uint8_t *workspace_qk, __gm__ uint8_t *g_packed,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t fixed_seq_len,
    uint64_t ffts_addr) {
  gate_qk_vec_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(workspace_qk),
      reinterpret_cast<__gm__ float *>(g_packed), cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" __global__ AICORE void launch_chunk_o_add_store(
    __gm__ uint8_t *workspace_qs, __gm__ uint8_t *workspace_qkv,
    __gm__ uint8_t *g_packed, __gm__ uint8_t *o, __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t fixed_seq_len, uint64_t ffts_addr) {
  add_store_vec_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(workspace_qs),
      reinterpret_cast<__gm__ half *>(workspace_qkv),
      reinterpret_cast<__gm__ float *>(g_packed),
      reinterpret_cast<__gm__ half *>(o), cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" void call_qk_kernel(uint32_t blockDim, void *stream, uint8_t *q,
                               uint8_t *k, uint8_t *workspace_qk,
                               int32_t *cu_seqlens, int64_t batch_size,
                               int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_o_qk<<<blockDim, nullptr, stream>>>(
      q, k, workspace_qk, cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}

extern "C" void call_qs_kernel(uint32_t blockDim, void *stream, uint8_t *q,
                               uint8_t *s_packed, uint8_t *workspace_qs,
                               int32_t *cu_seqlens, int64_t batch_size,
                               int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_o_qs<<<blockDim, nullptr, stream>>>(
      q, s_packed, workspace_qs, cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}

extern "C" void call_gate_qk_kernel(uint32_t blockDim, void *stream,
                                    uint8_t *workspace_qk, uint8_t *g_packed,
                                    int32_t *cu_seqlens, int64_t batch_size,
                                    int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_o_gate_qk<<<blockDim, nullptr, stream>>>(
      workspace_qk, g_packed, cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}

extern "C" void call_qkv_kernel(uint32_t blockDim, void *stream,
                                uint8_t *workspace_qk, uint8_t *v,
                                uint8_t *workspace_qkv, int32_t *cu_seqlens,
                                int64_t batch_size, int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_o_qkv<<<blockDim, nullptr, stream>>>(
      workspace_qk, v, workspace_qkv, cu_seqlens, batch_size, fixed_seq_len,
      ffts_addr);
}

extern "C" void call_add_store_kernel(uint32_t blockDim, void *stream,
                                      uint8_t *workspace_qs,
                                      uint8_t *workspace_qkv, uint8_t *g_packed,
                                      uint8_t *o, int32_t *cu_seqlens,
                                      int64_t batch_size,
                                      int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_o_add_store<<<blockDim, nullptr, stream>>>(
      workspace_qs, workspace_qkv, g_packed, o, cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}
