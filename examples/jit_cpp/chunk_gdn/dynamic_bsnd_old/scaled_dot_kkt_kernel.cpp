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

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void main_cube_kernel(__gm__ half *k, __gm__ half *workspace,
                             __gm__ int32_t *cu_seqlens, int64_t batch_size,
                             int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t KL1Addr = 0;

  using KGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using KGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using KGlobalDyn = GlobalTensor<half, KGlobalDynShape, KGlobalDynStride, Layout::ND>;
  using ChunkPackedGlobal =
      GlobalTensor<half, TileShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, ChunkSize, Layout::ND>, Layout::ND>;
  using KL1 = GdnL1Mat<half, ChunkSize, HiddenSize>;
  using KDynL1 = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                      DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t total_work = batch_size * NumHeads;

  KL1 k_l1;
  TASSIGN(k_l1, KL1Addr);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a_l0;
  TASSIGN(a_l0, 0);

#if defined(__DAV_C220_CUBE__)
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
      const uint32_t rows_left = static_cast<uint32_t>(seq.seq_len - row_start);
      const uint32_t valid_rows =
          rows_left < static_cast<uint32_t>(ChunkSize) ? rows_left
                                                       : static_cast<uint32_t>(ChunkSize);
      const int32_t token_offset = static_cast<int32_t>(
          (seq.bos + row_start) * NumHeads * HiddenSize +
          head_idx * HiddenSize);
      const int32_t packed_offset = static_cast<int32_t>(
          ((seq.chunk_offset + chunk_idx) * NumHeads + head_idx) *
          ChunkSquareElems);

      KDynL1 k_dyn(valid_rows, HiddenSize);
      TASSIGN(k_dyn, KL1Addr);
      KGlobalDyn k_global(
          k + token_offset,
          {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, NumHeads * HiddenSize, 1});
      TLOAD(k_dyn, k_global);
      pipe_barrier(PIPE_ALL);

      GdnMatmulL1<ChunkSize, ChunkSize, HiddenSize, false, true>(a_l0, k_l1, k_l1,
                                                                 true);
      ChunkPackedGlobal workspace_global(workspace + packed_offset);
      TSTORE(workspace_global, a_l0);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

template <int32_t NumHeads, int32_t ChunkSize>
AICORE void main_vec_kernel(__gm__ half *beta, __gm__ float *g, __gm__ float *msk,
                            __gm__ half *workspace, __gm__ half *a_out,
                            __gm__ int32_t *cu_seqlens, int64_t batch_size,
                            int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t VecNum = 2;
  constexpr int32_t HalfChunk = ChunkSize / VecNum;
  constexpr int32_t HeadTileCols = ((NumHeads + 15) / 16) * 16;
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t BetaHalfUbAddr = GUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t BetaUbAddr =
      BetaHalfUbAddr + HalfChunk * HeadTileCols * sizeof(half);
  constexpr int32_t GvUbAddr = BetaUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t AUbAddr = GvUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t GRUbAddr = AUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t GCUbAddr = GRUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t MskUbAddr = GCUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t GR2dUbAddr = MskUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t TmpUbAddr = GR2dUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t GC2dUbAddr = TmpUbAddr + 3 * HalfChunk * ChunkSize * sizeof(uint8_t);
  constexpr int32_t CoeffUbAddr = GC2dUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t AUbHalfAddr = GR2dUbAddr;

  using PackedGGlobal =
      GlobalTensor<float, TileShape2D<float, 1, ChunkSize, Layout::ND>,
                   BaseShape2D<float, 1, ChunkSize, Layout::ND>, Layout::ND>;
  using PackedGHalfShape = Shape<1, 1, 1, 1, DYNAMIC>;
  using PackedGHalfStride = Stride<1, 1, 1, 1, 1>;
  using PackedGHalfGlobal =
      GlobalTensor<float, PackedGHalfShape, PackedGHalfStride, Layout::ND>;
  using BetaBlockShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using BetaBlockStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using BetaBlockGlobal = GlobalTensor<half, BetaBlockShape, BetaBlockStride, Layout::ND>;
  using MaskGlobal =
      GlobalTensor<float, TileShape2D<float, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<float, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using HalfAOutDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using HalfAOutDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using HalfAOutGlobalDyn =
      GlobalTensor<half, HalfAOutDynShape, HalfAOutDynStride, Layout::ND>;
  using HalfAOutGlobal =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using GUb = Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, DYNAMIC,
                   DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using GHalfUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor, DYNAMIC,
                       DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using BetaBlockUb = Tile<TileType::Vec, half, HalfChunk, HeadTileCols, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using BetaUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor,
                      DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using AUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using AHalfUb = GdnUbND<half, HalfChunk, ChunkSize>;
  using GColUb = GdnUbDN<float, HalfChunk, 1>;
  using GRowUb = GdnUbND<float, 1, ChunkSize>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * NumHeads;

  GUb g_ub(1, ChunkSize);
  GColUb g_r_col_ub;
  GRowUb g_c_ub;
  AUb msk_ub;
  AUb g_r_2d_ub;
  AUb g_c_2d_ub;
  AUb coeff_ub;
  AUb a_ub;
  AHalfUb a_half_ub;
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(g_r_col_ub, GRUbAddr);
  TASSIGN(g_c_ub, GCUbAddr);
  TASSIGN(msk_ub, MskUbAddr);
  TASSIGN(g_r_2d_ub, GR2dUbAddr);
  TASSIGN(g_c_2d_ub, GC2dUbAddr);
  TASSIGN(coeff_ub, CoeffUbAddr);
  TASSIGN(a_ub, AUbAddr);
  TASSIGN(a_half_ub, AUbHalfAddr);

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
    const GdnSeqInfo seq =
        GetGdnSeqInfo(seq_idx, ChunkSize, static_cast<uint32_t>(fixed_seq_len),
                      cu_seqlens);
    const uint32_t chunk_num = GdnDivCeilU32(seq.seq_len, ChunkSize);

    for (uint32_t chunk_idx = 0; chunk_idx < chunk_num; ++chunk_idx) {
      const uint32_t row_start = chunk_idx * ChunkSize;
      const uint32_t rows_left = static_cast<uint32_t>(seq.seq_len - row_start);
      const uint32_t valid_rows =
          rows_left < static_cast<uint32_t>(ChunkSize) ? rows_left
                                                       : static_cast<uint32_t>(ChunkSize);
      const uint32_t row_offset = static_cast<uint32_t>(vid) * HalfChunk;
      const uint32_t local_valid_rows =
          valid_rows > row_offset
              ? min(static_cast<uint32_t>(valid_rows - row_offset),
                    static_cast<uint32_t>(HalfChunk))
              : 0;
      if (local_valid_rows == 0) {
        continue;
      }

      const int32_t packed_chunk_base = static_cast<int32_t>(
          ((seq.chunk_offset + chunk_idx) * NumHeads + head_idx));
      const int32_t g_offset = packed_chunk_base * ChunkSize;
      const int32_t beta_offset = static_cast<int32_t>(
          (seq.bos + row_start + row_offset) * NumHeads);
      const int32_t packed_square_offset = packed_chunk_base * ChunkSquareElems;

      PackedGGlobal g_global(g + g_offset);
      BetaBlockGlobal beta_global(
          beta + beta_offset,
          {1, 1, 1, static_cast<int32_t>(local_valid_rows), NumHeads},
          {1, 1, 1, NumHeads, 1});
      MaskGlobal mask_global(msk + row_offset * ChunkSize);
      BetaBlockUb beta_block_ub(HalfChunk, NumHeads);
      BetaUb beta_ub(1, HalfChunk);
      GHalfUb g_v_ub(1, HalfChunk);
      TASSIGN(beta_block_ub, BetaHalfUbAddr);
      TASSIGN(beta_ub, BetaUbAddr);
      TASSIGN(g_v_ub, GvUbAddr);

      TLOAD(g_ub, g_global);
      TLOAD(beta_block_ub, beta_global);
      pipe_barrier(PIPE_ALL);
      GdnSetFlag<PIPE_MTE2, PIPE_V>(2);
      GdnWaitFlag<PIPE_MTE2, PIPE_V>(2);
      GHalfUb g_ub_temp(1, HalfChunk);
      TASSIGN(g_ub_temp, GUbAddr + row_offset * sizeof(float));
      TMOV(g_v_ub, g_ub_temp);
      pipe_barrier(PIPE_V);

      for (uint32_t row = 0; row < local_valid_rows; ++row) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        beta_ub.SetValue(row, static_cast<float>(beta_block_ub.GetValue(row * HeadTileCols + head_idx)));
      }
      pipe_barrier(PIPE_V);
      TEXPANDS(coeff_ub, 0.0f);
      pipe_barrier(PIPE_V);
      TLOAD(msk_ub, mask_global);
      pipe_barrier(PIPE_ALL);
      for (uint32_t row = 0; row < local_valid_rows; ++row) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        GRowUb coeff_row;
        TASSIGN(coeff_row, CoeffUbAddr + row * ChunkSize * sizeof(float));
        TADDS(coeff_row, g_ub, -g_v_ub.GetValue(row));
      }
      pipe_barrier(PIPE_V);
      TEXPANDS(g_r_2d_ub, 0.0f);
      TSUB(g_c_2d_ub, g_r_2d_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TEXP(g_c_2d_ub, g_c_2d_ub);
      pipe_barrier(PIPE_V);
      for (uint32_t row = 0; row < local_valid_rows; ++row) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        GRowUb coeff_row;
        TASSIGN(coeff_row, GC2dUbAddr + row * ChunkSize * sizeof(float));
        TMULS(coeff_row, coeff_row,
              static_cast<float>(
                  beta_block_ub.GetValue(row * HeadTileCols + head_idx)));
      }
      pipe_barrier(PIPE_V);
      HalfAOutGlobal workspace_global(workspace + packed_square_offset +
                                      row_offset * ChunkSize);
      TLOAD(a_half_ub, workspace_global);
      pipe_barrier(PIPE_ALL);
      GdnSetFlag<PIPE_MTE2, PIPE_V>(0);
      GdnWaitFlag<PIPE_MTE2, PIPE_V>(0);
      TCVT(a_ub, a_half_ub, pto::RoundMode::CAST_NONE);
      TMUL(a_ub, a_ub, g_c_2d_ub);
      pipe_barrier(PIPE_V);
      for (uint32_t row = 0; row < local_valid_rows; ++row) {
        const uint32_t global_row = row_offset + row;
        for (uint32_t col = global_row; col < static_cast<uint32_t>(ChunkSize); ++col) {
          a_ub.SetValue(row * ChunkSize + col, 0.0f);
        }
      }
      pipe_barrier(PIPE_ALL);
      TCVT(a_half_ub, a_ub, pto::RoundMode::CAST_NONE);
      GdnSetFlag<PIPE_V, PIPE_MTE3>(0);
      GdnWaitFlag<PIPE_V, PIPE_MTE3>(0);
      HalfAOutGlobalDyn a_global(
          a_out + packed_square_offset + row_offset * ChunkSize,
          {1, 1, 1, static_cast<int32_t>(local_valid_rows), ChunkSize},
          {1, 1, 1, ChunkSize, 1});
      TSTORE(a_global, a_half_ub);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

template <int32_t NumHeads, int32_t HiddenSize, int32_t ChunkSize>
AICORE void main_kernel(__gm__ half *k, __gm__ half *beta, __gm__ float *g,
                        __gm__ float *msk, __gm__ half *workspace,
                        __gm__ half *a_out, __gm__ int32_t *cu_seqlens,
                        int64_t batch_size, int64_t fixed_seq_len,
                        uint64_t ffts_addr) {
  constexpr int32_t VecNum = 2;
  constexpr int32_t HalfChunk = ChunkSize / VecNum;
  constexpr int32_t HeadTileCols = ((NumHeads + 15) / 16) * 16;
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t KL1Addr = 0;
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t BetaHalfUbAddr = GUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t BetaUbAddr =
      BetaHalfUbAddr + HalfChunk * HeadTileCols * sizeof(half);
  constexpr int32_t GvUbAddr = BetaUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t AUbAddr = GvUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t GRUbAddr = AUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t GCUbAddr = GRUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t MskUbAddr = GCUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t GR2dUbAddr = MskUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t TmpUbAddr = GR2dUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t GC2dUbAddr = TmpUbAddr + 3 * HalfChunk * ChunkSize * sizeof(uint8_t);
  constexpr int32_t CoeffUbAddr = GC2dUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t AUbHalfAddr = GR2dUbAddr;

  using KGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using KGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using KGlobalDyn = GlobalTensor<half, KGlobalDynShape, KGlobalDynStride, Layout::ND>;
  using ChunkPackedGlobal =
      GlobalTensor<half, TileShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, ChunkSize, Layout::ND>, Layout::ND>;
  using KL1 = GdnL1Mat<half, ChunkSize, HiddenSize>;
  using KDynL1 = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                      DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  using PackedGGlobal =
      GlobalTensor<float, TileShape2D<float, 1, ChunkSize, Layout::ND>,
                   BaseShape2D<float, 1, ChunkSize, Layout::ND>, Layout::ND>;
  using BetaBlockShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using BetaBlockStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using BetaBlockGlobal = GlobalTensor<half, BetaBlockShape, BetaBlockStride, Layout::ND>;
  using MaskGlobal =
      GlobalTensor<float, TileShape2D<float, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<float, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using HalfAOutDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using HalfAOutDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using HalfAOutGlobalDyn =
      GlobalTensor<half, HalfAOutDynShape, HalfAOutDynStride, Layout::ND>;
  using HalfAOutGlobal =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using GUb = Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, DYNAMIC,
                   DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using GHalfUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor, DYNAMIC,
                       DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using BetaBlockUb = Tile<TileType::Vec, half, HalfChunk, HeadTileCols, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using BetaUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor,
                      DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using AUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using AHalfUb = GdnUbND<half, HalfChunk, ChunkSize>;
  using GColUb = GdnUbDN<float, HalfChunk, 1>;
  using GRowUb = GdnUbND<float, 1, ChunkSize>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * NumHeads;

  KL1 k_l1;
  TASSIGN(k_l1, KL1Addr);
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a_l0;
  TASSIGN(a_l0, 0);

  GUb g_ub(1, ChunkSize);
  GColUb g_r_col_ub;
  GRowUb g_c_ub;
  AUb msk_ub;
  AUb g_r_2d_ub;
  AUb g_c_2d_ub;
  AUb coeff_ub;
  AUb a_ub;
  AHalfUb a_half_ub;
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(g_r_col_ub, GRUbAddr);
  TASSIGN(g_c_ub, GCUbAddr);
  TASSIGN(msk_ub, MskUbAddr);
  TASSIGN(g_r_2d_ub, GR2dUbAddr);
  TASSIGN(g_c_2d_ub, GC2dUbAddr);
  TASSIGN(coeff_ub, CoeffUbAddr);
  TASSIGN(a_ub, AUbAddr);
  TASSIGN(a_half_ub, AUbHalfAddr);

#if defined(__DAV_C220_CUBE__)
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
      GdnWaitCrossFlag(1);
      pipe_barrier(PIPE_ALL);

      const uint32_t row_start = chunk_idx * ChunkSize;
      const uint32_t rows_left = static_cast<uint32_t>(seq.seq_len - row_start);
      const uint32_t valid_rows =
          rows_left < static_cast<uint32_t>(ChunkSize) ? rows_left
                                                       : static_cast<uint32_t>(ChunkSize);
      const int32_t token_offset = static_cast<int32_t>(
          (seq.bos + row_start) * NumHeads * HiddenSize +
          head_idx * HiddenSize);
      const int32_t packed_offset = static_cast<int32_t>(
          ((seq.chunk_offset + chunk_idx) * NumHeads + head_idx) *
          ChunkSquareElems);

      KDynL1 k_dyn(valid_rows, HiddenSize);
      TASSIGN(k_dyn, KL1Addr);
      KGlobalDyn k_global(
          k + token_offset,
          {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, NumHeads * HiddenSize, 1});
      TLOAD(k_dyn, k_global);
      pipe_barrier(PIPE_ALL);

      GdnMatmulL1<ChunkSize, ChunkSize, HiddenSize, false, true>(a_l0, k_l1, k_l1,
                                                                 true);
      ChunkPackedGlobal workspace_global(workspace + packed_offset);
      TSTORE(workspace_global, a_l0);
      pipe_barrier(PIPE_ALL);

      GdnSetCrossFlag<PIPE_FIX>(0, 2);
    }
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  GdnSetCrossFlag<PIPE_MTE3>(1, 2);

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
      GdnWaitCrossFlag(0);
      pipe_barrier(PIPE_ALL);

      const uint32_t row_start = chunk_idx * ChunkSize;
      const uint32_t rows_left = static_cast<uint32_t>(seq.seq_len - row_start);
      const uint32_t valid_rows =
          rows_left < static_cast<uint32_t>(ChunkSize) ? rows_left
                                                       : static_cast<uint32_t>(ChunkSize);
      const uint32_t row_offset = static_cast<uint32_t>(vid) * HalfChunk;
      const uint32_t local_valid_rows =
          valid_rows > row_offset
              ? min(static_cast<uint32_t>(valid_rows - row_offset),
                    static_cast<uint32_t>(HalfChunk))
              : 0;

      if (local_valid_rows != 0) {
        const int32_t packed_chunk_base = static_cast<int32_t>(
            ((seq.chunk_offset + chunk_idx) * NumHeads + head_idx));
        const int32_t g_offset = packed_chunk_base * ChunkSize;
        const int32_t beta_offset = static_cast<int32_t>(
            (seq.bos + row_start + row_offset) * NumHeads);
        const int32_t packed_square_offset = packed_chunk_base * ChunkSquareElems;

        PackedGGlobal g_global(g + g_offset);
        BetaBlockGlobal beta_global(
            beta + beta_offset,
            {1, 1, 1, static_cast<int32_t>(local_valid_rows), NumHeads},
            {1, 1, 1, NumHeads, 1});
        MaskGlobal mask_global(msk + row_offset * ChunkSize);
        BetaBlockUb beta_block_ub(HalfChunk, NumHeads);
        BetaUb beta_ub(1, HalfChunk);
        GHalfUb g_v_ub(1, HalfChunk);
        TASSIGN(beta_block_ub, BetaHalfUbAddr);
        TASSIGN(beta_ub, BetaUbAddr);
        TASSIGN(g_v_ub, GvUbAddr);

        TLOAD(g_ub, g_global);
        TLOAD(beta_block_ub, beta_global);
        pipe_barrier(PIPE_ALL);
        GdnSetFlag<PIPE_MTE2, PIPE_V>(2);
        GdnWaitFlag<PIPE_MTE2, PIPE_V>(2);
        GHalfUb g_ub_temp(1, HalfChunk);
        TASSIGN(g_ub_temp, GUbAddr + row_offset * sizeof(float));
        TMOV(g_v_ub, g_ub_temp);
        pipe_barrier(PIPE_V);

        for (uint32_t row = 0; row < local_valid_rows; ++row) {
          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
          beta_ub.SetValue(
              row,
              static_cast<float>(
                  beta_block_ub.GetValue(row * HeadTileCols + head_idx)));
        }
        pipe_barrier(PIPE_V);
        TEXPANDS(coeff_ub, 0.0f);
        pipe_barrier(PIPE_V);
        TLOAD(msk_ub, mask_global);
        pipe_barrier(PIPE_ALL);
        for (uint32_t row = 0; row < local_valid_rows; ++row) {
          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
          GRowUb coeff_row;
          TASSIGN(coeff_row, CoeffUbAddr + row * ChunkSize * sizeof(float));
          TADDS(coeff_row, g_ub, -g_v_ub.GetValue(row));
        }
        pipe_barrier(PIPE_V);
        TEXPANDS(g_r_2d_ub, 0.0f);
        TSUB(g_c_2d_ub, g_r_2d_ub, coeff_ub);
        pipe_barrier(PIPE_V);
        TEXP(g_c_2d_ub, g_c_2d_ub);
        pipe_barrier(PIPE_V);
        for (uint32_t row = 0; row < local_valid_rows; ++row) {
          set_flag(PIPE_V, PIPE_S, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
          GRowUb coeff_row;
          TASSIGN(coeff_row, GC2dUbAddr + row * ChunkSize * sizeof(float));
          TMULS(coeff_row, coeff_row,
                static_cast<float>(
                    beta_block_ub.GetValue(row * HeadTileCols + head_idx)));
        }
        pipe_barrier(PIPE_V);
        HalfAOutGlobal workspace_global(workspace + packed_square_offset +
                                        row_offset * ChunkSize);
        TLOAD(a_half_ub, workspace_global);
        pipe_barrier(PIPE_ALL);
        GdnSetFlag<PIPE_MTE2, PIPE_V>(0);
        GdnWaitFlag<PIPE_MTE2, PIPE_V>(0);
        TCVT(a_ub, a_half_ub, pto::RoundMode::CAST_NONE);
        TMUL(a_ub, a_ub, g_c_2d_ub);
        pipe_barrier(PIPE_V);
        for (uint32_t row = 0; row < local_valid_rows; ++row) {
          const uint32_t global_row = row_offset + row;
          for (uint32_t col = global_row;
               col < static_cast<uint32_t>(ChunkSize); ++col) {
            a_ub.SetValue(row * ChunkSize + col, 0.0f);
          }
        }
        pipe_barrier(PIPE_ALL);
        TCVT(a_half_ub, a_ub, pto::RoundMode::CAST_NONE);
        GdnSetFlag<PIPE_V, PIPE_MTE3>(0);
        GdnWaitFlag<PIPE_V, PIPE_MTE3>(0);
        HalfAOutGlobalDyn a_global(
            a_out + packed_square_offset + row_offset * ChunkSize,
            {1, 1, 1, static_cast<int32_t>(local_valid_rows), ChunkSize},
            {1, 1, 1, ChunkSize, 1});
        TSTORE(a_global, a_half_ub);
        pipe_barrier(PIPE_ALL);
      }

      GdnSetCrossFlag<PIPE_MTE3>(1, 2);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_scaled_dot_kkt(
    __gm__ uint8_t *k, __gm__ uint8_t *beta, __gm__ uint8_t *g,
    __gm__ uint8_t *msk, __gm__ uint8_t *workspace, __gm__ uint8_t *a_out,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t fixed_seq_len,
    uint64_t ffts_addr) {
  main_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(beta), reinterpret_cast<__gm__ float *>(g),
      reinterpret_cast<__gm__ float *>(msk),
      reinterpret_cast<__gm__ half *>(workspace),
      reinterpret_cast<__gm__ half *>(a_out), cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" __global__ AICORE void launch_scaled_dot_kkt_cube(
    __gm__ uint8_t *k, __gm__ uint8_t *workspace, __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t fixed_seq_len, uint64_t ffts_addr) {
  main_cube_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(k),
      reinterpret_cast<__gm__ half *>(workspace),
      cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *k, uint8_t *beta,
                            uint8_t *g, uint8_t *msk, uint8_t *workspace,
                            uint8_t *a_out, int32_t *cu_seqlens,
                            int64_t batch_size, int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_scaled_dot_kkt<<<blockDim, nullptr, stream>>>(
      k, beta, g, msk, workspace, a_out, cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" void call_cube_only(uint32_t blockDim, void *stream, uint8_t *k,
                               uint8_t *workspace, int32_t *cu_seqlens,
                               int64_t batch_size, int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_scaled_dot_kkt_cube<<<blockDim, nullptr, stream>>>(
      k, workspace, cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}
