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
AICORE void matmul_kernel(__gm__ half *a_packed, __gm__ half *x_bsnd,
                          __gm__ float *out_packed, __gm__ int32_t *cu_seqlens,
                          int64_t batch_size, int64_t fixed_seq_len,
                          uint64_t ffts_addr) {
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t AL1Addr = 0;
  constexpr int32_t XL1Addr = 32768;

  using PackedA = GlobalTensor<half, TileShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                               BaseShape2D<half, ChunkSize, ChunkSize, Layout::ND>, Layout::ND>;
  using PackedOut = GlobalTensor<float, TileShape2D<float, ChunkSize, HiddenSize, Layout::ND>,
                                 BaseShape2D<float, ChunkSize, HiddenSize, Layout::ND>, Layout::ND>;
  using XGlobalShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using XGlobalStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using XGlobal = GlobalTensor<half, XGlobalShape, XGlobalStride, Layout::ND>;
  using AL1 = GdnL1Mat<half, ChunkSize, ChunkSize>;
  using XL1 = GdnL1Mat<half, ChunkSize, HiddenSize>;
  using ADynL1 = Tile<TileType::Mat, half, ChunkSize, ChunkSize, BLayout::ColMajor,
                      DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  using XDynL1 = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                      DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t total_work = batch_size * NumHeads;

  AL1 a_l1;
  XL1 x_l1;
  TASSIGN(a_l1, AL1Addr);
  TASSIGN(x_l1, XL1Addr);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> out_l0;
  TASSIGN(out_l0, 0);

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
      const int32_t packed_chunk_base = static_cast<int32_t>(
          (seq.chunk_offset + chunk_idx) * NumHeads + head_idx);
      const int32_t a_offset = packed_chunk_base * ChunkSquareElems;
      const int32_t x_offset = static_cast<int32_t>(
          (seq.bos + row_start) * NumHeads * HiddenSize + head_idx * HiddenSize);
      const int32_t out_offset = packed_chunk_base * ChunkHiddenElems;

      ADynL1 a_dyn(valid_rows, ChunkSize);
      XDynL1 x_dyn(valid_rows, HiddenSize);
      TASSIGN(a_dyn, AL1Addr);
      TASSIGN(x_dyn, XL1Addr);
      PackedA a_global(a_packed + a_offset);
      XGlobal x_global(
          x_bsnd + x_offset,
          {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, NumHeads * HiddenSize, 1});
      TLOAD(a_dyn, a_global);
      TLOAD(x_dyn, x_global);
      pipe_barrier(PIPE_ALL);

      GdnMatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(out_l0, a_l1,
                                                                  x_l1, true);
      PackedOut out_global(out_packed + out_offset);
      TSTORE(out_global, out_l0);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_wy_fast_matmul(
    __gm__ uint8_t *a_packed, __gm__ uint8_t *x_bsnd, __gm__ uint8_t *out_packed,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t fixed_seq_len,
    uint64_t ffts_addr) {
  matmul_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(a_packed),
      reinterpret_cast<__gm__ half *>(x_bsnd),
      reinterpret_cast<__gm__ float *>(out_packed), cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" void call_matmul_kernel(uint32_t blockDim, void *stream, uint8_t *a_packed,
                                   uint8_t *x_bsnd, uint8_t *out_packed,
                                   int32_t *cu_seqlens, int64_t batch_size,
                                   int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_wy_fast_matmul<<<blockDim, nullptr, stream>>>(
      a_packed, x_bsnd, out_packed, cu_seqlens, batch_size, fixed_seq_len,
      ffts_addr);
}
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
AICORE void main_kernel(__gm__ half *k, __gm__ half *v, __gm__ half *beta,
                        __gm__ float *g_packed, __gm__ half *a_packed,
                        __gm__ half *workspace_a1, __gm__ half *workspace_a2,
                        __gm__ half *w_out, __gm__ half *u_out,
                        __gm__ int32_t *cu_seqlens, int64_t batch_size,
                        int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t QL1Addr = 0;
  constexpr int32_t XL1Addr = 32768;

  constexpr int32_t BetaHalfUbAddr = 0;
  constexpr int32_t BetaLocalHalfUbAddr =
      BetaHalfUbAddr + HalfChunk * NumHeads * sizeof(half);
  constexpr int32_t AUbHalfAddr = BetaLocalHalfUbAddr + HalfChunk * sizeof(half);
  constexpr int32_t BetaUbAddr = AUbHalfAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t Beta2dUbAddr = BetaUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t A1UbAddr = Beta2dUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t A2UbAddr = A1UbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t A2HalfUbAddr = A2UbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t GUbAddr = A2HalfUbAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t G2dUbAddr = GUbAddr + HalfChunk * sizeof(float);

  using PackedA =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using PackedAFull =
      GlobalTensor<half, TileShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, ChunkSize, Layout::ND>, Layout::ND>;
  using GLocalGlobalShape = Shape<1, 1, 1, 1, DYNAMIC>;
  using GLocalGlobalStride = Stride<1, 1, 1, 1, 1>;
  using GLocalGlobal =
      GlobalTensor<float, GLocalGlobalShape, GLocalGlobalStride, Layout::ND>;
  using PackedOut =
      GlobalTensor<half, TileShape2D<half, ChunkSize, HiddenSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, HiddenSize, Layout::ND>, Layout::ND>;
  using PackedOutDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using PackedOutDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using PackedOutDyn =
      GlobalTensor<half, PackedOutDynShape, PackedOutDynStride, Layout::ND>;
  using ChunkGlobalDynShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using ChunkGlobalDynStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using ChunkGlobalDyn =
      GlobalTensor<half, ChunkGlobalDynShape, ChunkGlobalDynStride, Layout::ND>;
  using BetaFlatGlobalShape = Shape<1, 1, 1, 1, DYNAMIC>;
  using BetaFlatGlobalStride = Stride<1, 1, 1, 1, 1>;
  using BetaFlatGlobal =
      GlobalTensor<half, BetaFlatGlobalShape, BetaFlatGlobalStride, Layout::ND>;
  using BetaFlatUb = GdnUbND<half, 1, HalfChunk * NumHeads>;
  using BetaHalfUb = GdnUbND<half, 1, HalfChunk>;
  using BetaUb = GdnUbND<float, 1, HalfChunk>;
  using AHalfUb = GdnUbND<half, HalfChunk, ChunkSize>;
  using AFloatUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using GUb = GdnUbND<float, 1, HalfChunk>;
  using GColUb = GdnUbDN<float, HalfChunk, 1>;
  using Beta2dUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using G2dUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using RowSliceUb = GdnUbND<float, 1, ChunkSize>;
  using AFullL1 = GdnL1Mat<half, ChunkSize, ChunkSize>;
  using XFullL1 = GdnL1Mat<half, ChunkSize, HiddenSize>;
  using ADynL1 = Tile<TileType::Mat, half, ChunkSize, ChunkSize, BLayout::ColMajor,
                      DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  using XDynL1 = Tile<TileType::Mat, half, ChunkSize, HiddenSize, BLayout::ColMajor,
                      DYNAMIC, DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * NumHeads;

  AFullL1 a_l1;
  XFullL1 x_l1;
  TASSIGN(a_l1, QL1Addr);
  TASSIGN(x_l1, XL1Addr);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> out_l0;
  TASSIGN(out_l0, 0);

  AHalfUb a_half_ub;
  AFloatUb a1_ub;
  AFloatUb a2_ub;
  AHalfUb a2_half_ub;
  BetaFlatUb beta_block_ub;
  BetaHalfUb beta_half_ub;
  BetaUb beta_ub;
  GUb g_ub;
  GColUb beta_col_ub;
  GColUb g_col_ub;
  Beta2dUb beta_2d_ub;
  G2dUb g_2d_ub;
  AHalfUb a1_half_ub;
  TASSIGN(beta_block_ub, BetaHalfUbAddr);
  TASSIGN(beta_half_ub, BetaLocalHalfUbAddr);
  TASSIGN(a_half_ub, AUbHalfAddr);
  TASSIGN(a1_ub, A1UbAddr);
  TASSIGN(a2_ub, A2UbAddr);
  TASSIGN(a2_half_ub, A2HalfUbAddr);
  TASSIGN(beta_ub, BetaUbAddr);
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(beta_col_ub, BetaUbAddr);
  TASSIGN(g_col_ub, GUbAddr);
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  TASSIGN(g_2d_ub, G2dUbAddr);
  TASSIGN(a1_half_ub, AUbHalfAddr);

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

      if (local_rows == 0) {
        GdnSetCrossFlag<PIPE_MTE3>(2, 2);
        GdnSetCrossFlag<PIPE_MTE3>(1, 2);
        continue;
      }

      PackedA a_global(a_packed + chunk_base * ChunkSquareElems +
                       row_offset * ChunkSize);
      PackedA a1_global(workspace_a1 + chunk_base * ChunkSquareElems +
                        row_offset * ChunkSize);
      PackedA a2_global(workspace_a2 + chunk_base * ChunkSquareElems +
                        row_offset * ChunkSize);
      GLocalGlobal g_global(g_packed + chunk_base * ChunkSize + row_offset,
                            {1, 1, 1, 1, static_cast<int32_t>(local_rows)},
                            {1, 1, 1, 1, 1});
      BetaFlatGlobal beta_global(
          beta + (seq.bos + row_start + row_offset) * NumHeads,
          {1, 1, 1, 1, static_cast<int32_t>(local_rows * NumHeads)},
          {1, 1, 1, 1, 1});

      TLOAD(beta_block_ub, beta_global);
      TLOAD(a_half_ub, a_global);
      TLOAD(g_ub, g_global);
      GdnSetFlag<PIPE_MTE2, PIPE_V>(0);
      GdnWaitFlag<PIPE_MTE2, PIPE_V>(0);

      for (uint32_t i = 0; i < HalfChunk; ++i) {
        beta_half_ub.SetValue(i, static_cast<half>(0.0f));
      }
      for (uint32_t i = 0; i < local_rows; ++i) {
        beta_half_ub.SetValue(i,
                              beta_block_ub.GetValue(i * NumHeads + head_idx));
      }
      pipe_barrier(PIPE_V);
      TCVT(beta_ub, beta_half_ub, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
      TCVT(a1_ub, a_half_ub, pto::RoundMode::CAST_NONE);
      TMOV(a2_ub, a1_ub);
      for (uint32_t row = 0; row < HalfChunk; ++row) {
        RowSliceUb a2_row;
        TASSIGN(a2_row, A2UbAddr + row * ChunkSize * sizeof(float));
        TMULS(a2_row, a2_row, row < local_rows ? beta_ub.GetValue(row) : 0.0f);
      }
      pipe_barrier(PIPE_V);
      TCVT(a2_half_ub, a2_ub, pto::RoundMode::CAST_NONE);
      GdnSetFlag<PIPE_V, PIPE_MTE3>(0);
      GdnWaitFlag<PIPE_V, PIPE_MTE3>(0);
      TSTORE(a2_global, a2_half_ub);
      pipe_barrier(PIPE_ALL);
      GdnSetCrossFlag<PIPE_MTE3>(2, 2);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      const float g_first = g_ub.GetValue(0);
      TEXP(g_ub, g_ub);
      pipe_barrier(PIPE_V);
      RowSliceUb g_exp_patch;
      TASSIGN(g_exp_patch, Beta2dUbAddr);
      TEXPANDS(g_exp_patch, 0.0f);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      g_exp_patch.SetValue(1, g_first);
      pipe_barrier(PIPE_V);
      TEXP(g_exp_patch, g_exp_patch);
      pipe_barrier(PIPE_V);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      g_ub.SetValue(0, g_exp_patch.GetValue(1));
      pipe_barrier(PIPE_V);
      for (uint32_t row = 0; row < HalfChunk; ++row) {
        RowSliceUb a1_row;
        TASSIGN(a1_row, A1UbAddr + row * ChunkSize * sizeof(float));
        TMULS(a1_row, a1_row, row < local_rows ? g_ub.GetValue(row) : 0.0f);
      }
      pipe_barrier(PIPE_V);
      TCVT(a1_half_ub, a1_ub, pto::RoundMode::CAST_NONE);
      GdnSetFlag<PIPE_V, PIPE_MTE3>(1);
      GdnWaitFlag<PIPE_V, PIPE_MTE3>(1);
      TSTORE(a1_global, a1_half_ub);
      pipe_barrier(PIPE_ALL);
      GdnSetCrossFlag<PIPE_MTE3>(1, 2);
    }
  }
#endif

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
      const int32_t chunk_base =
          static_cast<int32_t>((seq.chunk_offset + chunk_idx) * NumHeads + head_idx);
      const int32_t token_offset =
          static_cast<int32_t>(seq.token_base_offset + row_start * seq.row_stride);

      XDynL1 x_dyn(valid_rows, HiddenSize);
      ADynL1 a_dyn(valid_rows, ChunkSize);
      TASSIGN(x_dyn, XL1Addr);
      TASSIGN(a_dyn, QL1Addr);
      ChunkGlobalDyn xk_global(
          k + token_offset, {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
      ChunkGlobalDyn xv_global(
          v + token_offset, {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, static_cast<int32_t>(seq.row_stride), 1});
      PackedAFull a1_global(workspace_a1 + chunk_base * ChunkSquareElems);
      PackedAFull a2_global(workspace_a2 + chunk_base * ChunkSquareElems);

      GdnWaitCrossFlag(2);
      TLOAD(a_dyn, a2_global);
      TLOAD(x_dyn, xv_global);
      pipe_barrier(PIPE_ALL);
      GdnMatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(out_l0, a_l1,
                                                                  x_l1, true);
      PackedOutDyn u_global(
          u_out + chunk_base * ChunkHiddenElems,
          {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, HiddenSize, 1});
      TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> u_tail(valid_rows,
                                                                      HiddenSize);
      TASSIGN(u_tail, 0);
      TSTORE(u_global, u_tail);
      pipe_barrier(PIPE_ALL);

      GdnWaitCrossFlag(1);
      TLOAD(a_dyn, a1_global);
      TLOAD(x_dyn, xk_global);
      pipe_barrier(PIPE_ALL);
      GdnMatmulL1<ChunkSize, HiddenSize, ChunkSize, false, false>(out_l0, a_l1,
                                                                  x_l1, true);
      PackedOutDyn w_global(
          w_out + chunk_base * ChunkHiddenElems,
          {1, 1, 1, static_cast<int32_t>(valid_rows), HiddenSize},
          {1, 1, 1, HiddenSize, 1});
      TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> w_tail(valid_rows,
                                                                      HiddenSize);
      TASSIGN(w_tail, 0);
      TSTORE(w_global, w_tail);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_wy_fast(
    __gm__ uint8_t *k, __gm__ uint8_t *v, __gm__ uint8_t *beta,
    __gm__ uint8_t *g_packed, __gm__ uint8_t *a_packed,
    __gm__ uint8_t *workspace_a1, __gm__ uint8_t *workspace_a2,
    __gm__ uint8_t *w_out, __gm__ uint8_t *u_out, __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t fixed_seq_len, uint64_t ffts_addr) {
  main_kernel<GDN_H, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(k), reinterpret_cast<__gm__ half *>(v),
      reinterpret_cast<__gm__ half *>(beta),
      reinterpret_cast<__gm__ float *>(g_packed),
      reinterpret_cast<__gm__ half *>(a_packed),
      reinterpret_cast<__gm__ half *>(workspace_a1),
      reinterpret_cast<__gm__ half *>(workspace_a2),
      reinterpret_cast<__gm__ half *>(w_out),
      reinterpret_cast<__gm__ half *>(u_out), cu_seqlens, batch_size,
      fixed_seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *k,
                            uint8_t *v, uint8_t *beta, uint8_t *g_packed,
                            uint8_t *a_packed, uint8_t *workspace_a1,
                            uint8_t *workspace_a2, uint8_t *w_out,
                            uint8_t *u_out, int32_t *cu_seqlens,
                            int64_t batch_size, int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_wy_fast<<<blockDim, nullptr, stream>>>(
      k, v, beta, g_packed, a_packed, workspace_a1, workspace_a2, w_out, u_out,
      cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}
