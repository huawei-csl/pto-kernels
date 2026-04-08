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
  constexpr int32_t HeadTileCols = ((NumHeads + 15) / 16) * 16;
  constexpr int32_t ChunkSquareElems = ChunkSize * ChunkSize;
  constexpr int32_t ChunkHiddenElems = ChunkSize * HiddenSize;
  constexpr int32_t QL1Addr = 0;
  constexpr int32_t XL1Addr = 32768;

  constexpr int32_t BetaHalfUbAddr = 0;
  constexpr int32_t AUbHalfAddr = BetaHalfUbAddr + ChunkSize * sizeof(half);
  constexpr int32_t BetaUbAddr = AUbHalfAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t BetaRowUbAddr = BetaUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t Beta2dUbAddr = BetaRowUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t TmpUbAddr = Beta2dUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t A1UbAddr = TmpUbAddr + 24576 * sizeof(uint8_t);
  constexpr int32_t A2UbAddr = A1UbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t A2HalfUbAddr = A2UbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t GUbAddr = A2HalfUbAddr + HalfChunk * ChunkSize * sizeof(half);
  constexpr int32_t GRowUbAddr = GUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t G2dUbAddr = GRowUbAddr + ChunkSize * sizeof(float);

  using PackedA =
      GlobalTensor<half, TileShape2D<half, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<half, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using PackedAFull =
      GlobalTensor<half, TileShape2D<half, ChunkSize, ChunkSize, Layout::ND>,
                   BaseShape2D<half, ChunkSize, ChunkSize, Layout::ND>, Layout::ND>;
  using PackedGGlobal =
      GlobalTensor<float, TileShape2D<float, 1, ChunkSize, Layout::ND>,
                   BaseShape2D<float, 1, ChunkSize, Layout::ND>, Layout::ND>;
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
  using BetaBlockShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using BetaBlockStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using BetaBlockGlobal =
      GlobalTensor<half, BetaBlockShape, BetaBlockStride, Layout::ND>;
  using BetaBlockUb =
      Tile<TileType::Vec, half, ChunkSize, HeadTileCols, BLayout::RowMajor,
           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using BetaUb =
      Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, DYNAMIC,
           DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using AHalfUb = GdnUbND<half, HalfChunk, ChunkSize>;
  using AFloatUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using GUb =
      Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, DYNAMIC,
           DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using Beta2dUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using G2dUb = GdnUbND<float, HalfChunk, ChunkSize>;
  using GRowUb = GdnUbND<float, 1, ChunkSize>;
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
  BetaUb beta_ub(1, ChunkSize);
  GUb g_ub(1, ChunkSize);
  GRowUb beta_r_ub;
  GRowUb g_r_ub;
  Beta2dUb beta_2d_ub;
  G2dUb g_2d_ub;
  GdnUbND<uint8_t, 1, 24576> tmp_ub;
  TASSIGN(a_half_ub, AUbHalfAddr);
  TASSIGN(a1_ub, A1UbAddr);
  TASSIGN(a2_ub, A2UbAddr);
  TASSIGN(a2_half_ub, A2HalfUbAddr);
  TASSIGN(beta_ub, BetaUbAddr);
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(beta_r_ub, BetaRowUbAddr);
  TASSIGN(g_r_ub, GRowUbAddr);
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  TASSIGN(g_2d_ub, G2dUbAddr);
  TASSIGN(tmp_ub, TmpUbAddr);

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

      PackedA a_global(a_packed + chunk_base * ChunkSquareElems +
                       row_offset * ChunkSize);
      PackedA a1_global(workspace_a1 + chunk_base * ChunkSquareElems +
                        row_offset * ChunkSize);
      PackedA a2_global(workspace_a2 + chunk_base * ChunkSquareElems +
                        row_offset * ChunkSize);
      PackedGGlobal g_global(g_packed + chunk_base * ChunkSize);
      BetaBlockGlobal beta_global(
          beta + (seq.bos + row_start) * NumHeads + head_idx,
          {1, 1, 1, static_cast<int32_t>(valid_rows), NumHeads},
          {1, 1, 1, NumHeads, 1});
      BetaBlockUb beta_block_ub(valid_rows, NumHeads);
      TASSIGN(beta_block_ub, BetaHalfUbAddr);

      TLOAD(a_half_ub, a_global);
      TLOAD(g_ub, g_global);
      TLOAD(beta_block_ub, beta_global);
      pipe_barrier(PIPE_ALL);

      for (uint32_t i = 0; i < ChunkSize; ++i) {
        beta_ub.SetValue(i, 0.0f);
      }
      for (uint32_t i = 0; i < valid_rows; ++i) {
        beta_ub.SetValue(
            i, static_cast<float>(
                   beta_block_ub.GetValue(i * HeadTileCols + head_idx)));
      }
      pipe_barrier(PIPE_V);
      TCVT(a1_ub, a_half_ub, pto::RoundMode::CAST_NONE);
      TMOV(beta_r_ub, beta_ub);
      TCOLEXPAND(beta_2d_ub, beta_r_ub);
      TMUL(a2_ub, a1_ub, beta_2d_ub);
      pipe_barrier(PIPE_V);
      TCVT(a2_half_ub, a2_ub, pto::RoundMode::CAST_NONE);
      TSTORE(a2_global, a2_half_ub);
      pipe_barrier(PIPE_ALL);
      GdnSetCrossFlag<PIPE_MTE3>(2, 2);

      TEXP(g_ub, g_ub);
      pipe_barrier(PIPE_V);
      TMUL(g_ub, g_ub, beta_ub);
      TMOV(g_r_ub, g_ub);
      TCOLEXPAND(g_2d_ub, g_r_ub);
      TMUL(a1_ub, a1_ub, g_2d_ub);
      pipe_barrier(PIPE_V);
      TCVT(a_half_ub, a1_ub, pto::RoundMode::CAST_NONE);
      TSTORE(a1_global, a_half_ub);
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
