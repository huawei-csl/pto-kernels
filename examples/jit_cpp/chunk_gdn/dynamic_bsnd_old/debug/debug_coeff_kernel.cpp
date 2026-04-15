#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

#include "../gdn_seq_info.h"

using namespace pto;

#ifndef GDN_H
#define GDN_H 2
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

AICORE inline uint32_t GdnMinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

template <int32_t NumHeads, int32_t ChunkSize>
AICORE void main_kernel(__gm__ half *beta, __gm__ float *g, __gm__ float *out,
                        __gm__ int32_t *cu_seqlens, int64_t batch_size,
                        int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t HeadTileCols = ((NumHeads + 15) / 16) * 16;
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t BetaHalfUbAddr = GUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t BetaUbAddr = BetaHalfUbAddr + HalfChunk * HeadTileCols * sizeof(half);
  constexpr int32_t GvUbAddr = BetaUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t GRUbAddr = GvUbAddr + HalfChunk * sizeof(float);
  constexpr int32_t GCUbAddr = GRUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t GR2dUbAddr = GCUbAddr + ChunkSize * sizeof(float);
  constexpr int32_t GC2dUbAddr = GR2dUbAddr + HalfChunk * ChunkSize * sizeof(float);
  constexpr int32_t CoeffUbAddr = GC2dUbAddr + HalfChunk * ChunkSize * sizeof(float);

  using PackedGGlobal =
      GlobalTensor<float, TileShape2D<float, 1, ChunkSize, Layout::ND>,
                   BaseShape2D<float, 1, ChunkSize, Layout::ND>, Layout::ND>;
  using BetaBlockShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using BetaBlockStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using BetaBlockGlobal = GlobalTensor<half, BetaBlockShape, BetaBlockStride, Layout::ND>;
  using OutGlobal =
      GlobalTensor<float, TileShape2D<float, HalfChunk, ChunkSize, Layout::ND>,
                   BaseShape2D<float, HalfChunk, ChunkSize, Layout::ND>, Layout::ND>;
  using GUb = Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, DYNAMIC,
                   DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using GHalfUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor, DYNAMIC,
                       DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using GHalfRowUb =
      Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor, DYNAMIC,
           DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using BetaBlockUb = Tile<TileType::Vec, half, HalfChunk, HeadTileCols, BLayout::RowMajor,
                           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using BetaUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor,
                      DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using AUb = Tile<TileType::Vec, float, HalfChunk, ChunkSize, BLayout::RowMajor,
                   HalfChunk, ChunkSize, SLayout::NoneBox, 512, PadValue::Null>;
  using GColUb = Tile<TileType::Vec, float, HalfChunk, 1, BLayout::ColMajor,
                      HalfChunk, 1, SLayout::NoneBox, 512, PadValue::Null>;
  using GRowUb = Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor,
                      1, ChunkSize, SLayout::NoneBox, 512, PadValue::Null>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * NumHeads;

  GUb g_ub(1, ChunkSize);
  BetaBlockUb beta_block_ub(HalfChunk, NumHeads);
  BetaUb beta_ub(1, HalfChunk);
  GHalfUb g_v_ub(1, HalfChunk);
  GColUb g_r_col_ub;
  GHalfRowUb g_r_row_ub(1, HalfChunk);
  GRowUb g_c_ub;
  AUb g_r_2d_ub;
  AUb g_c_2d_ub;
  AUb coeff_ub;
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(beta_block_ub, BetaHalfUbAddr);
  TASSIGN(beta_ub, BetaUbAddr);
  TASSIGN(g_v_ub, GvUbAddr);
  TASSIGN(g_r_col_ub, GRUbAddr);
  TASSIGN(g_r_row_ub, GRUbAddr);
  TASSIGN(g_c_ub, GCUbAddr);
  TASSIGN(g_r_2d_ub, GR2dUbAddr);
  TASSIGN(g_c_2d_ub, GC2dUbAddr);
  TASSIGN(coeff_ub, CoeffUbAddr);

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
      const uint32_t valid_rows =
          GdnMinU32(static_cast<uint32_t>(seq.seq_len - row_start),
                    static_cast<uint32_t>(ChunkSize));
      const uint32_t row_offset = static_cast<uint32_t>(vid) * HalfChunk;
      const uint32_t local_rows =
          valid_rows > row_offset
              ? GdnMinU32(static_cast<uint32_t>(valid_rows - row_offset),
                          static_cast<uint32_t>(HalfChunk))
              : 0;
      if (local_rows == 0) continue;

      const int32_t chunk_base =
          static_cast<int32_t>((seq.chunk_offset + chunk_idx) * NumHeads + head_idx);
      const int32_t beta_offset = static_cast<int32_t>(
          (seq.bos + row_start + row_offset) * NumHeads);
      PackedGGlobal g_global(g + chunk_base * ChunkSize);
      BetaBlockGlobal beta_global(
          beta + beta_offset,
          {1, 1, 1, static_cast<int32_t>(local_rows), NumHeads},
          {1, 1, 1, NumHeads, 1});
      OutGlobal out_global(out + chunk_base * ChunkSize * ChunkSize +
                           row_offset * ChunkSize);

      TLOAD(g_ub, g_global);
      TLOAD(beta_block_ub, beta_global);
      pipe_barrier(PIPE_ALL);
      GHalfUb g_ub_temp(1, local_rows);
      TASSIGN(g_ub_temp, GUbAddr + row_offset * sizeof(float));
      TMOV(g_v_ub, g_ub_temp);
      pipe_barrier(PIPE_V);
      for (uint32_t row = 0; row < local_rows; ++row) {
        beta_ub.SetValue(row, static_cast<float>(beta_block_ub.GetValue(row * HeadTileCols + head_idx)));
      }
      pipe_barrier(PIPE_V);
      TEXPANDS(coeff_ub, 0.0f);
      pipe_barrier(PIPE_V);
      for (uint32_t row = 0; row < local_rows; ++row) {
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
      for (uint32_t row = 0; row < local_rows; ++row) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        GRowUb coeff_row;
        TASSIGN(coeff_row, GC2dUbAddr + row * ChunkSize * sizeof(float));
        TMULS(coeff_row, coeff_row,
              static_cast<float>(
                  beta_block_ub.GetValue(row * HeadTileCols + head_idx)));
      }
      pipe_barrier(PIPE_V);
      TSTORE(out_global, g_c_2d_ub);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_debug_coeff(
    __gm__ uint8_t *beta, __gm__ uint8_t *g, __gm__ uint8_t *out,
    __gm__ int32_t *cu_seqlens, int64_t batch_size, int64_t fixed_seq_len,
    uint64_t ffts_addr) {
  main_kernel<GDN_H, GDN_C>(reinterpret_cast<__gm__ half *>(beta),
                            reinterpret_cast<__gm__ float *>(g),
                            reinterpret_cast<__gm__ float *>(out), cu_seqlens,
                            batch_size, fixed_seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *beta,
                            uint8_t *g, uint8_t *out, int32_t *cu_seqlens,
                            int64_t batch_size, int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_debug_coeff<<<blockDim, nullptr, stream>>>(
      beta, g, out, cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}
