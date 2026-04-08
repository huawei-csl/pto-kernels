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
AICORE void main_kernel(__gm__ half *beta, __gm__ half *out,
                        __gm__ int32_t *cu_seqlens, int64_t batch_size,
                        int64_t fixed_seq_len, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t HeadTileCols = ((NumHeads + 15) / 16) * 16;
  constexpr int32_t BetaUbAddr = 0;

  using BetaBlockShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using BetaBlockStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using BetaBlockGlobal =
      GlobalTensor<half, BetaBlockShape, BetaBlockStride, Layout::ND>;
  using OutBlockGlobal =
      GlobalTensor<half, BetaBlockShape, BetaBlockStride, Layout::ND>;
  using BetaBlockUb =
      Tile<TileType::Vec, half, HalfChunk, HeadTileCols, BLayout::RowMajor,
           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t total_work = batch_size * NumHeads;

  BetaBlockUb beta_ub(HalfChunk, NumHeads);
  TASSIGN(beta_ub, BetaUbAddr);

#if defined(__DAV_C220_VEC__)
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
      const uint32_t row_offset = 0;
      const uint32_t local_rows =
          valid_rows > row_offset
              ? GdnMinU32(static_cast<uint32_t>(valid_rows - row_offset),
                          static_cast<uint32_t>(HalfChunk))
              : 0;
      if (local_rows == 0) {
        continue;
      }

      const int32_t beta_offset = static_cast<int32_t>(
          (seq.bos + row_start + row_offset) * NumHeads);
      const int32_t out_offset = static_cast<int32_t>(
          (((seq.chunk_offset + chunk_idx) * NumHeads + head_idx) * HalfChunk *
           NumHeads));

      BetaBlockGlobal beta_global(
          beta + beta_offset,
          {1, 1, 1, static_cast<int32_t>(local_rows), NumHeads},
          {1, 1, 1, NumHeads, 1});
      OutBlockGlobal out_global(
          out + out_offset,
          {1, 1, 1, static_cast<int32_t>(local_rows), NumHeads},
          {1, 1, 1, NumHeads, 1});
      TLOAD(beta_ub, beta_global);
      pipe_barrier(PIPE_ALL);
      TSTORE(out_global, beta_ub);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_debug_beta_block(
    __gm__ uint8_t *beta, __gm__ uint8_t *out, __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t fixed_seq_len, uint64_t ffts_addr) {
  main_kernel<GDN_H, GDN_C>(reinterpret_cast<__gm__ half *>(beta),
                            reinterpret_cast<__gm__ half *>(out), cu_seqlens,
                            batch_size, fixed_seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *beta,
                            uint8_t *out, int32_t *cu_seqlens,
                            int64_t batch_size, int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_debug_beta_block<<<blockDim, nullptr, stream>>>(
      beta, out, cu_seqlens, batch_size, fixed_seq_len, ffts_addr);
}
