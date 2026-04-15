#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

#include "gdn_seq_info.h"

using namespace pto;

#ifndef GDN_H
#define GDN_H 2
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

template <int32_t NumHeads, int32_t ChunkSize>
AICORE void main_kernel(__gm__ float *g, __gm__ float *s, __gm__ int32_t *cu_seqlens,
                        int64_t batch_size, int64_t fixed_seq_len,
                        uint64_t ffts_addr) {
  constexpr int32_t VecNum = 2;
  constexpr int32_t HeadTileCols = ((NumHeads + 7) / 8) * 8;
  static_assert((NumHeads % VecNum) == 0, "GDN_H must be divisible by 2.");

  using ChunkHeadBlockDyn =
      Tile<TileType::Vec, float, ChunkSize, HeadTileCols, BLayout::RowMajor,
           DYNAMIC, DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using ChunkOutDyn =
      Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, DYNAMIC,
           DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  using ChunkGlobalShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using ChunkInStride = Stride<1, 1, 1, DYNAMIC, 1>;
  using ChunkOutStride = Stride<1, 1, 1, 1, 1>;
  using ChunkInGlobal = GlobalTensor<float, ChunkGlobalShape, ChunkInStride, Layout::ND>;
  using ChunkOutGlobal = GlobalTensor<float, ChunkGlobalShape, ChunkOutStride, Layout::ND>;

  constexpr int32_t GUbAddr = 0;
  constexpr int32_t SUbAddr = GUbAddr + ChunkSize * HeadTileCols * sizeof(float);

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  const int64_t total_work = batch_size * (NumHeads / VecNum);

  ChunkHeadBlockDyn g_ub(ChunkSize, NumHeads);
  TASSIGN(g_ub, GUbAddr);

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  for (int64_t work_idx = 0; work_idx < (total_work + block_num - 1) / block_num;
       ++work_idx) {
    const int64_t pid = work_idx * block_num + cid;
    if (pid >= total_work) {
      continue;
    }

    const uint32_t head_pair_idx = static_cast<uint32_t>(pid % (NumHeads / VecNum));
    const uint32_t seq_idx = static_cast<uint32_t>(pid / (NumHeads / VecNum));
    const uint32_t head_idx = head_pair_idx * VecNum + static_cast<uint32_t>(vid);
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
          (seq.bos + row_start) * NumHeads);
      const int32_t out_offset = static_cast<int32_t>(
          ((seq.chunk_offset + chunk_idx) * NumHeads + head_idx) * ChunkSize);

      ChunkInGlobal g_global(g + token_offset,
                             {1, 1, 1, static_cast<int32_t>(valid_rows), NumHeads},
                             {1, 1, 1, NumHeads, 1});
      ChunkOutGlobal s_global(s + out_offset,
                              {1, 1, 1, 1, static_cast<int32_t>(valid_rows)},
                              {1, 1, 1, 1, 1});
      ChunkOutDyn s_ub(1, valid_rows);
      TASSIGN(s_ub, SUbAddr);
      TLOAD(g_ub, g_global);
      pipe_barrier(PIPE_ALL);

      s_ub.SetValue(0, g_ub.GetValue(head_idx));
      for (uint32_t i = 1; i < valid_rows; ++i) {
        const float next =
            s_ub.GetValue(i - 1) +
            g_ub.GetValue(i * HeadTileCols + head_idx);
        s_ub.SetValue(i, next);
      }
      pipe_barrier(PIPE_ALL);
      TSTORE(s_global, s_ub);
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_chunk_cumsum(
    __gm__ uint8_t *g, __gm__ uint8_t *s, __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t fixed_seq_len, uint64_t ffts_addr) {
  main_kernel<GDN_H, GDN_C>(reinterpret_cast<__gm__ float *>(g),
                            reinterpret_cast<__gm__ float *>(s), cu_seqlens,
                            batch_size, fixed_seq_len, ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *g, uint8_t *s,
                            int32_t *cu_seqlens, int64_t batch_size,
                            int64_t fixed_seq_len) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_chunk_cumsum<<<blockDim, nullptr, stream>>>(g, s, cu_seqlens,
                                                     batch_size, fixed_seq_len,
                                                     ffts_addr);
}
