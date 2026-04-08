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
