#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

using namespace pto;

#ifndef GDN_H
#define GDN_H 2
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

template <int32_t NumHeads, int32_t ChunkSize>
AICORE void main_kernel(__gm__ float *g, __gm__ float *out, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t GUbAddr = 0;
  constexpr int32_t GvUbAddr = GUbAddr + ChunkSize * sizeof(float);

  using PackedGGlobal =
      GlobalTensor<float, TileShape2D<float, 1, ChunkSize, Layout::ND>,
                   BaseShape2D<float, 1, ChunkSize, Layout::ND>, Layout::ND>;
  using OutGlobal =
      GlobalTensor<float, TileShape2D<float, 1, HalfChunk, Layout::ND>,
                   BaseShape2D<float, 1, HalfChunk, Layout::ND>, Layout::ND>;
  using GUb = Tile<TileType::Vec, float, 1, ChunkSize, BLayout::RowMajor, 1,
                   ChunkSize, SLayout::NoneBox, 512, PadValue::Zero>;
  using GHalfUb = Tile<TileType::Vec, float, 1, HalfChunk, BLayout::RowMajor, 1,
                       HalfChunk, SLayout::NoneBox, 512, PadValue::Zero>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  GUb g_ub;
  GHalfUb g_v_ub;
  TASSIGN(g_ub, GUbAddr);
  TASSIGN(g_v_ub, GvUbAddr);

#if defined(__DAV_C220_VEC__)
  PackedGGlobal g_global(g + cid * ChunkSize);
  TLOAD(g_ub, g_global);
  pipe_barrier(PIPE_ALL);
  GHalfUb g_ub_temp;
  TASSIGN(g_ub_temp, GUbAddr + vid * HalfChunk * sizeof(float));
  TMOV(g_v_ub, g_ub_temp);
  pipe_barrier(PIPE_V);
  OutGlobal out_global(out + cid * ChunkSize + vid * HalfChunk);
  TSTORE(out_global, g_v_ub);
  pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ AICORE void launch_debug_g_slice(__gm__ uint8_t *g,
                                                        __gm__ uint8_t *out,
                                                        uint64_t ffts_addr) {
  main_kernel<GDN_H, GDN_C>(reinterpret_cast<__gm__ float *>(g),
                            reinterpret_cast<__gm__ float *>(out), ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *g,
                            uint8_t *out) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_debug_g_slice<<<blockDim, nullptr, stream>>>(g, out, ffts_addr);
}
