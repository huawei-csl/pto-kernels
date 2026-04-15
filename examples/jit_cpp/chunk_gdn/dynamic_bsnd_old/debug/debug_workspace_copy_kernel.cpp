#include <pto/pto-inst.hpp>
#include <runtime/rt_ffts.h>

using namespace pto;

#ifndef GDN_C
#define GDN_C 128
#endif

AICORE void main_kernel(__gm__ half *workspace, __gm__ half *out, uint64_t ffts_addr) {
  constexpr int32_t HalfChunk = GDN_C / 2;
  constexpr int32_t ChunkSquareElems = GDN_C * GDN_C;
  constexpr int32_t AUbHalfAddr = 0;
  using HalfBlockGlobal =
      GlobalTensor<half, TileShape2D<half, HalfChunk, GDN_C, Layout::ND>,
                   BaseShape2D<half, HalfChunk, GDN_C, Layout::ND>, Layout::ND>;
  using AHalfUb =
      Tile<TileType::Vec, half, HalfChunk, GDN_C, BLayout::RowMajor, HalfChunk,
           GDN_C, SLayout::NoneBox, 512, PadValue::Null>;

  set_ffts_base_addr(ffts_addr);
  const int64_t cid = get_block_idx();
  const int64_t vid = get_subblockid();
  AHalfUb a_half_ub;
  TASSIGN(a_half_ub, AUbHalfAddr);

#if defined(__DAV_C220_VEC__)
  HalfBlockGlobal workspace_global(workspace + cid * ChunkSquareElems +
                                   vid * HalfChunk * GDN_C);
  HalfBlockGlobal out_global(out + cid * ChunkSquareElems +
                             vid * HalfChunk * GDN_C);
  TLOAD(a_half_ub, workspace_global);
  pipe_barrier(PIPE_ALL);
  TSTORE(out_global, a_half_ub);
  pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ AICORE void launch_debug_workspace_copy(
    __gm__ uint8_t *workspace, __gm__ uint8_t *out, uint64_t ffts_addr) {
  main_kernel(reinterpret_cast<__gm__ half *>(workspace),
              reinterpret_cast<__gm__ half *>(out), ffts_addr);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *workspace,
                            uint8_t *out) {
  uint32_t ffts_len = 0;
  uint64_t ffts_addr = 0;
  rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);
  launch_debug_workspace_copy<<<blockDim, nullptr, stream>>>(workspace, out,
                                                             ffts_addr);
}
