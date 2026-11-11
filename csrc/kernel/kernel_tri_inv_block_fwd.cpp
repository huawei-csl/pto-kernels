// Standalone mega-GDN inverse: vector 16x16 forward substitution followed by
// FP32 cube block joins. Input is strictly lower triangular, with implicit I.
#include "kernel_utils.h"
using namespace pto;
#include "gdn_mega/common.h"
#include "gdn_mega/inverse.h"

extern "C" __global__ AICORE void tri_inv_block_fwd(
    __gm__ half* input, __gm__ half* output, __gm__ float* workspace,
    uint32_t tiles, uint32_t heads) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  constexpr int C = 128, R = 64;
  const uint32_t stride = heads == 0 ? C : heads * C;
#if defined(__DAV_VEC__)
  const int stripe = get_subblockid() * R;
  Tile<TileType::Vec, float, R, C> a, identity, mask;
  Tile<TileType::Vec, half, R, C> packed;
  TASSIGN(a, 96 * 1024);
  TASSIGN(identity, 0);
  TASSIGN(mask, 64 * 1024);
  TASSIGN(packed, 32 * 1024);
  Tile<TileType::Vec, uint32_t, 1, 16 * R> indices;
  TASSIGN(indices, 176 * 1024);
#if __CCE_AICORE__ == 220
  constexpr int scale = sizeof(float);
#else
  constexpr int scale = 1;
#endif
  for (int j = 0; j < 16; ++j)
    for (int col = 0; col < R; ++col)
      indices.SetValue(j * R + col,
          (((col / 16) * 16 + j) * C + stripe + (col / 16) * 16) * scale);
  TEXPANDS(mask, 0.0f);
  pipe_barrier(PIPE_ALL);
  for (int row = 0; row < R; ++row) {
    const int columns = stripe + row;
    if (columns > 0) {
      Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, DYNAMIC>
          prefix(columns);
      TASSIGN(prefix, 64 * 1024 + row * C * sizeof(float));
      TEXPANDS(prefix, -1.0f);
    }
  }
  TEXPANDS(identity, 0.0f);
  pipe_barrier(PIPE_ALL);
  for (int row = 0; row < R; ++row)
    identity.SetValue(row * C + stripe + row, 1.0f);
  pipe_barrier(PIPE_ALL);
#endif
  for (uint32_t tile = get_block_idx(); tile < tiles; tile += get_block_num()) {
    const uint64_t offset = heads == 0 ? uint64_t(tile) * C * C
        : uint64_t(tile / heads) * C * stride + (tile % heads) * C;
    auto prepared = workspace + get_block_idx() * (C * C + R * R);
    auto scratch = prepared + C * C;
#if defined(__DAV_VEC__)
    GlobalTensor<half, Shape<1, 1, 1, R, C>,
                 pto::Stride<1, 1, 1, DYNAMIC, 1>> src(
        input + offset + stripe * stride, {}, {1, 1, 1, stride, 1});
    TLOAD(packed, src);
    pipe_barrier(PIPE_ALL);
    TCVT(a, packed, RoundMode::CAST_NONE);
    kernel_utils::PipeBarrierVec();
    // Negate only the strict lower triangle; the diagonal is implicit I.
    TMUL(a, a, mask);
    pipe_barrier(PIPE_ALL);
    mega_diagonal_inverses(a, identity, stripe);
    pipe_barrier(PIPE_ALL);
    GlobalTensor<float, Shape<1, 1, 1, R, C>,
                 pto::Stride<1, 1, 1, C, 1>> gm(prepared + stripe * C);
    TSTORE(gm, a);
    TCVT(packed, a, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_ALL);
    GlobalTensor<half, Shape<1, 1, 1, R, C>,
                 pto::Stride<1, 1, 1, DYNAMIC, 1>> dst(
        output + offset + stripe * stride, {}, {1, 1, 1, stride, 1});
    TSTORE(dst, packed);
    mega_signal<PIPE_MTE3>(0);
    mega_wait(1);
#endif
#if defined(__DAV_CUBE__)
    mega_wait(0);
    Tile<TileType::Mat, float, C, C, BLayout::ColMajor, C, C,
         SLayout::RowMajor, 512> inverse;
    TASSIGN(inverse, 64 * 1024);
    GlobalTensor<float, Shape<1, 1, 1, C, C>,
                 pto::Stride<1, 1, 1, C, 1>> gm(prepared);
    TLOAD(inverse, gm);
    pipe_barrier(PIPE_ALL);
    mega_float_join_level<16, C>(inverse, scratch, output + offset, stride);
    mega_float_join_level<32, C>(inverse, scratch, output + offset, stride);
    // Unlike the fused recurrence, return the fully materialized inverse.
    mega_float_join_level<64, C>(inverse, scratch, output + offset, stride);
    mega_signal<PIPE_FIX>(1);
#endif
  }
  pipe_barrier(PIPE_ALL);
}

extern "C" void pto_launch_tri_inv_block_fwd_fp16(
    uint32_t blockDim, void* stream, void* input, void* output,
    void* workspace, uint32_t tiles, uint32_t heads) {
  tri_inv_block_fwd<<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ half*)output, (__gm__ float*)workspace,
      tiles, heads);
}
