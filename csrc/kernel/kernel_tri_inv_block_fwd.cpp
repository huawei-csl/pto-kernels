// Standalone mega-GDN inverse: vector 16x16 forward substitution followed by
// FP32 cube block joins. Input is strictly lower triangular, with implicit I.
#include "kernel_utils.h"
using namespace pto;
// Algorithm-specific helpers live here so this kernel can be compiled and
// maintained independently of the fused GDN implementation. kernel_utils.h
// supplies only the common PTO types and architecture synchronization
// utilities.
namespace {

// Four independent 16x16 unit-triangular solves share one vector instruction
// stream. The gather repeats each block's elimination coefficient over that
// block's columns. This is forward substitution, never an inverse power series.
#if defined(__DAV_VEC__)
template <typename D, typename S>
AICORE inline void block_fwd_copy_block16(D& dst, S& src) {
#if __CCE_AICORE__ == 220
  set_vector_mask(0, 65535);
  vadds(dst.data(), src.data(), 0.0f, 16, 1, 1, D::Cols / 8, S::Cols / 8);
  kernel_utils::PipeBarrierVec();
  set_vector_mask(-1, -1);
#else
  TADDS(dst, src, 0.0f);
#endif
}

template <typename Stripe, typename Identity>
AICORE inline void block_fwd_diagonal_inverses(Stripe& a, Identity& identity,
                                               int stripe) {
  constexpr int B = 16, R = 64, Base = 128 * 1024,
                Bytes = B * R * sizeof(float);
  using Batch = Tile<TileType::Vec, float, B, R>;
  Batch inverse, coefficients;
  TASSIGN(inverse, Base);
  TASSIGN(coefficients, Base + 3 * Bytes);
  for (int b = 0; b < R / B; ++b) {
    using Source = Tile<TileType::Vec, float, B, 128, BLayout::RowMajor, B, B>;
    using Dest = Tile<TileType::Vec, float, B, R, BLayout::RowMajor, B, B>;
    Source unit;
    Dest idst;
    TASSIGN(unit, (uint64_t)identity.data() +
                      (b * B * 128 + stripe + b * B) * sizeof(float));

    TASSIGN(idst, Base + b * B * sizeof(float));
    block_fwd_copy_block16(idst, unit);
    kernel_utils::PipeBarrierVec();
  }
  Tile<TileType::Vec, uint32_t, 1, B * R> indices;
  TASSIGN(indices, 176 * 1024);
  for (int i = 0; i + 1 < B; ++i) {
    Tile<TileType::Vec, float, 1, R> solved, source;
    TASSIGN(solved, Base + i * R * sizeof(float));
    TASSIGN(source, (uint64_t)a.data() + i * sizeof(float));
#if __CCE_AICORE__ == 220
    set_mask_count();
    set_vector_mask(0, B * R);
    vgather(reinterpret_cast<__ubuf__ uint32_t*>(coefficients.data()),
            indices.data(), reinterpret_cast<uintptr_t>(source.data()), 8, 1);
    kernel_utils::PipeBarrierVec();
    set_mask_norm();
    set_vector_mask(-1, -1);
    // The solved row is reused by every destination row directly by the
    // vector instruction. No materialized 16-row broadcast is necessary.
    vmla(inverse.data(), coefficients.data(), solved.data(), B, 1, 1, 1, R / 8,
         R / 8, 0);
#else
    // A5 uses element indices and register-based vector instructions.
    Tile<TileType::Vec, uint32_t, B, R> gather_indices;
    TASSIGN(gather_indices, 176 * 1024);
    Batch broadcast;
    TASSIGN(broadcast, Base + Bytes);
    TGATHER(coefficients, source, gather_indices, broadcast);
    TCOLEXPAND(broadcast, solved);
    TMULADDDST(inverse, coefficients, broadcast);
#endif
    kernel_utils::PipeBarrierVec();
  }
  for (int b = 0; b < R / B; ++b) {
    using Source = Tile<TileType::Vec, float, B, R, BLayout::RowMajor, B, B>;
    using Dest = Tile<TileType::Vec, float, B, 128, BLayout::RowMajor, B, B>;
    Source src;
    Dest dst;
    TASSIGN(src, Base + b * B * sizeof(float));
    TASSIGN(dst, (uint64_t)a.data() +
                     (b * B * 128 + stripe + b * B) * sizeof(float));
    block_fwd_copy_block16(dst, src);
    kernel_utils::PipeBarrierVec();
  }
}
#endif

// Join all independent diagonal block pairs at one level together. Products
// retain separate accumulators. Export/reload phases are synchronized once
// per level, with independent DMA and matrix operations queued together.
template <int B, int C, bool FirstOnly = false, typename Mat,
          typename OutputT = half>
AICORE inline void block_fwd_join_level(Mat& inverse, __gm__ float* scratch,
                                        __gm__ OutputT* output = nullptr,
                                        uint32_t output_stride = C) {
#if defined(__DAV_CUBE__)
  constexpr int Groups = C / (2 * B), Rows = Groups * B;
  Tile<TileType::Mat, float, Rows, B, BLayout::ColMajor, Rows, B,
       SLayout::RowMajor, 512>
      temporary;
  TASSIGN(temporary, 128 * 1024);
  TileLeft<float, B, B> left;
  TileRight<float, B, B> right;
  TileAcc<float, B, B> acc;
  GlobalTensor<float, Shape<1, 1, 1, Rows, B>, pto::Stride<1, 1, 1, B, 1>> gm(
      scratch);
  for (int group = 0; group < Groups; ++group) {
    int offset = group * 2 * B;
    TASSIGN(left, group * B * B * sizeof(float));
    TASSIGN(right, group * B * B * sizeof(float));
    TASSIGN(acc, group * B * B * sizeof(float));
    TEXTRACT(left, inverse, offset + B, offset + B);
    TEXTRACT(right, inverse, offset + B, offset);
    kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(EVENT_ID0);
    TMATMUL(acc, left, right);
    kernel_utils::SetWaitFlag<PIPE_M, PIPE_FIX>(EVENT_ID0);
    GlobalTensor<float, Shape<1, 1, 1, B, B>, pto::Stride<1, 1, 1, B, 1>> out(
        scratch + group * B * B);
    TSTORE(out, acc);
  }
  kernel_utils::SetWaitFlag<PIPE_FIX, PIPE_MTE2>(EVENT_ID0);
  if constexpr (!FirstOnly) {
    TLOAD(temporary, gm);
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(EVENT_ID0);
    for (int group = 0; group < Groups; ++group) {
      int offset = group * 2 * B;
      TASSIGN(left, group * B * B * sizeof(float));
      TASSIGN(right, group * B * B * sizeof(float));
      TASSIGN(acc, group * B * B * sizeof(float));
      TEXTRACT(left, temporary, group * B, 0);
      TEXTRACT(right, inverse, offset, offset);
      kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(EVENT_ID0);
      TMATMUL(acc, left, right);
      kernel_utils::SetWaitFlag<PIPE_M, PIPE_FIX>(EVENT_ID0);
      GlobalTensor<float, Shape<1, 1, 1, B, B>, pto::Stride<1, 1, 1, B, 1>> out(
          scratch + group * B * B);
      TSTORE(out, acc);
      if (output != nullptr) {
        GlobalTensor<OutputT, Shape<1, 1, 1, B, B>,
                     pto::Stride<1, 1, 1, DYNAMIC, 1>>
            result(output + (offset + B) * output_stride + offset, {},
                   {1, 1, 1, output_stride, 1});
        TSTORE(result, acc);
      }
    }
    kernel_utils::SetWaitFlag<PIPE_FIX, PIPE_MTE2>(EVENT_ID0);
  }
  for (int group = 0; group < Groups; ++group) {
    int offset = group * 2 * B;
    Tile<TileType::Mat, float, C, C, BLayout::ColMajor, B, B, SLayout::RowMajor,
         512>
        destination;
    TASSIGN(destination, (uint64_t)inverse.data() +
                             (offset * C + (offset + B) * 8) * sizeof(float));
    GlobalTensor<float, Shape<1, 1, 1, B, B>, pto::Stride<1, 1, 1, B, 1>> block(
        scratch + group * B * B);
    TLOAD(destination, block);
  }
  pipe_barrier(PIPE_ALL);
#endif
}

}  // namespace

template <typename OutputT>
__global__ AICORE void tri_inv_block_fwd(__gm__ half* input,
                                         __gm__ OutputT* output,
                                         __gm__ float* workspace,
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
      indices.SetValue(
          j * R + col,
          (((col / 16) * 16 + j) * C + stripe + (col / 16) * 16) * scale);
  TEXPANDS(mask, 0.0f);
  pipe_barrier(PIPE_ALL);
  for (int row = 0; row < R; ++row) {
    const int columns = stripe + row;
    if (columns > 0) {
      Tile<TileType::Vec, float, 1, C, BLayout::RowMajor, 1, DYNAMIC> prefix(
          columns);
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
    const uint64_t offset =
        heads == 0 ? uint64_t(tile) * C * C
                   : uint64_t(tile / heads) * C * stride + (tile % heads) * C;
    auto prepared = workspace + get_block_idx() * (C * C + R * R);
    auto scratch = prepared + C * C;
#if defined(__DAV_VEC__)
    GlobalTensor<half, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, DYNAMIC, 1>>
        src(input + offset + stripe * stride, {}, {1, 1, 1, stride, 1});
    TLOAD(packed, src);
    pipe_barrier(PIPE_ALL);
    TCVT(a, packed, RoundMode::CAST_NONE);
    kernel_utils::PipeBarrierVec();
    // Negate only the strict lower triangle; the diagonal is implicit I.
    TMUL(a, a, mask);
    pipe_barrier(PIPE_ALL);
    block_fwd_diagonal_inverses(a, identity, stripe);
    pipe_barrier(PIPE_ALL);
    GlobalTensor<float, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, C, 1>> gm(
        prepared + stripe * C);
    TSTORE(gm, a);
    GlobalTensor<OutputT, Shape<1, 1, 1, R, C>,
                 pto::Stride<1, 1, 1, DYNAMIC, 1>>
        dst(output + offset + stripe * stride, {}, {1, 1, 1, stride, 1});
    if constexpr (std::is_same_v<OutputT, float>) {
      TSTORE(dst, a);
    } else {
      TCVT(packed, a, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_ALL);
      TSTORE(dst, packed);
    }
#if __CCE_AICORE__ == 220
    kernel_utils::SetCrossFlag<PIPE_MTE3>(0);
    wait_flag_dev(1);
#else
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_MTE3, 0);
    wait_intra_block(PIPE_MTE3, 1);
    pipe_barrier(PIPE_ALL);
#endif
#endif
#if defined(__DAV_CUBE__)
#if __CCE_AICORE__ == 220
    wait_flag_dev(0);
#else
    kernel_utils::WaitBothVecOnA5<PIPE_MTE2>(0);
    pipe_barrier(PIPE_ALL);
#endif
    Tile<TileType::Mat, float, C, C, BLayout::ColMajor, C, C, SLayout::RowMajor,
         512>
        inverse;
    TASSIGN(inverse, 64 * 1024);
    GlobalTensor<float, Shape<1, 1, 1, C, C>, pto::Stride<1, 1, 1, C, 1>> gm(
        prepared);
    TLOAD(inverse, gm);
    pipe_barrier(PIPE_ALL);
    block_fwd_join_level<16, C>(inverse, scratch, output + offset, stride);
    block_fwd_join_level<32, C>(inverse, scratch, output + offset, stride);
    // Unlike the fused recurrence, return the fully materialized inverse.
    block_fwd_join_level<64, C>(inverse, scratch, output + offset, stride);
#if __CCE_AICORE__ == 220
    kernel_utils::SetCrossFlag<PIPE_FIX>(1);
#else
    pipe_barrier(PIPE_ALL);
    kernel_utils::SignalBothVecOnA5<PIPE_FIX>(1);
#endif
#endif
  }
  pipe_barrier(PIPE_ALL);
}

extern "C" void pto_launch_tri_inv_block_fwd_fp16(uint32_t blockDim,
                                                  void* stream, void* input,
                                                  void* output, void* workspace,
                                                  uint32_t tiles,
                                                  uint32_t heads) {
  tri_inv_block_fwd<half>
      <<<blockDim, nullptr, stream>>>((__gm__ half*)input, (__gm__ half*)output,
                                      (__gm__ float*)workspace, tiles, heads);
}

extern "C" void pto_launch_tri_inv_block_fwd_fp32_out(
    uint32_t blockDim, void* stream, void* input, void* output, void* workspace,
    uint32_t tiles, uint32_t heads) {
  tri_inv_block_fwd<float><<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)output, (__gm__ float*)workspace,
      tiles, heads);
}
