#pragma once

// Two FP16 components retain a leading value and its rounding correction.
// Original FP16 inputs multiply both components, accumulating in FP32.
// Recurrent state has an authoritative FP32 copy; inverse and attention
// products use native FP32 operands.
template <int R, int K>
struct MegaPair;
#ifdef __CCE_AICORE__
template <int R, int K>
struct MegaPair {
  using Mat = Tile<TileType::Mat, half, R, K, BLayout::ColMajor, R, K,
                   SLayout::RowMajor, 512>;
  Mat high, low;
  AICORE inline void assign(uint32_t address) {
    TASSIGN(high, address);
    TASSIGN(low, address + R * K * sizeof(half));
  }
};
#endif

// The left operand is an original FP16 input, so its rounding residual is zero.
template <int R, int K, int N, typename Acc, typename A, typename B>
AICORE inline void mega_input_pair_product(Acc& result, A& a, B& b) {
#if defined(__DAV_CUBE__)
  TileLeft<half, R, K> left;
  TileRight<half, K, N> high, low;
  TASSIGN(left, 0);
  TASSIGN(high, 0);
  TASSIGN(low, K * N * sizeof(half));
  TEXTRACT(left, a, 0, 0);
  TMOV(high, b.high);
  TMOV(low, b.low);
  pipe_barrier(PIPE_ALL);
  TMATMUL(result, left, low);
  pipe_barrier(PIPE_M);
  TMATMUL_ACC(result, result, left, high);
  pipe_barrier(PIPE_ALL);
#endif
}

#ifdef __CCE_AICORE__
template <int R, int C>
using MegaFloat = Tile<TileType::Mat, float, R, C, BLayout::ColMajor, R, C,
                       SLayout::RowMajor, 512>;
#endif

// Join all independent diagonal block pairs at one level together. Products
// retain separate accumulators. Export/reload phases are synchronized once
// per level, with independent DMA and matrix operations queued together.
template <int B, int C, bool FirstOnly = false, typename Mat>
AICORE inline void mega_float_join_level(Mat& inverse, __gm__ float* scratch) {
#if defined(__DAV_CUBE__)
  constexpr int Groups = C / (2 * B), Rows = Groups * B;
  MegaFloat<Rows, B> temporary;
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
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(acc, left, right);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    GlobalTensor<float, Shape<1, 1, 1, B, B>, pto::Stride<1, 1, 1, B, 1>> out(
        scratch + group * B * B);
    TSTORE(out, acc);
  }
  set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
  if constexpr (!FirstOnly) {
    TLOAD(temporary, gm);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    for (int group = 0; group < Groups; ++group) {
      int offset = group * 2 * B;
      TASSIGN(left, group * B * B * sizeof(float));
      TASSIGN(right, group * B * B * sizeof(float));
      TASSIGN(acc, group * B * B * sizeof(float));
      TEXTRACT(left, temporary, group * B, 0);
      TEXTRACT(right, inverse, offset, offset);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      TMATMUL(acc, left, right);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      GlobalTensor<float, Shape<1, 1, 1, B, B>, pto::Stride<1, 1, 1, B, 1>> out(
          scratch + group * B * B);
      TSTORE(out, acc);
    }
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
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

template <int C, typename Mat>
AICORE inline void mega_float_inverse(Mat& inverse, __gm__ float* scratch) {
#if defined(__DAV_CUBE__)
  mega_float_join_level<16, C>(inverse, scratch);
  mega_float_join_level<32, C>(inverse, scratch);
  mega_float_join_level<64, C, true>(inverse, scratch);
#endif
}

// Assemble two independent left operands in L0A, sharing each R panel in L0B.
// K is split in half so the stacked FP32 operand fits the 64 KiB L0A capacity.
template <int C, int D, typename Acc, typename Attention, typename Gamma,
          typename Residual>
AICORE inline void mega_combined_residual_products(Acc& result,
                                                   Attention& attention,
                                                   Gamma& gamma,
                                                   Residual& residual) {
#if defined(__DAV_CUBE__)
  static_assert(C == D);
  TileLeft<float, 2 * C, C / 2> left;
  TileLeft<float, C, C / 2> top, bottom;
  TileRight<float, C / 2, D> right;
  TASSIGN(left, 0);
  TASSIGN(top, 0);
  TASSIGN(bottom, C * (C / 2) * sizeof(float));
  TASSIGN(right, 0);
  for (int part = 0; part < 2; ++part) {
    TEXTRACT(top, attention, 0, part * (C / 2));
    TEXTRACT(bottom, gamma, 0, part * (C / 2));
    TEXTRACT(right, residual, part * (C / 2), 0);
    pipe_barrier(PIPE_ALL);
    if (part == 0)
      TMATMUL(result, left, right);
    else
      TMATMUL_ACC(result, result, left, right);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

template <int C, int D, typename Mat, typename Rhs>
AICORE inline void mega_factored_solve(Mat& inverse, Rhs& rhs,
                                       __gm__ float* exchange) {
#if defined(__DAV_CUBE__)
  constexpr int B = C / 2;
  TileLeft<float, B, B> left0, left1;
  TileRight<float, B, D> right0, right1;
  TileAcc<float, B, D> acc0, acc1;
  TASSIGN(left0, 0);
  TASSIGN(left1, B * B * sizeof(float));
  TASSIGN(right0, 0);
  TASSIGN(right1, B * D * sizeof(float));
  TASSIGN(acc0, 0);
  TASSIGN(acc1, B * D * sizeof(float));
  TEXTRACT(left0, inverse, 0, 0);
  TEXTRACT(left1, inverse, B, B);
  TEXTRACT(right0, rhs, 0, 0);
  TEXTRACT(right1, rhs, B, 0);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(acc0, left0, right0);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  GlobalTensor<float, Shape<1, 1, 1, B, D>, pto::Stride<1, 1, 1, D, 1>> top(
      exchange),
      bottom(exchange + B * D);
  TSTORE(top, acc0);
  TMATMUL(acc1, left1, right1);
  set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
  Tile<TileType::Mat, float, C, D, BLayout::ColMajor, B, D, SLayout::RowMajor,
       512>
      solved_top;
  TASSIGN(solved_top, (uint64_t)rhs.data());
  TLOAD(solved_top, top);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TEXTRACT(left0, inverse, B, 0);
  TEXTRACT(right0, rhs, 0, 0);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL_ACC(acc1, acc1, left0, right0);
  pipe_barrier(PIPE_ALL);
  TSTORE(bottom, acc1);
  pipe_barrier(PIPE_ALL);
#endif
}
