#pragma once

#include "inverse.hpp"
#include "recurrence_pipeline.hpp"

// Public scratch allocations retain their original sizes. Paired operands use
// separate existing half buffers; the recurrent FP32 state stays in vector UB.
template <int R, int K, typename T, typename Full>
AICORE inline void mega_vector_load(Full& full, __gm__ T* pointer, int rows,
                                    int cols, int stride) {
#if defined(__DAV_VEC__)
  using G = GlobalTensor<T, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                         pto::Stride<1, 1, 1, DYNAMIC, 1>>;
  pipe_barrier(PIPE_ALL);
  if (rows == R && cols == K) {
    G gm(pointer, {1, 1, 1, R, K}, {1, 1, 1, stride, 1});
    TLOAD(full, gm);
    pipe_barrier(PIPE_ALL);
    return;
  }
  TEXPANDS(full, static_cast<T>(0));
  PipeBarrierVec();
  if (rows > 0) {
    using Part = Tile<TileType::Vec, T, R, K, BLayout::RowMajor, DYNAMIC,
                      DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
    Part part(rows, cols);
    TASSIGN(part, (uint64_t)full.data());
    G gm(pointer, {1, 1, 1, rows, cols}, {1, 1, 1, stride, 1});
    pipe_barrier(PIPE_ALL);
    TLOAD(part, gm);
    pipe_barrier(PIPE_ALL);
    Tile<TileType::Vec, T, R, K, BLayout::RowMajor, R, K, SLayout::NoneBox, 512,
         PadValue::Zero>
        padded;
    TASSIGN(padded, (uint64_t)full.data());
    TFILLPAD_INPLACE(padded, part);
    PipeBarrierVec();
  }
#endif
}

template <int R, int K, typename F, typename Half>
AICORE inline void mega_split_pair(F& value, Half& high, Half& low) {
#if defined(__DAV_VEC__)
  constexpr int Bytes = R * K * sizeof(float), Base = 96 * 1024;
  Tile<TileType::Vec, float, R, K> rounded;
  TASSIGN(rounded, Base);
  TASSIGN(high, Base + Bytes);
  TASSIGN(low, Base + Bytes + Bytes / 2);
  TCVT(high, value, RoundMode::CAST_RINT);
  PipeBarrierVec();
  TCVT(rounded, high, RoundMode::CAST_NONE);
  PipeBarrierVec();
  TSUB(rounded, value, rounded);
  PipeBarrierVec();
  TCVT(low, rounded, RoundMode::CAST_RINT);
  pipe_barrier(PIPE_ALL);
#endif
}

template <int R, int K, typename F>
AICORE inline void mega_store_pair(F& value, __gm__ half* high_gm,
                                   __gm__ half* low_gm, int rows = R,
                                   int stride = K) {
#if defined(__DAV_VEC__)
  Tile<TileType::Vec, half, R, K> high, low;
  mega_split_pair<R, K>(value, high, low);
  if (rows > 0) {
    using Part = Tile<TileType::Vec, half, R, K, BLayout::RowMajor, DYNAMIC, K>;
    Part ht(rows), lt(rows);
    TASSIGN(ht, (uint64_t)high.data());
    TASSIGN(lt, (uint64_t)low.data());
    using G = GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, K>,
                           pto::Stride<1, 1, 1, DYNAMIC, 1>>;
    G hg(high_gm, {1, 1, 1, rows, K}, {1, 1, 1, stride, 1}),
        lg(low_gm, {1, 1, 1, rows, K}, {1, 1, 1, stride, 1});
    TSTORE(hg, ht);
    TSTORE(lg, lt);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

template <int R, int K>
AICORE inline void mega_load_pair(MegaPair<R, K>& pair, __gm__ half* high,
                                  __gm__ half* low, int rows = R,
                                  int stride = K) {
#if defined(__DAV_CUBE__)
  using Part = Tile<TileType::Mat, half, R, K, BLayout::ColMajor, DYNAMIC, K,
                    SLayout::RowMajor, 512, PadValue::Zero>;
  Part ht(rows), lt(rows);
  TASSIGN(ht, (uint64_t)pair.high.data());
  TASSIGN(lt, (uint64_t)pair.low.data());
  using G = GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, K>,
                         pto::Stride<1, 1, 1, DYNAMIC, 1>>;
  G hg(high, {1, 1, 1, rows, K}, {1, 1, 1, stride, 1}),
      lg(low, {1, 1, 1, rows, K}, {1, 1, 1, stride, 1});
  TLOAD(ht, hg);
  TLOAD(lt, lg);
  pipe_barrier(PIPE_ALL);
  if (rows != R) {
    TFILLPAD(ht, ht);
    TFILLPAD(lt, lt);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

// FP32 dot products are split by columns across two existing scratch buffers,
// not rounded to half. A 64x128 stripe can reconstruct them without allocation.
template <int R, int C, typename F>
AICORE inline void mega_load_kk(F& value, __gm__ float* first,
                                __gm__ float* second, int rows, int stride) {
#if defined(__DAV_VEC__)
  using Part = Tile<TileType::Vec, float, R, C, BLayout::RowMajor, DYNAMIC,
                    DYNAMIC, SLayout::NoneBox, 512, PadValue::Zero>;
  TEXPANDS(value, 0.0f);
  PipeBarrierVec();
  if (rows > 0) {
    Part a(rows, C / 2), b(rows, C / 2);
    TASSIGN(a, (uint64_t)value.data());
    TASSIGN(b, (uint64_t)value.data() + C / 2 * sizeof(float));
    using G = GlobalTensor<float, Shape<1, 1, 1, DYNAMIC, C / 2>,
                           pto::Stride<1, 1, 1, DYNAMIC, 1>>;
    G ag(first, {1, 1, 1, rows, C / 2}, {1, 1, 1, stride, 1}),
        bg(second, {1, 1, 1, rows, C / 2}, {1, 1, 1, stride, 1});
    pipe_barrier(PIPE_ALL);
    TLOAD(a, ag);
    TLOAD(b, bg);
    pipe_barrier(PIPE_ALL);
    Part complete(rows, C);
    TASSIGN(complete, (uint64_t)value.data());
    Tile<TileType::Vec, float, R, C, BLayout::RowMajor, R, C, SLayout::NoneBox,
         512, PadValue::Zero>
        padded;
    TASSIGN(padded, (uint64_t)value.data());
    TFILLPAD_INPLACE(padded, complete);
    PipeBarrierVec();
  }
#endif
}

// Keep [32,96) KiB untouched: output base and recurrent state live there.
// Weight the next chunk before waiting for current projections. Keep its
// pending coefficients at [96,128) KiB across RHS preparation, then finish
// and export its inverse only after the cube has consumed the current one.
template <int H, int C>
AICORE inline void mega_prepare_weights(
    __gm__ float* g, __gm__ float* glow, __gm__ half* beta, __gm__ float* kk,
    __gm__ float* kk_upper, __gm__ float* qk, __gm__ float* lower,
    __gm__ float* full, __gm__ half* attention_high, __gm__ half* attention_low,
    int64_t pos, int valid, int head, int kh, int64_t tokens) {
#if defined(__DAV_VEC__)
  constexpr int R = C / 2, Gate = 160 * 1024;
  int stripe = get_subblockid() * R, count = max(0, min(R, valid - stripe)),
      key_head = head / (H / kh);
  using M = Tile<TileType::Vec, float, R, C>;
  using G = Tile<TileType::Vec, float, 1, C>;
  using Col = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1>;
  G gates, lows, betas;
  Tile<TileType::Vec, half, 1, C> beta_h;
  TASSIGN(gates, Gate);
  TASSIGN(lows, Gate + 512);
  TASSIGN(betas, Gate + 1024);
  TASSIGN(beta_h, Gate + 1536);
  mega_vector_load<1, C>(gates, g + head * tokens + pos, 1, valid, C);
  mega_vector_load<1, C>(lows, glow + head * tokens + pos, 1, valid, C);
  mega_vector_load<1, C>(beta_h, beta + head * tokens + pos, 1, valid, C);
  TCVT(betas, beta_h, RoundMode::CAST_NONE);
  PipeBarrierVec();
  M a, raw, mask;
  TASSIGN(a, 96 * 1024);
  TASSIGN(raw, 0);
  TASSIGN(mask, 128 * 1024);
  Col gcol, lcol, bcol;
  TASSIGN(gcol, Gate + stripe * sizeof(float));
  TASSIGN(lcol, Gate + 512 + stripe * sizeof(float));
  TASSIGN(bcol, Gate + 1024 + stripe * sizeof(float));
  // One decay matrix serves both QK and KK. Native FP32 attention stores
  // leave this tile intact, unlike the former paired-operand conversion.
  TCOLEXPAND(raw, gates);
  PipeBarrierVec();
  TMULS(raw, raw, -1.0f);
  PipeBarrierVec();
  TROWEXPANDADD(raw, raw, gcol);
  PipeBarrierVec();
  TCOLEXPAND(a, lows);
  PipeBarrierVec();
  TMULS(a, a, -1.0f);
  PipeBarrierVec();
  TROWEXPANDADD(a, a, lcol);
  PipeBarrierVec();
  TADD(raw, raw, a);
  PipeBarrierVec();
  // Invalid rows must be masked before exp: a padded zero prefix minus
  // a large negative real prefix would otherwise overflow, then 0*inf
  // would poison the solve and even valid output rows.
  mega_vector_load<R, C>(a, full + stripe * C, count, C, C);
  TMUL(raw, raw, a);
  PipeBarrierVec();
  TEXP(raw, raw);
  PipeBarrierVec();
  TMUL(raw, raw, a);
  pipe_barrier(PIPE_ALL);
  mega_vector_load<R, C>(a, qk + ((pos + stripe) * kh + key_head) * C, count, C,
                         kh * C);
  TMUL(a, a, raw);
  pipe_barrier(PIPE_ALL);
  if (count > 0) {
    auto storage = reinterpret_cast<__gm__ float*>(
        head < H / 2 ? attention_high : attention_low);
    GlobalTensor<float, Shape<1, 1, 1, DYNAMIC, C>, pto::Stride<1, 1, 1, C, 1>>
        gm(storage + ((head % (H / 2)) * tokens + pos + stripe) * C,
           {1, 1, 1, count, C});
    Tile<TileType::Vec, float, R, C, BLayout::RowMajor, DYNAMIC, C> rows(count);
    TASSIGN(rows, 96 * 1024);
    TSTORE(gm, rows);
    pipe_barrier(PIPE_ALL);
  }
  mega_load_kk<R, C>(a, kk + ((pos + stripe) * kh + key_head) * (C / 2),
                     kk_upper + ((pos + stripe) * kh + key_head) * (C / 2),
                     count, kh * C / 2);
  TMUL(a, a, raw);
  PipeBarrierVec();
  mega_vector_load<R, C>(mask, lower + stripe * C, R, C, C);
  TMUL(a, a, mask);
  PipeBarrierVec();
  TROWEXPANDMUL(a, a, bcol);
  PipeBarrierVec();
  TMULS(a, a, -1.0f);
  PipeBarrierVec();
#endif
}

template <int H, int C>
AICORE inline void mega_finish_inverse(
    __gm__ float* g, __gm__ float* glow, __gm__ float* lower,
    __gm__ float* full, __gm__ half* weighted_first, int64_t pos, int valid,
    int head, int kh, int64_t tokens, __gm__ half* first, __gm__ half* second,
    __gm__ half* original_k, __gm__ half* weighted_second) {
#if defined(__DAV_VEC__)
  constexpr int R = C / 2, Gate = 160 * 1024;
  int stripe = get_subblockid() * R, count = max(0, min(R, valid - stripe)),
      key_head = head / (H / kh);
  using M = Tile<TileType::Vec, float, R, C>;
  using G = Tile<TileType::Vec, float, 1, C>;
  using Col = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1>;
  M a;
  TASSIGN(a, 96 * 1024);
  G gates, lows;
  TASSIGN(gates, Gate);
  TASSIGN(lows, Gate + 512);
  mega_vector_load<1, C>(gates, g + head * tokens + pos, 1, valid, C);
  mega_vector_load<1, C>(lows, glow + head * tokens + pos, 1, valid, C);
  M raw, mask;
  TASSIGN(raw, 0);
  TASSIGN(mask, 128 * 1024);
  mega_vector_load<R, C>(raw, full + stripe * C, R, C, C);
  mega_vector_load<R, C>(mask, lower + stripe * C, R, C, C);
  TSUB(raw, raw, mask);
  PipeBarrierVec();
  mega_diagonal_inverses(a, raw, stripe);
  pipe_barrier(PIPE_ALL);
  GlobalTensor<float, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, C, 1>> native(
      reinterpret_cast<__gm__ float*>(stripe == 0 ? first : second));
  TSTORE(native, a);
  pipe_barrier(PIPE_ALL);
  // Move residual weighting into this independent preparation step:
  // Gamma_i = exp(G_last-G_i) * K_i. The recurrence consumes Gamma^T*R.
  G decay, delta;
  TASSIGN(decay, 164 * 1024);
  TASSIGN(delta, 165 * 1024);
  float last = gates.GetValue(valid - 1), last_low = lows.GetValue(valid - 1);
  TMULS(decay, gates, -1.0f);
  PipeBarrierVec();
  TADDS(decay, decay, last);
  PipeBarrierVec();
  TMULS(delta, lows, -1.0f);
  PipeBarrierVec();
  TADDS(delta, delta, last_low);
  PipeBarrierVec();
  TADD(decay, decay, delta);
  PipeBarrierVec();
  TEXP(decay, decay);
  PipeBarrierVec();
  Tile<TileType::Vec, half, R, C> key;
  TASSIGN(key, 128 * 1024);
  mega_vector_load<R, C>(key, original_k + ((pos + stripe) * kh + key_head) * C,
                         count, C, kh * C);
  TCVT(a, key, RoundMode::CAST_NONE);
  PipeBarrierVec();
  Col weight;
  TASSIGN(weight, 164 * 1024 + stripe * sizeof(float));
  TROWEXPANDMUL(a, a, weight);
  pipe_barrier(PIPE_ALL);
  if (count > 0) {
    auto storage = reinterpret_cast<__gm__ float*>(
        head < H / 2 ? weighted_first : weighted_second);
    GlobalTensor<float, Shape<1, 1, 1, DYNAMIC, C>, pto::Stride<1, 1, 1, C, 1>>
        gm(storage + ((head % (H / 2)) * tokens + pos + stripe) * C,
           {1, 1, 1, count, C});
    Tile<TileType::Vec, float, R, C, BLayout::RowMajor, DYNAMIC, C> rows(count);
    TASSIGN(rows, 96 * 1024);
    TSTORE(gm, rows);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

template <int H, int C>
AICORE inline void mega_prepare_inverse(
    __gm__ float* g, __gm__ float* glow, __gm__ half* beta, __gm__ float* kk,
    __gm__ float* kk_upper, __gm__ float* qk, __gm__ float* lower,
    __gm__ float* full, __gm__ half* weighted_first,
    __gm__ half* attention_high, __gm__ half* attention_low, int64_t pos,
    int valid, int head, int kh, int64_t tokens, __gm__ half* first,
    __gm__ half* second, __gm__ half* original_k,
    __gm__ half* weighted_second) {
  mega_prepare_weights<H, C>(g, glow, beta, kk, kk_upper, qk, lower, full,
                             attention_high, attention_low, pos, valid, head,
                             kh, tokens);
  mega_finish_inverse<H, C>(g, glow, lower, full, weighted_first, pos, valid,
                            head, kh, tokens, first, second, original_k,
                            weighted_second);
}

// State is resident at [64,96) KiB in each vector subblock for the entire head.
template <int D>
AICORE inline void mega_initialize_state(__gm__ half* initial,
                                         __gm__ half* final_state,
                                         __gm__ float* exchange,
                                         bool has_initial, bool empty) {
#if defined(__DAV_VEC__)
  constexpr int R = D / 2;
  Tile<TileType::Vec, half, R, D> source;
  Tile<TileType::Vec, float, R, D> state;
  TASSIGN(source, 0);
  TASSIGN(state, 64 * 1024);
  if (has_initial)
    mega_vector_load<R, D>(source, initial + get_subblockid() * R * D, R, D, D);
  else {
    TEXPANDS(source, static_cast<half>(0));
    PipeBarrierVec();
  }
  if (empty) {
    GlobalTensor<half, Shape<1, 1, 1, R, D>, pto::Stride<1, 1, 1, D, 1>> gm(
        final_state + get_subblockid() * R * D);
    TSTORE(gm, source);
    pipe_barrier(PIPE_ALL);
  } else {
    TCVT(state, source, RoundMode::CAST_NONE);
    pipe_barrier(PIPE_ALL);
    auto pair =
        reinterpret_cast<__gm__ half*>(exchange) + get_subblockid() * R * D * 2;
    mega_store_pair<R, D>(state, pair, pair + D, R, 2 * D);
  }
#endif
}

// Form B = beta * (V - exp(G) * K*S) before applying the inverse.
// Base output exp(G)*Q*S remains in UB while the cube solves for residuals.
template <int H, int C, int D>
AICORE inline float mega_prepare_rhs(__gm__ half* v, __gm__ float* g,
                                     __gm__ float* glow, __gm__ half* beta,
                                     __gm__ float* projection,
                                     __gm__ float* output_base, int64_t pos,
                                     int valid, int head, int64_t tokens) {
#if defined(__DAV_VEC__)
  constexpr int R = C / 2, Gate = 160 * 1024;
  Tile<TileType::Vec, float, 1, C> gates, lows, exponentials, betas, correction;
  Tile<TileType::Vec, half, 1, C> beta_h;
  TASSIGN(gates, Gate);
  TASSIGN(lows, Gate + 512);
  TASSIGN(exponentials, Gate + 1024);
  TASSIGN(betas, Gate + 1536);
  TASSIGN(beta_h, Gate + 2048);
  TASSIGN(correction, Gate + 2560);
  mega_vector_load<1, C>(gates, g + head * tokens + pos, 1, valid, C);
  mega_vector_load<1, C>(lows, glow + head * tokens + pos, 1, valid, C);
  mega_vector_load<1, C>(beta_h, beta + head * tokens + pos, 1, valid, C);
  TEXP(exponentials, gates);
  TCVT(betas, beta_h, RoundMode::CAST_NONE);
  PipeBarrierVec();
  TEXP(correction, lows);
  PipeBarrierVec();
  TMUL(exponentials, exponentials, correction);
  pipe_barrier(PIPE_ALL);
  float alpha = exponentials.GetValue(valid - 1);
  int stripe = get_subblockid() * R, count = max(0, min(R, valid - stripe));
  Tile<TileType::Vec, float, R, D> value, values, base;
  Tile<TileType::Vec, half, R, D> input;
  Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1> expcol, bcol;
  // [96,128) KiB holds the pending next-chunk coefficients. Convert V
  // before loading projections so both can reuse the first UB region.
  TASSIGN(value, 0);
  TASSIGN(base, 32 * 1024);
  TASSIGN(values, 128 * 1024);
  TASSIGN(input, 0);
  TASSIGN(expcol, Gate + 1024 + stripe * sizeof(float));
  TASSIGN(bcol, Gate + 1536 + stripe * sizeof(float));
  mega_vector_load<R, D>(input, v + ((pos + stripe) * H + head) * D, count, D,
                         H * D);
  TCVT(values, input, RoundMode::CAST_NONE);
  pipe_barrier(PIPE_ALL);
  mega_vector_load<R, D>(value, projection + stripe * D, R, D, D);
  TROWEXPANDMUL(value, value, expcol);
  pipe_barrier(PIPE_ALL);
  TSUB(value, values, value);
  PipeBarrierVec();
  TROWEXPANDMUL(value, value, bcol);
  pipe_barrier(PIPE_ALL);
  GlobalTensor<float, Shape<1, 1, 1, R, D>, pto::Stride<1, 1, 1, D, 1>> saved(
      projection + C * D + stripe * D);
  TSTORE(saved, value);
  pipe_barrier(PIPE_ALL);
  mega_vector_load<R, D>(base, output_base + stripe * D, R, D, D);
  TROWEXPANDMUL(base, base, expcol);
  pipe_barrier(PIPE_ALL);
  return alpha;
#else
  return 0.0f;
#endif
}

template <int H, int C, int D>
AICORE inline void mega_finish_output(__gm__ float* local, __gm__ half* output,
                                      int valid) {
#if defined(__DAV_VEC__)
  constexpr int R = C / 2;
  int stripe = get_subblockid() * R, count = max(0, min(R, valid - stripe));
  Tile<TileType::Vec, float, R, D> value, base;
  Tile<TileType::Vec, half, R, D> result;
  TASSIGN(value, 0);
  TASSIGN(base, 32 * 1024);
  TASSIGN(result, 96 * 1024);
  mega_vector_load<R, D>(value, local + stripe * D, R, D, D);
  TADD(value, value, base);
  PipeBarrierVec();
  TCVT(result, value, RoundMode::CAST_RINT);
  pipe_barrier(PIPE_ALL);
  if (count > 0) {
    Tile<TileType::Vec, half, R, D, BLayout::RowMajor, DYNAMIC, D> tail(count);
    TASSIGN(tail, 96 * 1024);
    GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, D>,
                 pto::Stride<1, 1, 1, H * D, 1>>
        gm(output + stripe * H * D, {1, 1, 1, count, D});
    TSTORE(gm, tail);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

template <int D>
AICORE inline void mega_update_state(__gm__ float* exchange, float alpha,
                                     __gm__ half* final_state, bool last) {
#if defined(__DAV_VEC__)
  constexpr int R = D / 2;
  Tile<TileType::Vec, float, R, D> state, update;
  Tile<TileType::Vec, half, R, D> result;
  TASSIGN(state, 64 * 1024);
  TASSIGN(update, 0);
  TASSIGN(result, 32 * 1024);
  int offset = get_subblockid() * R * D;
  GlobalTensor<float, Shape<1, 1, 1, R, D>, pto::Stride<1, 1, 1, D, 1>> gm(
      exchange + offset);
  TLOAD(update, gm);
  pipe_barrier(PIPE_ALL);
  TMULS(state, state, alpha);
  PipeBarrierVec();
  TADD(state, state, update);
  pipe_barrier(PIPE_ALL);
  // Interleaved high/low rows occupy exactly the FP32 update's footprint.
  // Each subblock overwrites only rows it has already loaded.
  auto pair = reinterpret_cast<__gm__ half*>(exchange) + 2 * offset;
  mega_store_pair<R, D>(state, pair, pair + D, R, 2 * D);
  if (last) {
    TCVT(result, state, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_ALL);
    GlobalTensor<half, Shape<1, 1, 1, R, D>, pto::Stride<1, 1, 1, D, 1>> fg(
        final_state + offset);
    TSTORE(fg, result);
    pipe_barrier(PIPE_ALL);
  }
#endif
}

template <int H, int C, int D>
AICORE inline void mega_fused_pipeline(
    __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ float* g,
    __gm__ float* glow, __gm__ half* beta, __gm__ half* weighted_first,
    __gm__ half* weighted_second, __gm__ half* att_high, __gm__ half* att_low,
    __gm__ half* uh_ws, __gm__ half* ul_ws, __gm__ half* k_ws,
    __gm__ half* state_ws, __gm__ half* solve_ws, __gm__ half* output,
    __gm__ half* final_state, __gm__ half* initial, bool has_initial,
    __gm__ int32_t* cu, int64_t sequences, int kh, int64_t tokens,
    __gm__ float* kk, __gm__ float* kk_upper, __gm__ float* qk,
    __gm__ float* lower, __gm__ float* full) {
#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, uint32_t, 1, 16 * (C / 2)> indices;
  TASSIGN(indices, 176 * 1024);
  // A2/A3 vgather takes byte offsets; A5 TGATHER takes element indices.
#if __CCE_AICORE__ == 220
  constexpr int gather_scale = sizeof(float);
#else
  constexpr int gather_scale = 1;
#endif
  for (int j = 0; j < 16; ++j)
    for (int col = 0; col < C / 2; ++col)
      indices.SetValue(j * (C / 2) + col,
                       (((col / 16) * 16 + j) * C + get_subblockid() * (C / 2) +
                        (col / 16) * 16) *
                           gather_scale);
  pipe_barrier(PIPE_ALL);
#endif
#if defined(__DAV_CUBE__)
  MegaPair<D, D> state;
  MegaFloat<C, C> attention, inverse;
  MegaFloat<C, D> native_rhs, native_residual, native_gamma;
  state.assign(0);
  TASSIGN(inverse, 64 * 1024);
  TASSIGN(native_rhs, 192 * 1024);
  TASSIGN(native_gamma, 256 * 1024);
  TASSIGN(attention, 384 * 1024);
  TASSIGN(native_residual, 448 * 1024);
  typename MegaPair<2 * C, D>::Mat inputs;
  TASSIGN(inputs, 256 * 1024);
  TileAcc<float, 2 * C, D> projections;
  TASSIGN(projections, 0);
  TileAcc<float, 2 * C, D, C, D> first_product, second_product;
  TASSIGN(first_product, 0);
  TASSIGN(second_product, C * 16 * sizeof(float));
  TileAcc<float, C, D> acc;
  TASSIGN(acc, 0);
  using Tail = Tile<TileType::Mat, half, 2 * C, D, BLayout::ColMajor, DYNAMIC,
                    D, SLayout::RowMajor, 512, PadValue::Zero>;
#endif
  int block = get_block_idx();
  auto exchange = reinterpret_cast<__gm__ float*>(state_ws + block * 4 * D * D);
  auto local_output = reinterpret_cast<__gm__ float*>(k_ws + block * 2 * C * D);
  auto first = uh_ws + block * C * C, second = ul_ws + block * C * C;
  auto solve_scratch =
      reinterpret_cast<__gm__ float*>(solve_ws + block * C * C);
  for (int64_t work = block; work < sequences * H; work += get_block_num()) {
    int seq = work / H, head = work % H;
    int64_t start = cu[seq], end = cu[seq + 1];
#if defined(__DAV_VEC__)
    if (start < end)
      mega_prepare_inverse<H, C>(
          g, glow, beta, kk, kk_upper, qk, lower, full, weighted_first,
          att_high, att_low, start, min(static_cast<int64_t>(C), end - start),
          head, kh, tokens, first, second, k, weighted_second);
    mega_initialize_state<D>(has_initial ? initial + work * D * D : initial,
                             final_state + work * D * D, exchange, has_initial,
                             start == end);
#endif
    if (start == end) continue;
#if defined(__DAV_VEC__)
    mega_signal<PIPE_MTE3>(1);
#endif
#if defined(__DAV_CUBE__)
    mega_wait(1);
    auto sh = reinterpret_cast<__gm__ half*>(exchange);
    mega_load_pair<D, D>(state, sh, sh + D, D, 2 * D);
    mega_signal<PIPE_MTE2>(7);
#endif
    for (int64_t pos = start; pos < end; pos += C) {
      int valid = min(static_cast<int64_t>(C), end - pos);
#if defined(__DAV_VEC__)
      mega_wait(7);
      if (pos + C < end)
        mega_prepare_weights<H, C>(g, glow, beta, kk, kk_upper, qk, lower, full,
                                   att_high, att_low, pos + C,
                                   min(static_cast<int64_t>(C), end - pos - C),
                                   head, kh, tokens);
      mega_wait(5);
      float alpha = mega_prepare_rhs<H, C, D>(
          v, g, glow, beta, exchange, local_output, pos, valid, head, tokens);
      mega_signal<PIPE_MTE3>(6);
      if (pos + C < end)
        mega_finish_inverse<H, C>(g, glow, lower, full, weighted_first, pos + C,
                                  min(static_cast<int64_t>(C), end - pos - C),
                                  head, kh, tokens, first, second, k,
                                  weighted_second);
      mega_wait(9);
      mega_finish_output<H, C, D>(local_output, output + (pos * H + head) * D,
                                  valid);
      mega_update_state<D>(exchange, alpha, final_state + work * D * D,
                           pos + C >= end);
      mega_signal<PIPE_MTE3>(6);
#endif
#if defined(__DAV_CUBE__)
      Tail ktail(valid), qtail(valid);
      TASSIGN(ktail, 256 * 1024);
      TASSIGN(qtail, 256 * 1024 + C * 16 * sizeof(half));
      using InputG = GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, D>,
                                  pto::Stride<1, 1, 1, DYNAMIC, 1>>;
      int64_t input_offset = (pos * kh + head / (H / kh)) * D;
      InputG kgm(k + input_offset, {1, 1, 1, valid, D}, {1, 1, 1, kh * D, 1}),
          qgm(q + input_offset, {1, 1, 1, valid, D}, {1, 1, 1, kh * D, 1});
      TLOAD(ktail, kgm);
      pipe_barrier(PIPE_ALL);
      // Pad the whole stacked tile before Q is loaded into its lower
      // half. This clears both tails without touching valid Q rows.
      if (valid != C) {
        TFILLPAD(ktail, ktail);
        pipe_barrier(PIPE_ALL);
      }
      TLOAD(qtail, qgm);
      pipe_barrier(PIPE_ALL);
      mega_input_pair_product<2 * C, D, D>(projections, inputs, state);
      GlobalTensor<float, Shape<1, 1, 1, C, D>, pto::Stride<1, 1, 1, D, 1>> ex(
          exchange),
          og(local_output);
      TSTORE(ex, first_product);
      TSTORE(og, second_product);
      pipe_barrier(PIPE_ALL);
      Tile<TileType::Mat, float, C, C, BLayout::ColMajor, C / 2, C,
           SLayout::RowMajor, 512>
          first_half, second_half;
      TASSIGN(first_half, 64 * 1024);
      TASSIGN(second_half, 64 * 1024 + (C / 2) * 8 * sizeof(float));
      GlobalTensor<float, Shape<1, 1, 1, C / 2, C>, pto::Stride<1, 1, 1, C, 1>>
          fg(reinterpret_cast<__gm__ float*>(first)),
          sg(reinterpret_cast<__gm__ float*>(second));
      TLOAD(first_half, fg);
      TLOAD(second_half, sg);
      pipe_barrier(PIPE_ALL);
      mega_signal<PIPE_FIX>(5);
      mega_float_inverse<C>(inverse, solve_scratch);
      {
        auto storage =
            reinterpret_cast<__gm__ float*>(head < H / 2 ? att_high : att_low);
        GlobalTensor<float, Shape<1, 1, 1, DYNAMIC, C>,
                     pto::Stride<1, 1, 1, C, 1>>
            ag(storage + ((head % (H / 2)) * tokens + pos) * C,
               {1, 1, 1, valid, C});
        Tile<TileType::Mat, float, C, C, BLayout::ColMajor, DYNAMIC, C,
             SLayout::RowMajor, 512, PadValue::Zero>
            part(valid);
        TASSIGN(part, 384 * 1024);
        TLOAD(part, ag);
        pipe_barrier(PIPE_ALL);
        if (valid != C) {
          TFILLPAD(part, part);
          pipe_barrier(PIPE_ALL);
        }
      }
      mega_wait(6);
      GlobalTensor<float, Shape<1, 1, 1, C, D>, pto::Stride<1, 1, 1, D, 1>> bg(
          exchange + C * D);
      TLOAD(native_rhs, bg);
      pipe_barrier(PIPE_ALL);
      mega_factored_solve<C, D>(inverse, native_rhs, exchange);
      TLOAD(native_residual, ex);
      pipe_barrier(PIPE_ALL);
      {
        auto storage = reinterpret_cast<__gm__ float*>(
            head < H / 2 ? weighted_first : weighted_second);
        GlobalTensor<float, Shape<1, 1, 1, DYNAMIC, C>,
                     pto::Stride<1, 1, 1, C, 1>>
            gg(storage + ((head % (H / 2)) * tokens + pos) * C,
               {1, 1, 1, valid, C});
        Tile<TileType::Mat, float, C, C, BLayout::ColMajor, DYNAMIC, C,
             SLayout::RowMajor, 512, PadValue::Zero>
            part(valid);
        TASSIGN(part, 256 * 1024);
        TLOAD(part, gg);
        pipe_barrier(PIPE_ALL);
        if (valid != C) {
          TFILLPAD(part, part);
          pipe_barrier(PIPE_ALL);
        }
      }
      using GT = Tile<TileType::Mat, float, D, C, BLayout::RowMajor, D, C,
                      SLayout::ColMajor, 512>;
      GT transpose;
      TRESHAPE(transpose, native_gamma);
      mega_combined_residual_products<C, D>(projections, attention, transpose,
                                            native_residual);
      TSTORE(og, first_product);
      TSTORE(ex, second_product);
      pipe_barrier(PIPE_ALL);
      mega_signal<PIPE_FIX>(9);
      mega_wait(6);
      mega_load_pair<D, D>(state, sh, sh + D, D, 2 * D);
      mega_signal<PIPE_MTE2>(7);
#endif
    }
#if defined(__DAV_VEC__)
    mega_wait(7);
#endif
  }
  pipe_barrier(PIPE_ALL);
}
