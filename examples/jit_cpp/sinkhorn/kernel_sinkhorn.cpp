/**
Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
See LICENSE in the root of the software repository for the full License text.
*/

/**
 * Doubly-stochastic Sinkhorn normalization (fp16 I/O).
 *
 * Input:  (N, K, K) fp16 — batch of K×K matrices.
 * Output: (N, K, K) fp16 — doubly-stochastic normalized.
 *
 * Algorithm per matrix (DeepSeek MHC sinkhorn):
 *   x = softmax(x, dim=-1) + eps
 *   x = x / col_sum(x)
 *   for repeat-1 times: x = x / row_sum(x); x = x / col_sum(x)
 *
 * K <= 64:  FP16 multi-matrix path — groups of (128/K) matrices in a tall
 *           tile, row ops amortized, col ops batched. Templated on TILE_COL
 *           (tile column width, >= K, 32-byte aligned: 16, 32, or 64).
 * K > 64:   FP32 per-matrix fallback (fp16 too lossy at K=128).
 */

#include <pto/pto-inst.hpp>

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

using namespace pto;

constexpr uint32_t UB_BYTES = 192 * 1024;
constexpr uint32_t MAX_DIM = 128;
constexpr uint32_t GROUP_ROWS =
    128;                           // tall-tile rows (= max K × max mats/group)
constexpr uint32_t MAX_MATS = 32;  // max matrices per group

#define A32(x) (((x) + 31u) & ~31u)

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

template <typename T, uint32_t N>
using V = Tile<TileType::Vec, T, 1, N, BLayout::RowMajor, -1, -1>;
template <typename T, uint32_t R, uint32_t C>
using T2 = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
template <typename T, uint32_t R>
using CV = Tile<TileType::Vec, T, R, 1, BLayout::ColMajor, DYNAMIC, DYNAMIC>;

using DS = Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T>
using S2 = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
template <typename T, uint32_t C>
using G2 = GlobalTensor<T, S2<T>, DS, Layout::ND>;

AICORE inline void pipeInit() {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}
AICORE inline void pipeDrain() {
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

// ---- FP16 multi-matrix path (K <= TILE_COL) ----
template <typename T, uint32_t TILE_COL>
AICORE void sinkhornMulti(__gm__ T *in, __gm__ T *out, uint32_t N, uint32_t K,
                          uint32_t repeat, float eps) {
  constexpr unsigned TC = TILE_COL, MR = GROUP_ROWS;
  constexpr unsigned rb = TC * sizeof(half);
  // UB: [tall_mat | tmp | col_vec | colsum_slots | batch_buf]
  constexpr unsigned MAT = 0, SZ = MR * TC * sizeof(half);
  constexpr unsigned TMP = A32(MAT + SZ), VC = A32(TMP + SZ);
  constexpr unsigned CS = A32(VC + A32(MR * sizeof(half)));
  constexpr unsigned BUF = A32(CS + MAX_MATS * A32(TC * sizeof(half)));
  constexpr unsigned MBR_R = (UB_BYTES - BUF) / (TC * sizeof(half));
  constexpr unsigned MBR = MBR_R < 4095 ? MBR_R : 4095;
  static_assert(BUF + MBR * TC * sizeof(half) <= UB_BYTES);

  set_mask_norm();
  set_vector_mask(-1, -1);
  if (K == 0 || K > TC) return;
  const uint32_t W = get_block_num() * get_subblockdim();
  const uint32_t w = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t KK = K * K, mpg = MR / K, bc = MBR / K;
  const half eh = (half)eps;
  const uint32_t bc0 = N / W, rem = N % W;
  const uint32_t s0 = w * bc0 + (w < rem ? w : rem),
                 cnt = bc0 + (w < rem ? 1 : 0);
  if (!cnt) return;
  constexpr unsigned CS_S = A32(TC * sizeof(half));

  pipeInit();
  for (uint32_t co = 0; co < cnt; co += bc) {
    const uint32_t ab = min(bc, cnt - co), ar = ab * K;
    __gm__ T *gi = in + (size_t)(s0 + co) * KK,
             *go = out + (size_t)(s0 + co) * KK;

    {
      V<T, MBR * TC> z(1, ar * TC);
      TASSIGN(z, BUF);
      TEXPANDS(z, (T)0);
      pipe_barrier(PIPE_V);
    }
    T2<T, MBR, TC> bh(ar, K);
    TASSIGN(bh, BUF);
    S2<T> bs(ar, K);
    DS bd(K);
    G2<T, TC> gi2(gi, bs, bd);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(bh, gi2);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (uint32_t g = 0; g < ab; g += mpg) {
      const uint32_t gc = min(mpg, ab - g), gr = gc * K, gf = gr * TC;
      const unsigned bo = BUF + g * K * TC * sizeof(T);

      {
        V<T, MR * TC> z(1, MR * TC);
        TASSIGN(z, MAT);
        TEXPANDS(z, (T)0);
        pipe_barrier(PIPE_V);
      }
      {
        V<T, MR * TC> s(1, gf), d(1, gf);
        TASSIGN(s, bo);
        TASSIGN(d, MAT);
        TMOV(d, s);
        pipe_barrier(PIPE_V);
      }

      T2<half, MR, TC> m(gr, K);
      TASSIGN(m, MAT);
      T2<half, MR, TC> t(gr, K);
      TASSIGN(t, TMP);
      CV<half, MR> v(gr, 1);
      TASSIGN(v, VC);

      // Softmax (6 barriers, amortized over gc matrices)
      TROWMAX(v, m, t);
      pipe_barrier(PIPE_V);
      TROWEXPANDSUB(m, m, v);
      pipe_barrier(PIPE_V);
      {
        V<half, MR * TC> f(1, gf);
        TASSIGN(f, MAT);
        TEXP(f, f);
        pipe_barrier(PIPE_V);
      }
      TROWSUM(v, m, t);
      pipe_barrier(PIPE_V);
      TROWEXPANDDIV(m, m, v);
      pipe_barrier(PIPE_V);
      {
        V<half, MR * TC> f(1, gf);
        TASSIGN(f, MAT);
        TADDS(f, f, eh);
        pipe_barrier(PIPE_V);
      }

      // Column normalize: K+1 barriers for all gc matrices
#define CN()                                \
  do {                                      \
    for (uint32_t i = 0; i < gc; ++i) {     \
      V<half, TC> c(1, K), r(1, K);         \
      TASSIGN(c, CS + i * CS_S);            \
      TASSIGN(r, MAT + i * K * rb);         \
      TMOV(c, r);                           \
    }                                       \
    pipe_barrier(PIPE_V);                   \
    for (uint32_t j = 1; j < K; ++j) {      \
      for (uint32_t i = 0; i < gc; ++i) {   \
        V<half, TC> c(1, K), r(1, K);       \
        TASSIGN(c, CS + i * CS_S);          \
        TASSIGN(r, MAT + (i * K + j) * rb); \
        TADD(c, c, r);                      \
      }                                     \
      pipe_barrier(PIPE_V);                 \
    }                                       \
    for (uint32_t i = 0; i < gc; ++i) {     \
      unsigned o = MAT + i * K * rb;        \
      V<half, TC> u(1, K);                  \
      TASSIGN(u, CS + i * CS_S);            \
      for (uint32_t j = 0; j < K; ++j) {    \
        V<half, TC> r(1, K);                \
        TASSIGN(r, o + j * rb);             \
        TDIV(r, r, u);                      \
      }                                     \
    }                                       \
    pipe_barrier(PIPE_V);                   \
  } while (0)

      CN();
      for (uint32_t it = 1; it < repeat; ++it) {
        TASSIGN(v, VC);
        TROWSUM(v, m, t);
        pipe_barrier(PIPE_V);
        TROWEXPANDDIV(m, m, v);
        pipe_barrier(PIPE_V);
        CN();
      }
#undef CN

      {
        V<T, MR * TC> s(1, gf), d(1, gf);
        TASSIGN(s, MAT);
        TASSIGN(d, bo);
        TMOV(d, s);
        pipe_barrier(PIPE_V);
      }
    }

    G2<T, TC> go2(go, bs, bd);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(go2, bh);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }
  pipeDrain();
}

// ---- FP32 per-matrix fallback (K > 64) ----
template <typename T>
AICORE void sinkhornFP32(__gm__ T *in, __gm__ T *out, uint32_t N, uint32_t K,
                         uint32_t repeat, float eps) {
  constexpr unsigned D = MAX_DIM, rf = D * sizeof(float);
  constexpr unsigned MH = 0, MF = MH + D * D * sizeof(half),
                     TF = MF + D * D * sizeof(float),
                     VF = TF + D * D * sizeof(float);
  static_assert(VF + D * sizeof(float) <= UB_BYTES);

  set_mask_norm();
  set_vector_mask(-1, -1);
  if (K == 0 || K > D) return;
  const uint32_t W = get_block_num() * get_subblockdim();
  const uint32_t w = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t KK = K * K, fl = K * D;

  pipeInit();
  for (uint32_t bi = w; bi < N; bi += W) {
    __gm__ T *gi = in + (size_t)bi * KK, *go = out + (size_t)bi * KK;
    {
      V<T, D * D> z(1, fl);
      TASSIGN(z, MH);
      TEXPANDS(z, (T)0);
      pipe_barrier(PIPE_V);
    }
    T2<T, D, D> mH(K, K);
    TASSIGN(mH, MH);
    S2<T> sh(K, K);
    DS st(K);
    G2<T, D> gI(gi, sh, st);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(mH, gI);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    {
      V<T, D * D> h(1, fl);
      V<float, D * D> f(1, fl);
      TASSIGN(h, MH);
      TASSIGN(f, MF);
      TCVT(f, h, RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
    }

    T2<float, D, D> m(K, K);
    TASSIGN(m, MF);
    T2<float, D, D> t(K, K);
    TASSIGN(t, TF);
    CV<float, D> v(K, 1);
    TASSIGN(v, VF);

    TROWMAX(v, m, t);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(m, m, v);
    pipe_barrier(PIPE_V);
    {
      V<float, D * D> f(1, fl);
      TASSIGN(f, MF);
      TEXP(f, f);
      pipe_barrier(PIPE_V);
    }
    TROWSUM(v, m, t);
    pipe_barrier(PIPE_V);
    {
      V<float, D> u(1, K);
      TASSIGN(u, VF);
      TADDS(u, u, eps);
      pipe_barrier(PIPE_V);
    }
    TROWEXPANDDIV(m, m, v);
    pipe_barrier(PIPE_V);
    {
      V<float, D * D> f(1, fl);
      TASSIGN(f, MF);
      TADDS(f, f, eps);
      pipe_barrier(PIPE_V);
    }
    {
      V<float, D> c(1, K);
      TASSIGN(c, VF);
      TCOLSUM(c, m, t, false);
      pipe_barrier(PIPE_V);
      TADDS(c, c, eps);
      pipe_barrier(PIPE_V);
    }
    for (uint32_t r = 0; r < K; ++r) {
      V<float, D> row(1, K), u(1, K);
      TASSIGN(row, MF + r * rf);
      TASSIGN(u, VF);
      TDIV(row, row, u);
      pipe_barrier(PIPE_V);
    }

    for (uint32_t it = 1; it < repeat; ++it) {
      TASSIGN(v, VF);
      TROWSUM(v, m, t);
      pipe_barrier(PIPE_V);
      {
        V<float, D> u(1, K);
        TASSIGN(u, VF);
        TADDS(u, u, eps);
        pipe_barrier(PIPE_V);
      }
      TROWEXPANDDIV(m, m, v);
      pipe_barrier(PIPE_V);
      {
        V<float, D> c(1, K);
        TASSIGN(c, VF);
        TCOLSUM(c, m, t, false);
        pipe_barrier(PIPE_V);
        TADDS(c, c, eps);
        pipe_barrier(PIPE_V);
      }
      for (uint32_t r = 0; r < K; ++r) {
        V<float, D> row(1, K), u(1, K);
        TASSIGN(row, MF + r * rf);
        TASSIGN(u, VF);
        TDIV(row, row, u);
        pipe_barrier(PIPE_V);
      }
    }

    {
      V<T, D * D> h(1, fl);
      V<float, D * D> f(1, fl);
      TASSIGN(h, MH);
      TASSIGN(f, MF);
      TCVT(h, f, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
    }
    G2<T, D> gO(go, sh, st);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gO, mH);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }
  pipeDrain();
}

// ---- Dispatch ----
template <typename T>
AICORE void sinkhorn(__gm__ T *in, __gm__ T *out, uint32_t N, uint32_t K,
                     uint32_t repeat, float eps) {
  if (K > 0 && K <= 16)
    sinkhornMulti<T, 16>(in, out, N, K, repeat, eps);
  else if (K <= 32)
    sinkhornMulti<T, 32>(in, out, N, K, repeat, eps);
  else if (K <= 64)
    sinkhornMulti<T, 64>(in, out, N, K, repeat, eps);
  else if (K <= 128)
    sinkhornFP32<T>(in, out, N, K, repeat, eps);
}
#endif

extern "C" __global__ AICORE void sinkhorn_ds_fp16(GM_ADDR input,
                                                   GM_ADDR output, uint32_t N,
                                                   uint32_t K, uint32_t repeat,
                                                   float eps) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  sinkhorn<half>((__gm__ half *)input, (__gm__ half *)output, N, K, repeat,
                 eps);
#else
  (void)input;
  (void)output;
  (void)N;
  (void)K;
  (void)repeat;
  (void)eps;
#endif
}

extern "C" void call_sinkhorn_ds_kernel(uint32_t blockDim, void *stream,
                                        uint8_t *input, uint8_t *output,
                                        uint32_t N, uint32_t K, uint32_t repeat,
                                        float eps) {
  sinkhorn_ds_fp16<<<blockDim * 2, nullptr, stream>>>(input, output, N, K,
                                                      repeat, eps);
}
