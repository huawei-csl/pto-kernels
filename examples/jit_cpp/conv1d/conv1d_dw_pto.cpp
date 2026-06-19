#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))
#define ALIGN_UP(x, y) (DIV_ROUNDUP((x), (y)) * (y))

// ===========================================================================
// DEPTHWISE fused causal conv1d + bias + (optional) SiLU   (per-channel, K=4)
//
//   y[b,i,c] = act( bias[c] + sum_{k=max(0,K-1-i)..K-1} W[k,c] * x[b, i-K+1+k, c] ),
//   x[<0]=0
//
// Per-channel K-tap filter (Mamba/GDN short conv). x,y are [batch, seqLen, W]
// row-major; W (= channels) is the lane axis, seqLen (= seq) the conv axis.
// Weights W[K,W] + bias[W] are fp32 GM tensors. fp16 OR bf16 I/O, fp32
// accumulate.
//
// 2-D-plus-batch work grid: units = batch x num_wt(W-tiles) x
// lchunks(L-chunks). Each unit produces outputs [l0,l1) for channels
// [wbase,wbase+lanes) of one sequence, replaying K-1 causal halo rows to prime
// its accumulators. The grid fills all cores: batch supplies parallelism first,
// then W-tiles, then L-chunks; col_w is widened for coalesced stores. (See
// processUnit + the grid below.)
//
// UB (per lane, fp32 unless noted): W0..W3(4) + bias(1) + acc0..3(4) +
//   t1..t3(3) + xin_f(1) = 13 fp32; xin_h + out0 + out1 = 3 IoT. MAX_W=3072.
// NOTE: TAXPY/TMUL/TCVT need pto-isa v9.0.0; namespace `csilu` (v9 has
// pto::detail).
// ===========================================================================

namespace csilu {

template <typename T, typename TileT>
AICORE inline void siluTile(TileT& dst, TileT& src, TileT& tmp) {
  TMULS(tmp, src, (T)-1);
  pipe_barrier(PIPE_V);
  TEXP(tmp, tmp);
  pipe_barrier(PIPE_V);
  TADDS(tmp, tmp, (T)1);
  pipe_barrier(PIPE_V);
  TDIV(dst, src, tmp);
}

// Process ONE unit: outputs [l0,l1) for channels [wbase,wbase+lanes) of the
// sequence whose first row is at element offset row_base. x[<0]=0 (no cache).
template <typename IoT, typename AccT, uint32_t K, uint32_t MAX_W>
AICORE inline void processUnit(__gm__ IoT* x, __gm__ IoT* y, __gm__ AccT* wgt,
                               __gm__ AccT* bia, uint32_t W, uint64_t row_base,
                               uint32_t wbase, int32_t lanes, uint32_t l0,
                               uint32_t l1, uint32_t activation) {
  using GShapeIo = pto::Shape<1, 1, 1, 1, DYNAMIC>;
  using GStride = pto::Stride<1, 1, 1, 1, 1>;
  using GIo = pto::GlobalTensor<IoT, GShapeIo, GStride>;
  using GAcc = pto::GlobalTensor<AccT, GShapeIo, GStride>;
  using TIo = Tile<TileType::Vec, IoT, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;
  using TAcc =
      Tile<TileType::Vec, AccT, 1, MAX_W, BLayout::RowMajor, 1, DYNAMIC>;

  constexpr uint32_t FB = MAX_W * sizeof(AccT);
  constexpr uint32_t HB = MAX_W * sizeof(IoT);
  const uint32_t UB_W[4] = {0, FB, 2 * FB, 3 * FB};
  const uint32_t UB_BIAS = 4 * FB;
  const uint32_t UB_ACC0 = 5 * FB;
  const uint32_t UB_T[4] = {0, 9 * FB, 10 * FB, 11 * FB};
  const uint32_t UB_XINF = 12 * FB;
  const uint32_t UB_XINH = 13 * FB;
  const uint32_t UB_OUT[2] = {13 * FB + HB, 13 * FB + 2 * HB};

  const uint32_t hstart = (l0 > (K - 1)) ? (l0 - (K - 1)) : 0u;

  // ---- per-channel weights + bias (resident for this unit) ----
  for (uint32_t k = 0; k < K; ++k) {
    GAcc wG(wgt + (uint64_t)k * W + wbase, {lanes});
    TAcc wT(lanes);
    TASSIGN(wT, UB_W[k]);
    TLOAD(wT, wG);
  }
  {
    GAcc bG(bia + wbase, {lanes});
    TAcc bT(lanes);
    TASSIGN(bT, UB_BIAS);
    TLOAD(bT, bG);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // xin_h initially free
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);

  // PROLOGUE: issue the first input load (x[hstart]) so iter 0 can prefetch
  // x[1].
  if (hstart < l1) {
    GIo xG0(x + row_base + (uint64_t)hstart * W + wbase, {lanes});
    TIo xin_h0(lanes);
    TASSIGN(xin_h0, UB_XINH);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(xin_h0, xG0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  }

  for (uint32_t j = hstart; j < l1; ++j) {
    TIo xin_h(lanes);
    TAcc xin_f(lanes);
    TASSIGN(xin_h, UB_XINH);
    TASSIGN(xin_f, UB_XINF);

    // (1) consume current: x[j] was loaded by the prologue / previous prefetch.
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(xin_f, xin_h, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // xin_h free again

    // (2) prefetch next: load x[j+1] into the SAME xin_h; overlaps compute
    // below.
    if (j + 1 < l1) {
      GIo xGn(x + row_base + (uint64_t)(j + 1) * W + wbase, {lanes});
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(xin_h, xGn);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    pipe_barrier(PIPE_V);

    // scatter: products (only outputs in [l0,l1))
    for (uint32_t k = 0; k < K; ++k) {
      const uint32_t out = j + (K - 1) - k;
      if (out < l0 || out >= l1) continue;
      TAcc wT(lanes);
      TASSIGN(wT, UB_W[k]);
      if (j == 0 || k == 0) {
        TAcc acc(lanes);
        TASSIGN(acc, UB_ACC0 + (out & (K - 1)) * FB);
        TMUL(acc, xin_f, wT);
      } else {
        TAcc t(lanes);
        TASSIGN(t, UB_T[k]);
        TMUL(t, xin_f, wT);
      }
    }
    pipe_barrier(PIPE_V);
    if (j != 0) {
      for (uint32_t k = 1; k < K; ++k) {
        const uint32_t out = j + (K - 1) - k;
        if (out < l0 || out >= l1) continue;
        TAcc acc(lanes);
        TAcc t(lanes);
        TASSIGN(acc, UB_ACC0 + (out & (K - 1)) * FB);
        TASSIGN(t, UB_T[k]);
        TADD(acc, acc, t);
      }
    }
    pipe_barrier(PIPE_V);

    if (j < l0) continue;  // halo row: primed accumulators only

    const uint32_t slot = j & (K - 1);
    const uint32_t ob = j & 1u;
    const event_t oev = (event_t)(1u + ob);
    TAcc acc(lanes);
    TAcc bT(lanes);
    TAcc tmp(lanes);
    TIo outT(lanes);
    TASSIGN(acc, UB_ACC0 + slot * FB);
    TASSIGN(bT, UB_BIAS);
    TASSIGN(tmp, UB_T[1]);
    TASSIGN(outT, UB_OUT[ob]);

    TADD(acc, acc, bT);
    pipe_barrier(PIPE_V);
    if (activation) {
      siluTile<AccT>(acc, acc, tmp);
      pipe_barrier(PIPE_V);
    }
    wait_flag(PIPE_MTE3, PIPE_V, oev);
    TCVT(outT, acc, pto::RoundMode::CAST_NONE);

    GIo yG(y + row_base + (uint64_t)j * W + wbase, {lanes});
    set_flag(PIPE_V, PIPE_MTE3, oev);
    wait_flag(PIPE_V, PIPE_MTE3, oev);
    TSTORE(yG, outT);
    set_flag(PIPE_MTE3, PIPE_V, oev);
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
}

template <typename IoT, typename AccT, uint32_t K, uint32_t MAX_W>
AICORE void runConvSiluBatched(__gm__ IoT* x, __gm__ IoT* y, __gm__ AccT* wgt,
                               __gm__ AccT* bia, uint32_t batch,
                               uint32_t seqLen, uint32_t W,
                               uint32_t activation) {
  static_assert((K > 0u) && ((K & (K - 1)) == 0u), "K power of two");
  static_assert(K == 4, "this prototype is specialised to K=4");

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_cores = get_block_num();
  const uint32_t core_id = get_block_idx();
  if (seqLen == 0 || batch == 0 || W == 0) return;

  constexpr uint32_t LC_MIN = 32u;
  // batch supplies parallelism first; each sequence needs `target` (wt x
  // lchunk) units.
  uint32_t target = DIV_ROUNDUP(num_cores, batch);
  if (target < 1) target = 1;

  uint32_t max_chunks = DIV_ROUNDUP(seqLen, LC_MIN);
  if (max_chunks < 1) max_chunks = 1;

  const uint32_t nwt_lb = DIV_ROUNDUP(W, MAX_W);
  const uint32_t nwt_fill = DIV_ROUNDUP(target, max_chunks);
  uint32_t num_wt = nwt_lb > nwt_fill ? nwt_lb : nwt_fill;

  const uint32_t nwt_128 = DIV_ROUNDUP(W, 128u);
  if (num_wt > nwt_128) num_wt = nwt_128;
  if (num_wt < 1) num_wt = 1;

  uint32_t col_w = ALIGN_UP(DIV_ROUNDUP(W, num_wt), 128u);
  if (col_w < 128u) col_w = 128u;
  if (col_w > MAX_W) col_w = MAX_W;
  if (col_w > W) col_w = W;

  num_wt = DIV_ROUNDUP(W, col_w);
  uint32_t lchunks = DIV_ROUNDUP(target, num_wt);
  if (lchunks < 1) lchunks = 1;
  if (lchunks > max_chunks) lchunks = max_chunks;

  const uint32_t lc_len = DIV_ROUNDUP(seqLen, lchunks);
  const uint32_t total_units = batch * num_wt * lchunks;

  for (uint32_t unit = core_id; unit < total_units; unit += num_cores) {
    const uint32_t lc = unit % lchunks;
    const uint32_t t2 = unit / lchunks;
    const uint32_t wt = t2 % num_wt;
    const uint32_t seq = t2 / num_wt;
    const uint32_t wbase = wt * col_w;
    const uint32_t rem = W - wbase;
    const int32_t lanes = rem > col_w ? (int32_t)col_w : (int32_t)rem;
    const uint32_t l0 = lc * lc_len;
    if (l0 >= seqLen) continue;
    uint32_t l1 = l0 + lc_len;
    if (l1 > seqLen) l1 = seqLen;
    const uint64_t row_base = (uint64_t)seq * seqLen * W;
    processUnit<IoT, AccT, K, MAX_W>(x, y, wgt, bia, W, row_base, wbase, lanes,
                                     l0, l1, activation);
  }
}

}  // namespace csilu

// ---- single-sequence entry (back-compat: x,y [L,W] fp16, wgt[K,W]/bia[W]
// fp32) ----
extern "C" __global__ AICORE void conv1d_dw_kernel(__gm__ uint8_t* x,
                                                   __gm__ uint8_t* y,
                                                   __gm__ uint8_t* wgt,
                                                   __gm__ uint8_t* bia,
                                                   uint32_t L_in, uint32_t W) {
#if defined(__DAV_VEC__)
  constexpr uint32_t K = 4, MAX_W = 3072;
  csilu::runConvSiluBatched<half, float, K, MAX_W>(
      (__gm__ half*)x, (__gm__ half*)y, (__gm__ float*)wgt, (__gm__ float*)bia,
      1u, L_in, W, 1u);
#else
  (void)x;
  (void)y;
  (void)wgt;
  (void)bia;
  (void)L_in;
  (void)W;
#endif
}

// ---- batched fp16 entry: x,y [batch,seqLen,W] fp16, wgt[K,W]/bia[W] fp32 ----
extern "C" __global__ AICORE void conv1d_dw_batched_kernel(
    __gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* wgt,
    __gm__ uint8_t* bia, uint32_t batch, uint32_t seqLen, uint32_t W,
    uint32_t activation) {
#if defined(__DAV_VEC__)
  constexpr uint32_t K = 4, MAX_W = 3072;
  csilu::runConvSiluBatched<half, float, K, MAX_W>(
      (__gm__ half*)x, (__gm__ half*)y, (__gm__ float*)wgt, (__gm__ float*)bia,
      batch, seqLen, W, activation);
#else
  (void)x;
  (void)y;
  (void)wgt;
  (void)bia;
  (void)batch;
  (void)seqLen;
  (void)W;
  (void)activation;
#endif
}

// ---- batched bf16 entry: x,y [batch,seqLen,W] bf16, wgt[K,W]/bia[W] fp32 ----
extern "C" __global__ AICORE void conv1d_dw_batched_bf16_kernel(
    __gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* wgt,
    __gm__ uint8_t* bia, uint32_t batch, uint32_t seqLen, uint32_t W,
    uint32_t activation) {
#if defined(__DAV_VEC__)
  constexpr uint32_t K = 4, MAX_W = 3072;
  csilu::runConvSiluBatched<bfloat16_t, float, K, MAX_W>(
      (__gm__ bfloat16_t*)x, (__gm__ bfloat16_t*)y, (__gm__ float*)wgt,
      (__gm__ float*)bia, batch, seqLen, W, activation);
#else
  (void)x;
  (void)y;
  (void)wgt;
  (void)bia;
  (void)batch;
  (void)seqLen;
  (void)W;
  (void)activation;
#endif
}

extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* x,
                            uint8_t* y, uint8_t* wgt, uint8_t* bia,
                            uint32_t L_in, uint32_t W) {
  conv1d_dw_kernel<<<blockDim * 2, nullptr, stream>>>(x, y, wgt, bia, L_in, W);
}

extern "C" void call_kernel_batched(uint32_t blockDim, void* stream, uint8_t* x,
                                    uint8_t* y, uint8_t* wgt, uint8_t* bia,
                                    uint32_t batch, uint32_t seqLen, uint32_t W,
                                    uint32_t activation) {
  conv1d_dw_batched_kernel<<<blockDim * 2, nullptr, stream>>>(
      x, y, wgt, bia, batch, seqLen, W, activation);
}

extern "C" void call_kernel_batched_bf16(uint32_t blockDim, void* stream,
                                         uint8_t* x, uint8_t* y, uint8_t* wgt,
                                         uint8_t* bia, uint32_t batch,
                                         uint32_t seqLen, uint32_t W,
                                         uint32_t activation) {
  conv1d_dw_batched_bf16_kernel<<<blockDim * 2, nullptr, stream>>>(
      x, y, wgt, bia, batch, seqLen, W, activation);
}
