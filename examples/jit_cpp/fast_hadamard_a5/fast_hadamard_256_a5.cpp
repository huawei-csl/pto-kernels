// fast_hadamard_256_a5 — Walsh-Hadamard by deinterleave-load butterfly.
// The even/odd split rides the MTE2 load (vlds DINTLV_B16) and the
// halves-concat recombine rides the MTE3 store, so only vadd/vsub reach the
// vector pipe. In-place on a UB tile. N = 32..2048, each within 0.90..0.96 of
// its own DMA copy floor; the floor reference is copy_ref_256_a5.cpp, built
// separately.
#include <pto/pto-inst.hpp>
using namespace pto;

#ifndef HAD_N
#define HAD_N 256
#endif
#ifndef ROWS_PER_TILE
#define ROWS_PER_TILE 64
#endif
#ifndef NBUF
#define NBUF 4
#endif
#ifndef PREFETCH
#define PREFETCH 2
#endif

#ifdef __CCE_AICORE__
constexpr unsigned VL16 = sizeof(vector_f16) / sizeof(half);  // lanes/register
constexpr unsigned WIN = 2 * VL16;  // butterfly window: two registers
constexpr unsigned SLOTS = 8;       // DOU() below is hand-written this wide
static_assert(SLOTS == 8, "DOU() emits 8 slots; GPI is derived from SLOTS");
constexpr unsigned EVENTS = 8;  // size of ev[] below
constexpr unsigned UB_ALIGN = 512;
// The toolkit's own UB size: 256 KB on A5, 192 on A2/A3, 128 on Kirin. Defined
// only in the device pass, like __DAV_VEC__, which is why it is read in here.
constexpr unsigned UB_BYTES = PTO_UBUF_SIZE_BYTES;

// A constexpr *function* cannot be called from [aicore] code; a value template
// can.
template <unsigned N>
struct Log2 {
  static constexpr unsigned v = 1 + Log2<(N >> 1)>::v;
};
template <>
struct Log2<1> {
  static constexpr unsigned v = 0;
};

// Shape of one instantiation. A "group" is the unit one unroll slot processes.
//   N < WIN  PACK:  R rows share a window. The split is on the low bit of the
//                   within-row index, so rows never mix; the result emerges
//                   rotated right by log2(N), which ROT vdintlv rounds undo.
//   N > WIN  CHUNK: stages pair adjacent elements and their output halves
//                   concatenate, so a row splits into independent WIN chunks.
// Packing and chunking are mutually exclusive, asserted below.
template <unsigned N, unsigned ROWS, unsigned NB, unsigned PF>
struct Cfg {
  static constexpr unsigned LOG2N = Log2<N>::v, LOG2W = Log2<WIN>::v;
  static constexpr unsigned R = N < WIN ? WIN / N : 1;  // rows per window
  static constexpr unsigned GRP = N * R;                // == WIN when N <= WIN
  static constexpr unsigned ROT = LOG2W - (LOG2N < LOG2W ? LOG2N : LOG2W);
  static constexpr unsigned PEEL = ROT ? 1 : 0;  // stages peeled off the sweep
  static constexpr unsigned LANES = GRP / 2;  // half a window, not half a row
  static constexpr unsigned VL = LANES < VL16 ? LANES : VL16;
  static constexpr unsigned CHUNKS = LANES / VL;
  static constexpr unsigned GPI = SLOTS / CHUNKS;  // groups per iteration
  static constexpr unsigned GROUPS = ROWS / R, ITERS = GROUPS / GPI;
  static constexpr unsigned FLAT = ROWS * N;  // f16 elements per tile
  static constexpr unsigned XSTRIDE =
      (FLAT * sizeof(half) + UB_ALIGN - 1) & ~(UB_ALIGN - 1);

  static_assert(N >= 32 && N <= 2048 && !(N & (N - 1)), "N: pow2 in [32,2048]");
  static_assert(CHUNKS <= SLOTS && SLOTS % CHUNKS == 0, "N too wide to unroll");
  static_assert(LANES % VL == 0, "GRP/2 must be whole vectors");
  static_assert(R == 1 || CHUNKS == 1, "pack and chunk are exclusive");
  static_assert(ROWS % R == 0, "ROWS must be a multiple of WIN/N");
  static_assert(GROUPS % GPI == 0, "GROUPS must be a multiple of GPI");
  static_assert(NB * XSTRIDE <= UB_BYTES, "UB overflow");
  static_assert(NB <= EVENTS, "NBUF exceeds the event-id array");
  static_assert(PF < NB, "PF == NB drains every MTE3->MTE2 token: deadlock");
};

#ifdef __DAV_VEC__
#define DOU(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)

template <unsigned N, unsigned ROWS, unsigned NB, unsigned PF>
__tf__ static AICORE void bfly(__ubuf__ half *wb) {
  using C = Cfg<N, ROWS, NB, PF>;
  constexpr unsigned GRP = C::GRP, LANES = C::LANES, VL = C::VL;
  constexpr unsigned CHUNKS = C::CHUNKS, GPI = C::GPI, ITERS = C::ITERS;
  __VEC_SCOPE__ {
    uint32_t la = VL;
    MaskReg pAll = CreatePredicate<half>(la);
    vector_f16 e0, e1, e2, e3, e4, e5, e6, e7, o0, o1, o2, o3, o4, o5, o6, o7;
    vector_f16 s0, s1, s2, s3, s4, s5, s6, s7, d0, d1, d2, d3, d4, d5, d6, d7;
    // Slot i is (group, chunk) = (i / CHUNKS, i % CHUNKS), so all loads precede
    // all stores. Required: a stage's sums compact into the group's lower half,
    // which a lower-numbered chunk still has to read, so a per-chunk load/store
    // loop aliases in place -- and it passes at GRP=512 before silently
    // corrupting at larger sizes.
#define GRP_OF(i) (((uint32_t)(i) / CHUNKS) * GRP)
#define CH_OF(i) ((uint32_t)(i) % CHUNKS)
#define LD(i) \
  vlds(e##i, o##i, wb + base + GRP_OF(i) + CH_OF(i) * 2u * VL, 0, DINTLV_B16);
#define AD(i) vadd(s##i, e##i, o##i, pAll);
#define SU(i) vsub(d##i, e##i, o##i, pAll);
#define ST(i, LO, HI)                                                    \
  vsts(LO##i, wb + base + GRP_OF(i) + CH_OF(i) * VL, 0, NORM_B16, pAll); \
  vsts(HI##i, wb + base + GRP_OF(i) + LANES + CH_OF(i) * VL, 0, NORM_B16, pAll);
#define ST_SD(i) ST(i, s, d)
#define ST_EO(i) ST(i, e, o)
#define NOROT(i)
    // Rotation tail, register-only: each vdintlv rotates the window right by
    // one and ping-pongs into the pair AD/SU just freed. Odd ROT lands in
    // (e,o).
#define R1(i) vdintlv(e##i, o##i, s##i, d##i);
#define R2(i) R1(i) vdintlv(s##i, d##i, e##i, o##i);
#define R3(i) R2(i) vdintlv(e##i, o##i, s##i, d##i);
    // Step by a literal 1 with the stride folded into base: the loop analyser
    // verifies a tripcount only for a literal step, and 1 divides any bound.
#define PASS(ROT_, ST_)                               \
  for (uint16_t it = 0; it < (uint16_t)ITERS; ++it) { \
    const uint32_t base = (uint32_t)it * (GPI * GRP); \
    DOU(LD) DOU(AD) DOU(SU) DOU(ROT_) DOU(ST_)        \
  }                                                   \
  mem_bar(VST_VLD);
    for (uint16_t st = 0; st < (uint16_t)(C::LOG2N - C::PEEL); ++st) {
      PASS(NOROT, ST_SD)
    }
    // Last stage peeled, not branched: a runtime branch in __VEC_SCOPE__ is
    // risky.
    if constexpr (C::ROT == 1) {
      PASS(R1, ST_EO)
    } else if constexpr (C::ROT == 2) {
      PASS(R2, ST_SD)
    } else if constexpr (C::ROT == 3) {
      PASS(R3, ST_EO)
    }
  }
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

template <unsigned N, unsigned ROWS, unsigned NB, unsigned PF>
__global__ AICORE void hadamard(__gm__ void *x_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  using C = Cfg<N, ROWS, NB, PF>;
  constexpr unsigned FLAT = C::FLAT, XSTRIDE = C::XSTRIDE;
  set_mask_norm();
  set_vector_mask(-1, -1);
  using Sh = pto::Shape<1, 1, 1, 1, FLAT>;
  using St = pto::Stride<1, 1, 1, FLAT, 1>;
  using T = Tile<TileType::Vec, half, 1, FLAT, BLayout::RowMajor, 1, FLAT>;
  const event_t ev[EVENTS] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3,
                              EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7};
  const uint32_t cid = get_block_idx(), nc = get_block_num();
  const uint32_t tiles = batch / ROWS;

#define XOFF(i) ((unsigned)(i) * XSTRIDE)
  // async load of this core's K-th tile into buffer K % NB
#define ISSUE_LOAD(K)                                        \
  do {                                                       \
    uint32_t _tb = cid + (uint32_t)(K) * nc;                 \
    if (_tb < tiles) {                                       \
      const int _p = (uint32_t)(K) % NB;                     \
      wait_flag(PIPE_MTE3, PIPE_MTE2, ev[_p]);               \
      T _xt;                                                 \
      TASSIGN(_xt, XOFF(_p));                                \
      GlobalTensor<half, Sh, St> _g(                         \
          (__gm__ half *)x_gm + (uint64_t)_tb * FLAT, Sh()); \
      TLOAD(_xt, _g);                                        \
      set_flag(PIPE_MTE2, PIPE_V, ev[_p]);                   \
    }                                                        \
  } while (0)

  for (unsigned i = 0; i < NB; ++i) set_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);
  for (unsigned i = 0; i < PF; ++i) ISSUE_LOAD(i);

  uint32_t k = 0;
  for (uint32_t tb = cid; tb < tiles; tb += nc, ++k) {
    const int p = k % NB;
    ISSUE_LOAD(k + PF);  // overlaps this tile's compute
    wait_flag(PIPE_MTE2, PIPE_V, ev[p]);
    bfly<N, ROWS, NB, PF>((__ubuf__ half *)(uintptr_t)XOFF(p));
    set_flag(PIPE_V, PIPE_MTE3, ev[p]);
    wait_flag(PIPE_V, PIPE_MTE3, ev[p]);
    T xt;
    TASSIGN(xt, XOFF(p));
    GlobalTensor<half, Sh, St> g((__gm__ half *)x_gm + (uint64_t)tb * FLAT,
                                 Sh());
    TSTORE(g, xt);
    set_flag(PIPE_MTE3, PIPE_MTE2, ev[p]);
  }
  for (unsigned i = 0; i < NB; ++i) wait_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);
#else
  (void)x_gm;
  (void)batch;
#endif
}

extern "C" void call_hadamard256(uint32_t bd, void *s, uint8_t *x, uint32_t b) {
  hadamard<HAD_N, ROWS_PER_TILE, NBUF, PREFETCH><<<bd, nullptr, s>>>(x, b);
}
