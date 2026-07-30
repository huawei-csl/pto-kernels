// fast_hadamard_256_a5 — Walsh-Hadamard via DEINTERLEAVE-LOAD butterfly.
// Default block size is HAD_N=256; N=32..2048 are supported and all of them run
// within 0.90..0.96 of their own DMA copy floor, so none is "the fast one" --
// N=32 in fact measures highest (0.96) because it needs the fewest stages.
// Each stage does the even/odd split on the MTE2 load (vlds DINTLV_B16) and the
// concat-halves recombine on the MTE3 store, leaving only vadd/vsub on the
// vector-execute pipe. In-place on a UB tile.
//
// Standalone: this TU builds and runs on its own. The copy-floor reference it
// is benchmarked against lives in copy_ref_256_a5.cpp, which the benchmark
// builds separately with a matching ROWS_PER_TILE.
#include <pto/pto-inst.hpp>
using namespace pto;

#ifndef HAD_N
#define HAD_N 256
#endif
#ifndef HAD_LOG2N
#define HAD_LOG2N 8
#endif
#ifndef ROWS_PER_TILE
#define ROWS_PER_TILE 64
#endif
#ifndef HAD_UNROLL
#define HAD_UNROLL 8
#endif

#ifdef __CCE_AICORE__
// Half a row: the deinterleave load splits HAD_N elements into two LANES-wide
// halves, so LANES is HAD_N/2 by construction. It must also fit one fp16 vector
// register (128 elements) -- at HAD_N=512 the predicate silently covers only
// part of each half and the kernel computes garbage *without* failing, so this
// existed, because the predicate silently covered only part of each half-row.
//
// Rows wider than one vector are handled by CHUNKS: a stage pairs only ADJACENT
// elements, and its two output halves concatenate, so a row splits into
// independent 2*VL-element chunks (verified against x@H for N up to 2048). The
// chunk loop reuses the same registers, so wide N costs no extra register
// pressure and HAD_UNROLL need not shrink.
#define VEC_F16_LANES 128
// ---- packing: for N < 256, R = 256/N rows share one 256-element window -----
// The butterfly is index-blind, so log2(N) full-width stages over a window of R
// rows compute each row's transform independently. Rows never mix: the split is
// on the low bit of the WITHIN-ROW index and N is even, so row r's evens always
// land contiguously in group r. Without packing, a stage at N=32 drives only 16
// of 128 lanes and the kernel is vector-issue-bound rather than memory-bound.
//
// A packed window comes out with its index rotated right by log2(N), so
// HAD_ROT = 8 - log2(N) further rotations finish it. One vdintlv rotates a
// whole 256-element (2-register) window right by one, register-to-register;
// that is device-verified for 1, 2 and 3 rounds against ror(index, k).
//
// "Group" is the unit one DOU slot processes: R rows when N<256, one row at
// N=256, one chunk of a row when N>256. GRP = N*R is 256 for every N <= 256, so
// every derivation below is the pre-packing one with HAD_N -> HAD_GRP.
#define HAD_R (HAD_N < 256 ? (256 / HAD_N) : 1)
#define HAD_GRP (HAD_N * HAD_R)
#define HAD_ROT (8 - (HAD_LOG2N < 8 ? HAD_LOG2N : 8))  // == log2(HAD_R)
#define HAD_PEEL (HAD_ROT ? 1 : 0)                     // stages peeled off
// Groups wider than one vector are handled by CHUNKS: a stage pairs only
// ADJACENT elements and its two output halves concatenate, so a wide group
// splits into independent 2*VL-element pieces. Packing and chunking are
// mutually exclusive (asserted below), which is what keeps this readable.
#define HAD_VL ((HAD_GRP / 2) < VEC_F16_LANES ? (HAD_GRP / 2) : VEC_F16_LANES)
#define HAD_CHUNKS ((HAD_GRP / 2) / HAD_VL)
// Groups per iteration = HAD_UNROLL / CHUNKS, written as literals: the vector
// loop analyser only verifies the tripcount when the step is a literal token,
// and an arithmetic expression here costs a "tripcount would be inaccurate"
// warning. The static_asserts below check these literals against the real
// derivation.
#if HAD_GRP <= 256
#define HAD_GRPS_PER_ITER 8
#elif HAD_GRP == 512
#define HAD_GRPS_PER_ITER 4
#elif HAD_GRP == 1024
#define HAD_GRPS_PER_ITER 2
#elif HAD_GRP == 2048
#define HAD_GRPS_PER_ITER 1
#else
#define HAD_GRPS_PER_ITER 0  // unsupported; the static_assert below fires
#endif
constexpr unsigned GRP = HAD_GRP;
constexpr unsigned LANES = HAD_GRP / 2;  // half a WINDOW (not half a row)
constexpr unsigned VL = HAD_VL;
constexpr unsigned CHUNKS = HAD_CHUNKS;  // 1 for N <= 256
constexpr unsigned GRPS_PER_ITER = HAD_GRPS_PER_ITER;
constexpr unsigned GROUPS = ROWS_PER_TILE / HAD_R;
static_assert(HAD_UNROLL == 8, "the DOU() unroll body is exactly 8 slots wide");
static_assert(CHUNKS <= HAD_UNROLL, "HAD_GRP too wide for one unroll set");
static_assert(
    GRPS_PER_ITER * CHUNKS == HAD_UNROLL,
    "groups-per-iteration literal disagrees with HAD_UNROLL / CHUNKS");
static_assert(HAD_UNROLL % CHUNKS == 0, "HAD_UNROLL must divide by CHUNKS");
static_assert(LANES % VL == 0, "HAD_GRP/2 must be a whole number of vectors");
static_assert((1u << HAD_LOG2N) == HAD_N, "HAD_LOG2N must be log2(HAD_N)");
static_assert((1u << HAD_ROT) == HAD_R, "HAD_ROT must be log2(HAD_R)");
static_assert(HAD_R == 1 || CHUNKS == 1,
              "packing and chunking are mutually exclusive by construction");
static_assert(ROWS_PER_TILE % HAD_R == 0,
              "a packed window must not straddle a tile: ROWS_PER_TILE must be "
              "a multiple of 256/HAD_N");
static_assert(GROUPS % GRPS_PER_ITER == 0,
              "groups per tile must be a multiple of groups per iteration");
constexpr unsigned FLAT = ROWS_PER_TILE * HAD_N;  // f16 elems/tile
constexpr unsigned X_BYTES = FLAT * sizeof(half);
constexpr unsigned aln(unsigned b) { return (b + 511u) & ~511u; }
#ifndef NBUF
#define NBUF 4  // pipeline depth (buffers)
#endif
#ifndef PREFETCH
#define PREFETCH 2  // tiles to prefetch ahead
#endif
#define XOFF(i) ((unsigned)(i) * ((X_BYTES + 511u) & ~511u))
// A5 UB is 248 KB physical; the 192 KB default is simply what NBUF=4 needs at
// these tile sizes. Neither is the bottleneck: the transform is HBM-bound, and
// a measured NBUF=6 build (192 KB, exactly) is ~1% *slower* than NBUF=4
// (2634 vs 2668 GB/s at batch 65536, ROWS_PER_TILE=64), so extra pipeline depth
// buys nothing. Raise UB_USABLE_BYTES only if a deeper build is wanted.
#ifndef UB_USABLE_BYTES
#define UB_USABLE_BYTES (192u * 1024u)
#endif
static_assert(NBUF * aln(X_BYTES) <= UB_USABLE_BYTES, "UB overflow");
// PREFETCH == NBUF would make the prologue consume every MTE3->MTE2 token, so
// the first ISSUE_LOAD blocks on a flag only its own iteration's store can set.
static_assert(PREFETCH < NBUF,
              "PREFETCH must be < NBUF (else the pipeline deadlocks)");

#ifdef __DAV_VEC__
#define DOU(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
// Deinterleave-LOAD 256-point WHT: even/odd split on the MTE load (vlds
// DINTLV_B16), concat-halves recombine on the store (vsts to
// [0:128]/[128:256]), only vadd/vsub on the vector-execute pipe -> memory-bound
// (~copy speed).
__tf__ static AICORE void bfly256(__ubuf__ half *wb) {
  __VEC_SCOPE__ {
    uint32_t la = VL;
    MaskReg pAll = CreatePredicate<half>(la);
    vector_f16 e0, e1, e2, e3, e4, e5, e6, e7, o0, o1, o2, o3, o4, o5, o6, o7;
    vector_f16 s0, s1, s2, s3, s4, s5, s6, s7, d0, d1, d2, d3, d4, d5, d6, d7;
    // Slot i covers (group, chunk) = (i / CHUNKS, i % CHUNKS), both
    // compile-time constants, so every DOU slot is a fixed address. This puts
    // ALL loads ahead of ALL stores, which is required: a stage's sums compact
    // into the lower half of the group, exactly where a lower-numbered chunk
    // still has to read from, so a per-chunk load/store loop aliases in place
    // -- and it *passes* at HAD_GRP=512 while the stores have not landed yet,
    // then corrupts silently at larger sizes. Spanning the unroll removes the
    // hazard by construction rather than by timing.
#define GRP_OF(i) (((uint32_t)(i) / CHUNKS) * GRP)
#define CH_OF(i) ((uint32_t)(i) % CHUNKS)
#define LD(i) \
  vlds(e##i, o##i, wb + base + GRP_OF(i) + CH_OF(i) * 2u * VL, 0, DINTLV_B16);
#define AD(i) vadd(s##i, e##i, o##i, pAll);
#define SU(i) vsub(d##i, e##i, o##i, pAll);
#define ST_PAIR(i, LO, HI)                                               \
  vsts(LO##i, wb + base + GRP_OF(i) + CH_OF(i) * VL, 0, NORM_B16, pAll); \
  vsts(HI##i, wb + base + GRP_OF(i) + LANES + CH_OF(i) * VL, 0, NORM_B16, pAll);
#define ST(i) ST_PAIR(i, s, d)
#define NOROT(i)  // no rotation on the unpeeled stages
    // The tail: HAD_ROT rotate-right-by-1 of the 256-element (s,d) window,
    // ping-ponging into the (e,o) pair, which is dead after AD/SU. Costs no
    // extra registers, no UB traffic and no barrier -- and being register-only
    // it adds no aliasing hazard, unlike a rotation pass through UB, which
    // would have to re-read the window it just wrote.
#if HAD_ROT == 1
#define ROT(i) vdintlv(e##i, o##i, s##i, d##i);
#define STF(i) ST_PAIR(i, e, o)
#elif HAD_ROT == 2
#define ROT(i)                     \
  vdintlv(e##i, o##i, s##i, d##i); \
  vdintlv(s##i, d##i, e##i, o##i);
#define STF(i) ST_PAIR(i, s, d)
#elif HAD_ROT == 3
#define ROT(i)                     \
  vdintlv(e##i, o##i, s##i, d##i); \
  vdintlv(s##i, d##i, e##i, o##i); \
  vdintlv(e##i, o##i, s##i, d##i);
#define STF(i) ST_PAIR(i, e, o)
#endif
    // One sweep over the tile. The last stage is PEELED rather than selected by
    // a runtime branch: the loop analyser is fussy about anything non-literal
    // in the loop shape, and a branch inside __VEC_SCOPE__ risks worse than a
    // warning.
#define PASS(ROT_, ST_)                                                \
  for (uint16_t g = 0; g < (uint16_t)GROUPS; g += HAD_GRPS_PER_ITER) { \
    const uint32_t base = (uint32_t)g * GRP;                           \
    DOU(LD) DOU(AD) DOU(SU) DOU(ROT_) DOU(ST_)                         \
  }                                                                    \
  mem_bar(VST_VLD);
    for (uint16_t st = 0; st < (uint16_t)(HAD_LOG2N - HAD_PEEL); ++st) {
      PASS(NOROT, ST)
    }
#if HAD_ROT
    PASS(ROT, STF)  // final stage, with the rotation tail fused in
#endif
#undef PASS
#undef NOROT
#undef ST
#undef ST_PAIR
#undef SU
#undef AD
#undef LD
#undef CH_OF
#undef GRP_OF
#ifdef ROT
#undef ROT
#undef STF
#endif
  }
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

__global__ AICORE void hadamard256(__gm__ void *x_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  set_mask_norm();
  set_vector_mask(-1, -1);
  using Sh = pto::Shape<1, 1, 1, 1, FLAT>;
  using St = pto::Stride<1, 1, 1, FLAT, 1>;
  using T = Tile<TileType::Vec, half, 1, FLAT, BLayout::RowMajor, 1, FLAT>;
  const event_t ev[8] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3,
                         EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7};
  const uint32_t cid = get_block_idx(), nc = get_block_num();
  const uint32_t tiles = batch / ROWS_PER_TILE;

  // issue the async load for this core's K-th tile into buffer K%NBUF
#define ISSUE_LOAD(K)                                        \
  do {                                                       \
    uint32_t _tb = cid + (uint32_t)(K) * nc;                 \
    if (_tb < tiles) {                                       \
      const int _pp = (uint32_t)(K) % NBUF;                  \
      wait_flag(PIPE_MTE3, PIPE_MTE2, ev[_pp]);              \
      T _xt;                                                 \
      TASSIGN(_xt, XOFF(_pp));                               \
      GlobalTensor<half, Sh, St> _g(                         \
          (__gm__ half *)x_gm + (uint64_t)_tb * FLAT, Sh()); \
      TLOAD(_xt, _g);                                        \
      set_flag(PIPE_MTE2, PIPE_V, ev[_pp]);                  \
    }                                                        \
  } while (0)

  for (int i = 0; i < NBUF; ++i)
    set_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);  // all free
  for (uint32_t kk = 0; kk < (uint32_t)PREFETCH; ++kk)
    ISSUE_LOAD(kk);  // prologue

  uint32_t k = 0;
  for (uint32_t tb = cid; tb < tiles; tb += nc, ++k) {
    const int pp = k % NBUF;
    ISSUE_LOAD(k + PREFETCH);              // prefetch (overlaps this compute)
    wait_flag(PIPE_MTE2, PIPE_V, ev[pp]);  // this tile's load done
    bfly256((__ubuf__ half *)(uintptr_t)XOFF(pp));
    set_flag(PIPE_V, PIPE_MTE3, ev[pp]);
    wait_flag(PIPE_V, PIPE_MTE3, ev[pp]);
    T xt;
    TASSIGN(xt, XOFF(pp));
    GlobalTensor<half, Sh, St> g((__gm__ half *)x_gm + (uint64_t)tb * FLAT,
                                 Sh());
    TSTORE(g, xt);
    set_flag(PIPE_MTE3, PIPE_MTE2, ev[pp]);  // buffer free
  }
  for (int i = 0; i < NBUF; ++i) wait_flag(PIPE_MTE3, PIPE_MTE2, ev[i]);
#else
  (void)x_gm;
  (void)batch;
#endif
}

extern "C" void call_hadamard256(uint32_t bd, void *s, uint8_t *x, uint32_t b) {
  hadamard256<<<bd, nullptr, s>>>(x, b);
}
