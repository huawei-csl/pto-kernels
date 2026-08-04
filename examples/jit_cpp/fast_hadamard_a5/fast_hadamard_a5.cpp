// fast_hadamard_a5 — Walsh-Hadamard transform on the Ascend A5 (dav-c310).
//
// For each row of a (batch, N) fp16 matrix, y = x * H in place, where H is the
// +/-1 Hadamard matrix of order N. Unnormalized: scale by 1/sqrt(N). N is
// 32..2048, a power of two.
//
// Memory-bound: each row is read and written once, so the design spends as
// little as possible on the vector pipe. One stage turns adjacent pairs (a, b)
// into (a+b, a-b); the load splits even/odd lanes for free and the store's two
// halves concatenate into the next stage's input, leaving only vadd/vsub.
//
// Layout: KernelShape holds every derived size, sweep does one stage over SLOTS
// groups, butterfly does log2(N) stages over a UB tile, hadamard runs the
// GM<->UB pipeline, and call_hadamard dispatches on n so one .so serves every
// N.
//
// README.md has the rationale, the measured numbers, and the three
// non-obvious constraints this file is shaped around.
#include <pto/pto-inst.hpp>
#include <utility>
using namespace pto;

// Block sizes with an instantiation. Adding one here is the only edit needed.
constexpr unsigned SUPPORTED_N[] = {32, 64, 128, 256, 512, 1024, 2048};
constexpr unsigned SUPPORTED_COUNT =
    sizeof(SUPPORTED_N) / sizeof(SUPPORTED_N[0]);
constexpr unsigned DEFAULT_N = 256;   // used when a caller does not choose
constexpr unsigned DEF_BUFFERS = 4;   // UB buffers in the GM<->UB pipeline
constexpr unsigned DEF_PREFETCH = 2;  // tiles in flight ahead of the compute

// Rows per GM<->UB tile: a 16 KB tile, floored at 4 rows. Must agree with
// rows_for() in jit_util_a5.py, which needs it before any .so exists;
// test_rows_for_matches_kernel pins the two together. 2u rather than
// sizeof(half) because this is read in the HOST pass, where half does not
// exist. README "Tiling" has the sweep behind 16 KB.
constexpr unsigned TILE_BYTES = 16u * 1024u;
template <unsigned N>
struct RowsFor {
  static constexpr unsigned quotient = TILE_BYTES / (N * 2u);
  static constexpr unsigned value = quotient > 4u ? quotient : 4u;
};

#ifdef __CCE_AICORE__
constexpr unsigned F16_LANES = sizeof(vector_f16) / sizeof(half);
constexpr unsigned WINDOW = 2 * F16_LANES;  // butterfly window: two registers
constexpr unsigned SLOTS = 8;        // unroll width: register sets per sweep
constexpr unsigned EVENT_SLOTS = 8;  // size of the event-id array
// hadamard() writes eight EVENT_IDs into buffer_free by hand. A larger array
// zero-fills, aliasing every extra buffer onto EVENT_ID0 (which is 0), and the
// NBuffers <= EVENT_SLOTS assert below would still pass.
static_assert(EVENT_SLOTS == 8, "extend buffer_free's initialiser first");
constexpr unsigned UB_ALIGN = 512;
// 256 KB on A5, 192 on A2/A3, 128 on Kirin. Device pass only.
constexpr unsigned UB_BYTES = PTO_UBUF_SIZE_BYTES;

// A constexpr *function* cannot be called from [aicore] code; a value template
// can.
template <unsigned Value>
struct Log2 {
  static constexpr unsigned value = 1 + Log2<(Value >> 1)>::value;
};
template <>
struct Log2<1> {
  static constexpr unsigned value = 0;
};

// Every derived size for one instantiation; a "group" is one unroll slot's work
// and is always one WINDOW wide. N < WINDOW packs several rows per window,
// N > WINDOW splits a row into chunks; see README "Block size".
template <unsigned N, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
struct KernelShape {
  static constexpr unsigned log2_n = Log2<N>::value;
  static constexpr unsigned log2_window = Log2<WINDOW>::value;
  static constexpr unsigned rows_per_window = N < WINDOW ? WINDOW / N : 1;
  static constexpr unsigned group =
      N * rows_per_window;  // == WINDOW if N <= WINDOW
  static constexpr unsigned rotations =
      log2_window - (log2_n < log2_window ? log2_n : log2_window);
  static constexpr unsigned upper = group / 2;  // half a WINDOW, not half a row
  static constexpr unsigned lanes = upper < F16_LANES ? upper : F16_LANES;
  static constexpr unsigned chunks = upper / lanes;  // 1 for N <= WINDOW
  static constexpr unsigned groups_per_iter = SLOTS / chunks;
  static constexpr unsigned iters = Rows / rows_per_window / groups_per_iter;
  static constexpr unsigned sweep_stride = groups_per_iter * group;
  static constexpr unsigned tile_elems = Rows * N;
  static constexpr unsigned tile_stride =
      (tile_elems * sizeof(half) + UB_ALIGN - 1) & ~(UB_ALIGN - 1);

  static_assert(N >= 32 && N <= 2048 && !(N & (N - 1)), "N: pow2 in [32,2048]");
  static_assert(sizeof(half) == 2, "RowsFor assumes 2-byte elements");
  static_assert(chunks <= SLOTS && SLOTS % chunks == 0, "N too wide to unroll");
  static_assert(upper % lanes == 0, "half a group must be whole vectors");
  static_assert(rows_per_window == 1 || chunks == 1,
                "pack and chunk are exclusive");
  static_assert(Rows % rows_per_window == 0,
                "Rows must be a multiple of WINDOW/N");
  static_assert(Rows / rows_per_window % groups_per_iter == 0,
                "groups per tile must divide by groups_per_iter");
  static_assert(NBuffers * tile_stride <= UB_BYTES, "UB overflow");
  static_assert(NBuffers <= EVENT_SLOTS, "NBUF exceeds the event-id array");
  static_assert(NPrefetch < NBuffers,
                "PREFETCH == NBUF drains every MTE3->MTE2 token: deadlock");
};

#ifdef __DAV_VEC__
// A flat run of Elems fp16 in GM, and the matching UB tile.
template <unsigned Elems>
using GmShape = pto::Shape<1, 1, 1, 1, Elems>;
template <unsigned Elems>
using GmStride = pto::Stride<1, 1, 1, Elems, 1>;
template <unsigned Elems>
using UbTile = Tile<TileType::Vec, half, 1, Elems, BLayout::RowMajor, 1, Elems>;

using RegSet = vector_f16[SLOTS];

// One unrolled sweep: SLOTS groups loaded, combined, stored back. Slot i covers
// (group, chunk) = (i / chunks, i % chunks), which keeps every load ahead of
// every store -- required, not stylistic. Rotations > 0 fuses the packing tail
// in. Pack indices rather than a loop variable, and why the ordering matters:
// see README "Implementation notes".
template <typename Shape, unsigned Rotations, std::size_t... Slot>
inline AICORE void sweep(__ubuf__ half *tile, uint32_t base, MaskReg all,
                         RegSet &even, RegSet &odd, RegSet &sum, RegSet &diff,
                         std::index_sequence<Slot...>) {
  constexpr unsigned g = Shape::group, up = Shape::upper;
  constexpr unsigned ln = Shape::lanes, ch = Shape::chunks;
  (vlds(even[Slot], odd[Slot],
        tile + base + Slot / ch * g + Slot % ch * 2u * ln, 0, DINTLV_B16),
   ...);
  (vadd(sum[Slot], even[Slot], odd[Slot], all), ...);
  (vsub(diff[Slot], even[Slot], odd[Slot], all), ...);
  if constexpr (Rotations >= 1) {
    (vdintlv(even[Slot], odd[Slot], sum[Slot], diff[Slot]), ...);
  }
  if constexpr (Rotations >= 2) {
    (vdintlv(sum[Slot], diff[Slot], even[Slot], odd[Slot]), ...);
  }
  if constexpr (Rotations >= 3) {
    (vdintlv(even[Slot], odd[Slot], sum[Slot], diff[Slot]), ...);
  }
  RegSet &lo = (Rotations % 2 == 1) ? even : sum;
  RegSet &hi = (Rotations % 2 == 1) ? odd : diff;
  (vsts(lo[Slot], tile + base + Slot / ch * g + Slot % ch * ln, 0, NORM_B16,
        all),
   ...);
  (vsts(hi[Slot], tile + base + Slot / ch * g + up + Slot % ch * ln, 0,
        NORM_B16, all),
   ...);
}

// log2(N) stages over one UB tile, in place.
template <typename Shape>
__tf__ static AICORE void butterfly(__ubuf__ half *tile) {
  constexpr auto slots = std::make_index_sequence<SLOTS>{};
  constexpr unsigned plain = Shape::log2_n - (Shape::rotations ? 1 : 0);
  __VEC_SCOPE__ {
    uint32_t lane_count = Shape::lanes;
    MaskReg all = CreatePredicate<half>(lane_count);
    vector_f16 even[SLOTS], odd[SLOTS], sum[SLOTS], diff[SLOTS];
    // Stepping by a literal 1 with the stride folded into `base` is
    // load-bearing: the loop analyser verifies a tripcount only for a literal
    // step, and 1 divides any bound, so `iters` may be template-dependent.
    for (uint16_t stage = 0; stage < (uint16_t)plain; ++stage) {
      for (uint16_t iter = 0; iter < (uint16_t)Shape::iters; ++iter)
        sweep<Shape, 0>(tile, (uint32_t)iter * Shape::sweep_stride, all, even,
                        odd, sum, diff, slots);
      mem_bar(VST_VLD);
    }
    // The last stage is peeled rather than branched, because a runtime branch
    // inside __VEC_SCOPE__ is riskier than a second call.
    if constexpr (Shape::rotations > 0) {
      for (uint16_t iter = 0; iter < (uint16_t)Shape::iters; ++iter)
        sweep<Shape, Shape::rotations>(tile,
                                       (uint32_t)iter * Shape::sweep_stride,
                                       all, even, odd, sum, diff, slots);
      mem_bar(VST_VLD);
    }
  }
}

// Move one tile between GM and UB buffer `buf`. Both directions share the view
// setup; only the final TLOAD/TSTORE differs.
template <typename Shape, bool ToUb>
inline AICORE void transfer(uint32_t tile_index, uint32_t buf,
                            __gm__ void *x_gm) {
  constexpr unsigned elems = Shape::tile_elems;
  UbTile<elems> ub;
  TASSIGN(ub, buf * Shape::tile_stride);
  GlobalTensor<half, GmShape<elems>, GmStride<elems>> gm(
      (__gm__ half *)x_gm + (uint64_t)tile_index * elems, GmShape<elems>());
  if constexpr (ToUb) {
    TLOAD(ub, gm);
  } else {
    TSTORE(gm, ub);
  }
}

// Start the async load of this core's nth tile, if it has one. A function, not
// a lambda: set_flag/wait_flag do not resolve inside a lambda.
template <typename Shape, unsigned Buffers>
inline AICORE void issue_load(uint32_t nth, uint32_t core_id,
                              uint32_t core_count, uint32_t tiles,
                              const event_t *buffer_free, __gm__ void *x_gm) {
  const uint32_t tile_index = core_id + nth * core_count;
  if (tile_index >= tiles) return;
  const uint32_t buf = nth % Buffers;
  wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buf]);
  transfer<Shape, true>(tile_index, buf, x_gm);
  set_flag(PIPE_MTE2, PIPE_V, buffer_free[buf]);
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

// The pipeline: each core walks a strided subset of the tiles, keeping Prefetch
// loads in flight so DMA and the vector pipe overlap.
template <unsigned N, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
__global__ AICORE void hadamard(__gm__ void *x_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  using Shape = KernelShape<N, Rows, NBuffers, NPrefetch>;
  set_mask_norm();
  set_vector_mask(-1, -1);
  const event_t buffer_free[EVENT_SLOTS] = {EVENT_ID0, EVENT_ID1, EVENT_ID2,
                                            EVENT_ID3, EVENT_ID4, EVENT_ID5,
                                            EVENT_ID6, EVENT_ID7};
  const uint32_t core_id = get_block_idx(), core_count = get_block_num();
  const uint32_t tiles = batch / Rows;

  for (unsigned i = 0; i < NBuffers; ++i)  // every buffer starts free
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
  for (unsigned i = 0; i < NPrefetch; ++i)
    issue_load<Shape, NBuffers>(i, core_id, core_count, tiles, buffer_free,
                                x_gm);

  uint32_t issued = 0;
  for (uint32_t tile_index = core_id; tile_index < tiles;
       tile_index += core_count, ++issued) {
    const uint32_t buf = issued % NBuffers;
    // issued ahead of the wait below, so this load overlaps this tile's compute
    issue_load<Shape, NBuffers>(issued + NPrefetch, core_id, core_count, tiles,
                                buffer_free, x_gm);
    wait_flag(PIPE_MTE2, PIPE_V, buffer_free[buf]);
    butterfly<Shape>((__ubuf__ half *)(uintptr_t)(buf * Shape::tile_stride));
    set_flag(PIPE_V, PIPE_MTE3, buffer_free[buf]);
    wait_flag(PIPE_V, PIPE_MTE3, buffer_free[buf]);
    transfer<Shape, false>(tile_index, buf, x_gm);
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buf]);
  }
  for (unsigned i = 0; i < NBuffers; ++i)  // drain
    wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
#else
  (void)x_gm;
  (void)batch;
#endif
}

// ---------------------------------------------------------------- entry points
// One .so serves every size. These fold over SUPPORTED_N to find the matching
// instantiation, which keeps the list of sizes in one place and needs no
// case-label macros.
template <std::size_t... Idx>
inline void launch_for_n(uint32_t bd, void *stream, uint8_t *x, uint32_t batch,
                         uint32_t n, std::index_sequence<Idx...>) {
  ((n == SUPPORTED_N[Idx]
        ? (void)(hadamard<SUPPORTED_N[Idx], RowsFor<SUPPORTED_N[Idx]>::value,
                          DEF_BUFFERS, DEF_PREFETCH>
                 <<<bd, nullptr, stream>>>(x, batch))
        : (void)0),
   ...);
}

// An unsupported n does nothing at all -- it cannot report from here, so the
// host validates before calling (jit_util_a5.check_n).
extern "C" void call_hadamard(uint32_t bd, void *stream, uint8_t *x,
                              uint32_t batch, uint32_t n) {
  launch_for_n(bd, stream, x, batch, n,
               std::make_index_sequence<SUPPORTED_COUNT>{});
}

// Default shape, N = DEFAULT_N, for callers that do not choose.
extern "C" void call_hadamard_default(uint32_t bd, void *stream, uint8_t *x,
                                      uint32_t batch) {
  call_hadamard(bd, stream, x, batch, DEFAULT_N);
}

template <std::size_t... Idx>
inline uint32_t rows_for_n(uint32_t n, std::index_sequence<Idx...>) {
  uint32_t rows = 0;
  ((n == SUPPORTED_N[Idx] ? (void)(rows = RowsFor<SUPPORTED_N[Idx]>::value)
                          : (void)0),
   ...);
  return rows;  // 0 for an unsupported n; the host raises on that
}

// So the host does not have to restate the tiling rule.
extern "C" uint32_t hadamard_rows_for(uint32_t n) {
  return rows_for_n(n, std::make_index_sequence<SUPPORTED_COUNT>{});
}

// Tuning entry point for the benchmark's (ROWS x NBUF) sweep, which varies
// shape beyond what the dispatcher fixes. Deliberately has NO defaults: a
// tuning build must pass all four -D or it fails to compile, rather than
// silently measuring some other shape.
#ifdef HAD_TUNE
extern "C" void call_hadamard_tuned(uint32_t bd, void *stream, uint8_t *x,
                                    uint32_t batch) {
  hadamard<HAD_N, ROWS_PER_TILE, NBUF, PREFETCH>
      <<<bd, nullptr, stream>>>(x, batch);
}
#endif
