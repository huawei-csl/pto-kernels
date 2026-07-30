// fast_hadamard_256_a5 — Walsh-Hadamard by deinterleave-load butterfly.
// The even/odd split rides the MTE2 load (vlds DINTLV_B16) and the concat
// of halves rides the MTE3 store, so only vadd/vsub reach the vector pipe.
// In-place on a UB tile. N = 32..2048, each within 0.90..0.96 of its own DMA
// copy floor; that floor is measured by copy_ref_256_a5.cpp, built separately.
#include <pto/pto-inst.hpp>
#include <utility>
using namespace pto;

// Supported block sizes. One instantiation each; call_hadamard dispatches on n.
#define HAD_SIZES(F) F(32) F(64) F(128) F(256) F(512) F(1024) F(2048)

// Default pipeline shape. ROWS is derived per N below; these two are flat.
constexpr unsigned DEF_BUFFERS = 4;  // measured: 6 is ~1% slower, not faster
constexpr unsigned DEF_PREFETCH = 2;

// A 32 KB tile, floored at 8 rows. Must agree with rows_for() in
// jit_util_a5.py, which the padding wrapper needs before any .so exists; a test
// pins that. 2u, not sizeof(half): this is read in the host pass, where half is
// absent.
constexpr unsigned TILE_BYTES = 32u * 1024u;
template <unsigned N>
struct RowsFor {
  static constexpr unsigned quotient = TILE_BYTES / (N * 2u);
  static constexpr unsigned value = quotient > 8u ? quotient : 8u;
};

#ifdef __CCE_AICORE__
constexpr unsigned F16_LANES = sizeof(vector_f16) / sizeof(half);
constexpr unsigned WINDOW = 2 * F16_LANES;  // butterfly window: two registers
constexpr unsigned SLOTS = 8;        // unroll width: register sets per sweep
constexpr unsigned EVENT_SLOTS = 8;  // size of buffer_free[] below
constexpr unsigned UB_ALIGN = 512;
// 256 KB on A5, 192 on A2/A3, 128 on Kirin. Defined in the device pass only.
constexpr unsigned UB_BYTES = PTO_UBUF_SIZE_BYTES;

// A constexpr *function* cannot be called from [aicore] code; a value
// template can.
template <unsigned Value>
struct Log2 {
  static constexpr unsigned value = 1 + Log2<(Value >> 1)>::value;
};
template <>
struct Log2<1> {
  static constexpr unsigned value = 0;
};

// Shape of one instantiation. A "group" is the unit one unroll slot handles.
//
// PACK (N < WINDOW): rows_per_window rows share a window. The split is on the
// low bit of the within-row index, so rows never mix; the result emerges
// rotated right by log2(N), which `rotations` vdintlv rounds undo.
//
// CHUNK (N > WINDOW): stages pair adjacent elements and their output halves
// concatenate, so a row splits into independent WINDOW-sized chunks.
//
// The two are mutually exclusive, asserted below.
template <unsigned N, unsigned Rows, unsigned Buffers, unsigned Prefetch>
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
  static_assert(Rows == RowsFor<N>::value || Rows != 0,
                "Rows must be positive");
  static_assert(chunks <= SLOTS && SLOTS % chunks == 0, "N too wide to unroll");
  static_assert(upper % lanes == 0, "half a group must be whole vectors");
  static_assert(rows_per_window == 1 || chunks == 1,
                "pack and chunk are exclusive");
  static_assert(Rows % rows_per_window == 0,
                "Rows must be a multiple of WINDOW/N");
  static_assert(Rows / rows_per_window % groups_per_iter == 0,
                "groups per tile must divide by groups_per_iter");
  static_assert(Buffers * tile_stride <= UB_BYTES, "UB overflow");
  static_assert(Buffers <= EVENT_SLOTS, "NBUF exceeds the event-id array");
  static_assert(Prefetch < Buffers,
                "PREFETCH == NBUF drains every MTE3->MTE2 token: deadlock");
};

#ifdef __DAV_VEC__
using RegSet = vector_f16[SLOTS];

// One unrolled sweep: SLOTS groups loaded, combined and stored back.
//
// Slot i covers (group, chunk) = (i / chunks, i % chunks), so all loads precede
// all stores. Required, not stylistic: a stage's sums compact into the group's
// lower half, which a lower-numbered chunk still has to read, so a per-chunk
// load/store loop aliases in place -- and it passes at group=512 before
// silently corrupting at larger sizes. The comma fold evaluates left to right,
// which is what holds that ordering.
//
// Slot indices come from an index_sequence pack rather than a loop: a vector
// register array indexed by a `#pragma unroll` loop variable crashes the
// backend
// ("Unsupported Inst must be hoisted"), while a pack index is a true
// compile-time constant. That is what lets this be ordinary code instead of
// token-pasting macros.
//
// Rotations > 0 fuses the packing tail into this sweep: each vdintlv rotates
// the window right by one and ping-pongs into the pair vadd/vsub just freed, so
// it costs no extra registers, no UB traffic, no barrier, and cannot alias --
// which a rotation through UB would. Device-verified for 1, 2 and 3 rounds
// against ror(index, k). An odd count leaves the result in (even, odd).
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

template <unsigned N, unsigned Rows, unsigned Buffers, unsigned Prefetch>
__tf__ static AICORE void butterfly(__ubuf__ half *tile) {
  using Shape = KernelShape<N, Rows, Buffers, Prefetch>;
  constexpr auto slots = std::make_index_sequence<SLOTS>{};
  __VEC_SCOPE__ {
    uint32_t lane_count = Shape::lanes;
    MaskReg all = CreatePredicate<half>(lane_count);
    vector_f16 even[SLOTS], odd[SLOTS], sum[SLOTS], diff[SLOTS];
    // Stepping by a literal 1 with the stride folded into `base` is
    // load-bearing: the loop analyser verifies a tripcount only for a literal
    // step, and 1 divides any bound, so `iters` may be template-dependent
    // without warning.
    constexpr unsigned plain = Shape::log2_n - (Shape::rotations ? 1 : 0);
    for (uint16_t stage = 0; stage < (uint16_t)plain; ++stage) {
      for (uint16_t iter = 0; iter < (uint16_t)Shape::iters; ++iter)
        sweep<Shape, 0>(tile, (uint32_t)iter * Shape::sweep_stride, all, even,
                        odd, sum, diff, slots);
      mem_bar(VST_VLD);
    }
    // Last stage peeled rather than branched: a runtime branch in __VEC_SCOPE__
    // is riskier than a duplicated call.
    if constexpr (Shape::rotations > 0) {
      for (uint16_t iter = 0; iter < (uint16_t)Shape::iters; ++iter)
        sweep<Shape, Shape::rotations>(tile,
                                       (uint32_t)iter * Shape::sweep_stride,
                                       all, even, odd, sum, diff, slots);
      mem_bar(VST_VLD);
    }
  }
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

template <unsigned N, unsigned Rows, unsigned Buffers, unsigned Prefetch>
__global__ AICORE void hadamard(__gm__ void *x_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  using Shape = KernelShape<N, Rows, Buffers, Prefetch>;
  constexpr unsigned elems = Shape::tile_elems, stride = Shape::tile_stride;
  set_mask_norm();
  set_vector_mask(-1, -1);
  using GmShape = pto::Shape<1, 1, 1, 1, elems>;
  using GmStride = pto::Stride<1, 1, 1, elems, 1>;
  using UbTile =
      Tile<TileType::Vec, half, 1, elems, BLayout::RowMajor, 1, elems>;
  const event_t buffer_free[EVENT_SLOTS] = {EVENT_ID0, EVENT_ID1, EVENT_ID2,
                                            EVENT_ID3, EVENT_ID4, EVENT_ID5,
                                            EVENT_ID6, EVENT_ID7};
  const uint32_t core_id = get_block_idx(), core_count = get_block_num();
  const uint32_t tiles = batch / Rows;

  // Async load of this core's issue-th tile into buffer issue % Buffers. A
  // macro, not a lambda: the pipeline flag intrinsics must sit directly in the
  // kernel body (from a lambda, set_flag/wait_flag do not resolve).
#define ISSUE_LOAD(issue)                                         \
  do {                                                            \
    const uint32_t _t = core_id + (uint32_t)(issue) * core_count; \
    if (_t < tiles) {                                             \
      const uint32_t _b = (uint32_t)(issue) % Buffers;            \
      wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[_b]);           \
      UbTile _ub;                                                 \
      TASSIGN(_ub, _b *stride);                                   \
      GlobalTensor<half, GmShape, GmStride> _gm(                  \
          (__gm__ half *)x_gm + (uint64_t)_t * elems, GmShape()); \
      TLOAD(_ub, _gm);                                            \
      set_flag(PIPE_MTE2, PIPE_V, buffer_free[_b]);               \
    }                                                             \
  } while (0)

  for (unsigned i = 0; i < Buffers; ++i)
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
  for (unsigned i = 0; i < Prefetch; ++i) ISSUE_LOAD(i);

  uint32_t issued = 0;
  for (uint32_t tile_index = core_id; tile_index < tiles;
       tile_index += core_count, ++issued) {
    const uint32_t buf = issued % Buffers;
    ISSUE_LOAD(issued + Prefetch);  // overlaps this tile's compute
    wait_flag(PIPE_MTE2, PIPE_V, buffer_free[buf]);
    butterfly<N, Rows, Buffers, Prefetch>(
        (__ubuf__ half *)(uintptr_t)(buf * stride));
    set_flag(PIPE_V, PIPE_MTE3, buffer_free[buf]);
    wait_flag(PIPE_V, PIPE_MTE3, buffer_free[buf]);
    UbTile ub;
    TASSIGN(ub, buf * stride);
    GlobalTensor<half, GmShape, GmStride> gm(
        (__gm__ half *)x_gm + (uint64_t)tile_index * elems, GmShape());
    TSTORE(gm, ub);
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buf]);
  }
  for (unsigned i = 0; i < Buffers; ++i)
    wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
#undef ISSUE_LOAD
#else
  (void)x_gm;
  (void)batch;
#endif
}

// One .so serves every size: the caller passes n and this picks the
// instantiation, so no -D and no per-N rebuild.
extern "C" void call_hadamard(uint32_t bd, void *stream, uint8_t *x,
                              uint32_t batch, uint32_t n) {
#define LAUNCH(SZ)                                              \
  case SZ:                                                      \
    hadamard<SZ, RowsFor<SZ>::value, DEF_BUFFERS, DEF_PREFETCH> \
        <<<bd, nullptr, stream>>>(x, batch);                    \
    return;
  switch (n) {
    HAD_SIZES(LAUNCH)
    default:
      return;  // unsupported n; the caller validates before getting here
  }
#undef LAUNCH
}

// So the host does not have to restate the tiling rule.
extern "C" uint32_t hadamard_rows_for(uint32_t n) {
#define ROWS_CASE(SZ) \
  case SZ:            \
    return RowsFor<SZ>::value;
  switch (n) {
    HAD_SIZES(ROWS_CASE)
    default:
      return 0;
  }
#undef ROWS_CASE
}

// Tuning entry point for the benchmark's (ROWS x NBUF) sweep, which varies
// shape beyond what the dispatcher fixes. Deliberately has NO defaults: a
// tuning build must pass all four -D, or it fails to compile rather than
// silently measuring 256/64/4/2.
#ifdef HAD_TUNE
extern "C" void call_hadamard_tuned(uint32_t bd, void *stream, uint8_t *x,
                                    uint32_t batch) {
  hadamard<HAD_N, ROWS_PER_TILE, NBUF, PREFETCH>
      <<<bd, nullptr, stream>>>(x, batch);
}
#endif
