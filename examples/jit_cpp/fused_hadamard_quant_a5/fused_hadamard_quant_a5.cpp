// Full-row Hadamard fused with MXFP4 quantization, one launch.
//
//   x  ->  (x @ H_K) -> E2M1 nibbles + one E8M0 scale per 32
//
// The rotation is order K -- the whole row -- so K must be a power of two.
// Sylvester factors as H_K = H_(K/256) (x) H_256, so a row is rotated as K/256
// register-local windows (phase 1) and then the cross-window stages that
// finish the transform (phase 2). Neither phase holds more than one
// 256-element window in registers, so width is not capped by register
// pressure: SUPPORTED_K runs 32 to 16384.
//
// fused_hadamard_quant_b32_a5 next to it rotates independent 32-blocks
// instead; there a scale covers exactly one rotated block, where a row-wide
// rotation spreads an outlier over every block's shared scale. See the README
// and fused_hadamard_quant_common.hpp for what the two have in common.
#include "fused_hadamard_quant_common.hpp"

// Row widths with an instantiation. The full set the quantizer supports: a
constexpr unsigned SUPPORTED_K[] = {32,   64,   128,  256,  512,
                                    1024, 2048, 4096, 8192, 16384};
constexpr unsigned SUPPORTED_COUNT =
    sizeof(SUPPORTED_K) / sizeof(SUPPORTED_K[0]);

#ifdef __CCE_AICORE__

#ifdef __DAV_VEC__

// --- phase 2: the cross-window stages ---------------------------------------
//
// After phase 1 every 256-element window holds its own order-256 transform, and
// H_K = H_(K/256) (x) H_256 leaves log2(K/256) stages that pair windows
// elementwise. Measured, these stages were 69-73% of the kernel at K=8192 and
// 16384, against phase 1 costing nothing at all -- phase 1 walks windows in
// order while a phase-2 stage reads windows 256*t apart, up to 16 KB at the
// last stage, and that stride is the cost. Op count is not the explanation:
// phase 1 does more loads and stores per element and is entirely hidden.
//
// So the stages are FUSED. A group of 2^R windows is loaded once, R stages run
// register to register, and the group is stored once -- R strided passes become
// one. FUSED_CROSS_FUSE=1 reproduces one stage per pass, which is what the A/B
// against this is.
#ifndef FUSED_CROSS_FUSE
#define FUSED_CROSS_FUSE 3
#endif
constexpr unsigned CROSS_FUSE = FUSED_CROSS_FUSE;
static_assert(
    (1u << CROSS_FUSE) <= SLOTS,
    "a fused group holds 2^FUSED_CROSS_FUSE windows, and each needs a "
    "register slot in both ping-pong arrays");

// One stage across the window axis, register to register. Reading from `src`
// and writing to `dst` is what removes the copy an in-place butterfly needs:
// window m's new value combines it with its partner m^bit, added when the bit
// is clear and subtracted the other way when it is set, so every window is one
// instruction and no temporary survives the stage.
template <unsigned Bit, std::size_t M>
inline AICORE void one_window(MaskReg all, HadRegs &s_lo, HadRegs &s_hi,
                              HadRegs &d_lo, HadRegs &d_hi) {
  constexpr std::size_t P = M ^ (1u << Bit);
  if constexpr ((M & (1u << Bit)) == 0) {
    vadd((vector_bf16 &)d_lo[M], (vector_bf16 &)s_lo[M], (vector_bf16 &)s_lo[P],
         all);
    vadd((vector_bf16 &)d_hi[M], (vector_bf16 &)s_hi[M], (vector_bf16 &)s_hi[P],
         all);
  } else {
    vsub((vector_bf16 &)d_lo[M], (vector_bf16 &)s_lo[P], (vector_bf16 &)s_lo[M],
         all);
    vsub((vector_bf16 &)d_hi[M], (vector_bf16 &)s_hi[P], (vector_bf16 &)s_hi[M],
         all);
  }
}

template <unsigned Bit, std::size_t... M>
inline AICORE void window_stage(MaskReg all, HadRegs &s_lo, HadRegs &s_hi,
                                HadRegs &d_lo, HadRegs &d_hi,
                                std::index_sequence<M...>) {
  (one_window<Bit, M>(all, s_lo, s_hi, d_lo, d_hi), ...);
}

// R stages over the same registers, alternating direction so neither array is
// read and written in the same stage.
template <unsigned R, unsigned I, std::size_t... M>
inline AICORE void window_stages(MaskReg all, HadRegs &a_lo, HadRegs &a_hi,
                                 HadRegs &b_lo, HadRegs &b_hi,
                                 std::index_sequence<M...> ms) {
  if constexpr (I < R) {
    if constexpr (I % 2u == 0u)
      window_stage<I>(all, a_lo, a_hi, b_lo, b_hi, ms);
    else
      window_stage<I>(all, b_lo, b_hi, a_lo, a_hi, ms);
    window_stages<R, I + 1u>(all, a_lo, a_hi, b_lo, b_hi, ms);
  }
}

// One group: 2^R windows in, R stages, 2^R windows out. Window m of the group
// sits at m * 2^S0 windows from the base, because m's bits map onto window bits
// S0..S0+R-1 and those are contiguous.
template <typename Shape, unsigned S0, unsigned R, std::size_t... M>
inline AICORE void cross_group(__ubuf__ uint16_t *tile, uint32_t base,
                               MaskReg all, HadRegs &a_lo, HadRegs &a_hi,
                               HadRegs &b_lo, HadRegs &b_hi,
                               std::index_sequence<M...> ms) {
  constexpr unsigned ln = B16_LANES, win = Shape::had_window;
  constexpr unsigned step = (1u << S0) * win;
  (vlds(a_lo[M], tile + base + M * step, 0, NORM), ...);
  (vlds(a_hi[M], tile + base + M * step + ln, 0, NORM), ...);
  window_stages<R, 0u>(all, a_lo, a_hi, b_lo, b_hi, ms);
  // R stages land back in a_* when R is even and in b_* when it is odd
  if constexpr (R % 2u == 0u) {
    (vsts(a_lo[M], tile + base + M * step, 0, NORM_B16, all), ...);
    (vsts(a_hi[M], tile + base + M * step + ln, 0, NORM_B16, all), ...);
  } else {
    (vsts(b_lo[M], tile + base + M * step, 0, NORM_B16, all), ...);
    (vsts(b_hi[M], tile + base + M * step + ln, 0, NORM_B16, all), ...);
  }
}

// Every group for one fused pass. A group's base has window bits S0..S0+R-1
// clear, so the bases are (hi << (S0+R)) | lo over the two remaining ranges,
// and rows never mix.
template <typename Shape, unsigned S0, unsigned R>
inline AICORE void cross_pass(__ubuf__ uint16_t *tile, MaskReg all,
                              HadRegs &a_lo, HadRegs &a_hi, HadRegs &b_lo,
                              HadRegs &b_hi) {
  constexpr unsigned win = Shape::had_window;
  constexpr unsigned nwin = Shape::windows_per_row;
  constexpr unsigned lo_span = 1u << S0;
  constexpr unsigned hi_step = 1u << (S0 + R);
  constexpr auto ms = std::make_index_sequence<(1u << R)>{};
  static_assert(nwin % hi_step == 0, "a fused group must fit the row");
  for (uint16_t row = 0; row < (uint16_t)Shape::rows_in_tile; ++row)
    for (uint16_t hi = 0; hi < (uint16_t)(nwin / hi_step); ++hi)
      for (uint16_t lo = 0; lo < (uint16_t)lo_span; ++lo)
        cross_group<Shape, S0, R>(
            tile,
            (uint32_t)row * Shape::row_elems +
                ((uint32_t)hi * hi_step + (uint32_t)lo) * win,
            all, a_lo, a_hi, b_lo, b_hi, ms);
}

// Walk the stages in fused groups, largest first, with whatever remains taking
// a narrower final group.
template <typename Shape, unsigned S0>
inline AICORE void cross_from(__ubuf__ uint16_t *tile, MaskReg all,
                              HadRegs &a_lo, HadRegs &a_hi, HadRegs &b_lo,
                              HadRegs &b_hi) {
  if constexpr (S0 < Shape::phase2_stages) {
    constexpr unsigned left = Shape::phase2_stages - S0;
    constexpr unsigned R = left < CROSS_FUSE ? left : CROSS_FUSE;
    cross_pass<Shape, S0, R>(tile, all, a_lo, a_hi, b_lo, b_hi);
    // the next group reads what this one wrote
    mem_bar(VST_VLD);
    cross_from<Shape, S0 + R>(tile, all, a_lo, a_hi, b_lo, b_hi);
  }
}

template <typename Shape>
__tf__ static AICORE void cross_windows(__ubuf__ uint16_t *tile) {
  __VEC_SCOPE__ {
    uint32_t lane_count = B16_LANES;
    MaskReg all = CreatePredicate<bfloat16_t>(lane_count);
    vector_u16 a_lo[SLOTS], a_hi[SLOTS], b_lo[SLOTS], b_hi[SLOTS];
    cross_from<Shape, 0u>(tile, all, a_lo, a_hi, b_lo, b_hi);
  }
}

#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

// The pipeline: each core walks a strided subset of the tiles, keeping Prefetch
// loads in flight so DMA and the vector pipe overlap.
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
// A device function rather than the kernel body, so a caller that wants the
// pipeline over a sub-range can reach it directly. The kernel below is the
// entry point and the only caller here.
template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
inline AICORE void quant_tiles(__gm__ void *input_gm, __gm__ void *nibble_gm,
                               __gm__ void *scale_gm, uint32_t batch) {
  // Order = K: one rotation spans the whole row.
  using Shape = QuantShape<K, K, Rows, NBuffers, NPrefetch>;
  // Full-row only: phase 2 pairs whole windows, so a row must be a whole
  // number of them. Block-32 has legal widths that are not (896, say).
  static_assert(
      Shape::windows_per_row * Shape::had_window == K || K < Shape::had_window,
      "a row must be a whole number of windows");
  using Offsets = SlotOffset<Shape>;
  set_mask_norm();
  set_vector_mask(-1, -1);
  const event_t buffer_free[EVENT_SLOTS] = {EVENT_ID0, EVENT_ID1, EVENT_ID2,
                                            EVENT_ID3, EVENT_ID4, EVENT_ID5,
                                            EVENT_ID6, EVENT_ID7};
  const uint32_t core_id = get_block_idx(), core_count = get_block_num();
  // the remainder, if any, rides along as one extra partial tile
  const uint32_t full_tiles = batch / Rows;
  const uint32_t tail_elems = (batch % Rows) * K;
  const uint32_t tiles = full_tiles + (tail_elems ? 1u : 0u);

  for (unsigned i = 0; i < NBuffers; ++i)  // every buffer starts free
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
  for (unsigned i = 0; i < NPrefetch; ++i)
    issue_tile_load<Shape, NBuffers>(i, core_id, core_count, tiles, full_tiles,
                                     tail_elems, buffer_free, input_gm);

  uint32_t issued = 0;
  for (uint32_t tile_index = core_id; tile_index < tiles;
       tile_index += core_count, ++issued) {
    const uint32_t buffer = issued % NBuffers;
    // issued ahead of the wait below, so this load overlaps this tile's compute
    issue_tile_load<Shape, NBuffers>(issued + NPrefetch, core_id, core_count,
                                     tiles, full_tiles, tail_elems, buffer_free,
                                     input_gm);
    wait_flag(PIPE_MTE2, PIPE_V, buffer_free[buffer]);
    const uint32_t slot_base = buffer * Shape::slot_stride;
    // name the UB regions once; inline casts are noise at every call site
    using B16 = __ubuf__ uint16_t *;
    B16 input_ub = (B16)(uintptr_t)(slot_base + Offsets::input);
    B16 scale_ub = (B16)(uintptr_t)(slot_base + Offsets::scales);
    B16 maxima_ub = (B16)(uintptr_t)Offsets::maxima;
    B16 packed_ub = (B16)(uintptr_t)Offsets::packed;
    B16 recips_ub = (B16)(uintptr_t)Offsets::reciprocal;
    __ubuf__ uint8_t *nibble_ub =
        (__ubuf__ uint8_t *)(uintptr_t)(slot_base + Offsets::nibbles);
    // rotate in place, then quantize the rotated tile without it ever leaving
    // UB
#ifndef FUSED_NO_ROTATE
    rotate<Shape>(input_ub);
    // phase 2 finishes the transform when a row is wider than one window. A
    // separate call because __tf__ may not call __tf__, and a separate
    // __VEC_SCOPE__ because it needs its own register set.
    if constexpr (Shape::phase2_stages > 0) {
      cross_windows<Shape>(input_ub);
    }
#else
    // Diagnostic build: same kernel, same tiling, same UB layout and buffer
    // count -- only the butterfly removed. Comparing this against the quantizer
    // alone separates the butterfly's vector cost from the cost of fusing at
    // all (extra UB regions, so fewer buffers, so less overlap).
    (void)0;
#endif
#ifndef FUSED_ROTATE_ONLY
    block_abs_max<Shape>(input_ub, maxima_ub);
    compact_maxima<Shape>(maxima_ub, packed_ub);
    derive_scales<Shape>(packed_ub, recips_ub, scale_ub);
    pack_nibbles<Shape>(input_ub, recips_ub, nibble_ub);
#else
    // The unfused reference's first half: butterfly only, storing the rotated
    // bf16 tile. Chained with a FUSED_NO_ROTATE build it is two launches over
    // HBM, 4 + 2.53 B/elem against the fused kernel's 2.53. Same tiling, UB
    // layout and buffer count, so only the skipped work differs.
    (void)scale_ub;
    (void)maxima_ub;
    (void)packed_ub;
    (void)recips_ub;
    (void)nibble_ub;
#endif
    set_flag(PIPE_V, PIPE_MTE3, buffer_free[buffer]);
    wait_flag(PIPE_V, PIPE_MTE3, buffer_free[buffer]);
#ifdef FUSED_ROTATE_ONLY
    // `nibble_gm` carries the rotated bf16 tile here and `scale_gm` is
    // untouched, so the launcher signature does not change. The harness
    // allocates 2K bytes per row for it, not K/2.
    if (tile_index == full_tiles) {
      move_tile<bfloat16_t, Shape::tile_elems, false, true>(
          tile_index, slot_base + Offsets::input, nibble_gm, tail_elems);
    } else {
      move_tile<bfloat16_t, Shape::tile_elems, false>(
          tile_index, slot_base + Offsets::input, nibble_gm);
    }
    (void)scale_gm;
#else
    if (tile_index == full_tiles) {
      move_tile<uint8_t, Shape::q_bytes, false, true>(
          tile_index, slot_base + Offsets::nibbles, nibble_gm, tail_elems / 2u);
      move_tile<uint8_t, Shape::scale_bytes, false, true>(
          tile_index, slot_base + Offsets::scales, scale_gm,
          tail_elems / MX_BLOCK);
    } else {
      move_tile<uint8_t, Shape::q_bytes, false>(
          tile_index, slot_base + Offsets::nibbles, nibble_gm);
      move_tile<uint8_t, Shape::scale_bytes, false>(
          tile_index, slot_base + Offsets::scales, scale_gm);
    }
#endif
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buffer]);
  }
  for (unsigned i = 0; i < NBuffers; ++i)  // drain
    wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
}
#endif  // __CCE_AICORE__ && __DAV_VEC__

template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
__global__ AICORE void fused_hadamard_mxfp4_full(__gm__ void *input_gm,
                                                 __gm__ void *nibble_gm,
                                                 __gm__ void *scale_gm,
                                                 uint32_t batch) {
#ifdef __DAV_VEC__
  quant_tiles<K, Rows, NBuffers, NPrefetch>(input_gm, nibble_gm, scale_gm,
                                            batch);
#else
  (void)input_gm;
  (void)nibble_gm;
  (void)scale_gm;
  (void)batch;
#endif
}

#ifndef FUSED_INCLUDE_ONLY  // define to take the device code without hosts
// ---------------------------------------------------------------- entry points
// One .so serves every K: fold over SUPPORTED_K for the instantiation.
template <std::size_t... Idx>
inline void launch_for_k(uint32_t block_dim, void *stream, uint8_t *input,
                         uint8_t *nibbles, uint8_t *scales, uint32_t batch,
                         uint32_t k, std::index_sequence<Idx...>) {
  ((k == SUPPORTED_K[Idx]
        ? (void)(fused_hadamard_mxfp4_full<SUPPORTED_K[Idx],
                                           RowsFor<SUPPORTED_K[Idx]>::value,
                                           DEF_BUFFERS, DEF_PREFETCH>
                 <<<block_dim, nullptr, stream>>>(input, nibbles, scales,
                                                  batch))
        : (void)0),
   ...);
}

// An unsupported k is a silent no-op; the host validates
// (check_row_width).
extern "C" void call_hadamard_mxfp4_full(uint32_t block_dim, void *stream,
                                         uint8_t *input, uint8_t *nibbles,
                                         uint8_t *scales, uint32_t batch,
                                         uint32_t k) {
  launch_for_k(block_dim, stream, input, nibbles, scales, batch, k,
               std::make_index_sequence<SUPPORTED_COUNT>{});
}

template <std::size_t... Idx>
inline uint32_t rows_for_k(uint32_t k, std::index_sequence<Idx...>) {
  uint32_t rows = 0;
  ((k == SUPPORTED_K[Idx] ? (void)(rows = RowsFor<SUPPORTED_K[Idx]>::value)
                          : (void)0),
   ...);
  return rows;  // 0 for an unsupported k
}

extern "C" uint32_t hadamard_mxfp4_full_rows_for(uint32_t k) {
  return rows_for_k(k, std::make_index_sequence<SUPPORTED_COUNT>{});
}
#endif  // FUSED_INCLUDE_ONLY
