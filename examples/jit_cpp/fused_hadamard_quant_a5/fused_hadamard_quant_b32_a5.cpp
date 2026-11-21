// Block-32 Hadamard fused with MXFP4 quantization, one launch.
//
//   x  ->  (x @ H_32 per block) -> E2M1 nibbles + one E8M0 scale per 32
//
// A row is a run of independent 32-blocks, so K need not be a power of two:
// 4096, 5120 and 14336 are all instantiable. The window is 256 = eight blocks
// and blocks never straddle a row, so one window covers eight of them
// regardless of K. The MXFP4 group is 32 too, so a scale covers exactly one
// rotated block.
//
// Not every multiple of 32 is legal: RowsFor still needs a whole grain, which
// at the default TILE_ELEMS 76 of the 512 multiples of 32 up to 16384 satisfy;
// SUPPORTED_K instantiates 28. One that does not (11008) fails a
// static_assert.
//
// Everything else is in fused_hadamard_quant_common.hpp, shared with the
// full-row kernel next door.
#include "fused_hadamard_quant_common.hpp"

// Row widths with an instantiation. A 32-wide rotation puts no power-of-two
// constraint on the row, so 4096- and 14336-style widths are in. A new width
// must still satisfy RowsFor (see above) and be added to the jit helper.
constexpr unsigned SUPPORTED_K[] = {32,   64,   96,   128,  192,  256,   512,
                                    768,  896,  1024, 1152, 1280, 1408,  1536,
                                    1664, 1792, 2048, 2560, 2816, 3072,  3584,
                                    4096, 5120, 6144, 7168, 8192, 14336, 16384};
constexpr unsigned SUPPORTED_COUNT =
    sizeof(SUPPORTED_K) / sizeof(SUPPORTED_K[0]);

#ifdef __CCE_AICORE__
#ifdef __DAV_VEC__
// log2(K) stages over the tile already in UB, in place. The quant passes read
// the same buffer straight afterwards, which is the point of the fusion.
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
  // Order = MX_BLOCK: one rotation spans a single scale's worth of elements,
  // whatever K is. That one argument is the whole difference from the
  // full-row kernel.
  using Shape = QuantShape<MX_BLOCK, K, Rows, NBuffers, NPrefetch>;
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
__global__ AICORE void fused_hadamard_mxfp4_b32(__gm__ void *input_gm,
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
        ? (void)(fused_hadamard_mxfp4_b32<SUPPORTED_K[Idx],
                                          RowsFor<SUPPORTED_K[Idx]>::value,
                                          DEF_BUFFERS, DEF_PREFETCH>
                 <<<block_dim, nullptr, stream>>>(input, nibbles, scales,
                                                  batch))
        : (void)0),
   ...);
}

// An unsupported k is a silent no-op; the host validates
// (check_row_width).
extern "C" void call_hadamard_mxfp4_b32(uint32_t block_dim, void *stream,
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

extern "C" uint32_t hadamard_mxfp4_b32_rows_for(uint32_t k) {
  return rows_for_k(k, std::make_index_sequence<SUPPORTED_COUNT>{});
}
#endif  // FUSED_INCLUDE_ONLY
