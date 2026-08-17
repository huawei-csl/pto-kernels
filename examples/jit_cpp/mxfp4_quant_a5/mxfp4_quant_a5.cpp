// mxfp4_quant_a5 — MXFP4 block quantization on Ascend A5 (dav-c310). A5 only.
// (batch, K) bf16 -> nibbles (batch, K/2) + scales (batch, K/32), both uint8.
// Each run of 32 elements shares one E8M0 scale byte and becomes 32 E2M1
// nibbles, per OCP MX v1.0 6.3 Algorithm 1 (FLOOR).
#include <pto/pto-inst.hpp>
#include <type_traits>
#include <utility>
using namespace pto;

// Row widths with an instantiation. Also add it to SUPPORTED_K in
// jit_util_mxfp4_a5.py; test_rows_for_matches_kernel pins the two together.
constexpr unsigned SUPPORTED_K[] = {64,   96,   128,  192,  256,  512,  768,
                                    896,  1024, 1152, 1280, 1408, 1536, 1664,
                                    1792, 2048, 2560, 2816, 3072, 3584, 4096,
                                    5120, 6144, 7168, 8192, 14336};
constexpr unsigned SUPPORTED_COUNT =
    sizeof(SUPPORTED_K) / sizeof(SUPPORTED_K[0]);
constexpr unsigned MX_BLOCK = 32;    // MXFP4 block: 32 elements, one E8M0 scale
constexpr unsigned DEF_BUFFERS = 4;  // UB buffers in the GM<->UB pipeline
constexpr unsigned DEF_PREFETCH = 2;  // tiles in flight ahead of the compute

// 32 KB bf16 tile. Must match rows_for() in jit_util_mxfp4_a5.py.
constexpr unsigned TILE_ELEMS = 16384;
// RULE: every GM move_tile is one row and a Tile refuses a row under 32 bytes.
// The scale row is the smallest, at tile_elems/32 bytes, so a tile must be a
// whole multiple of 32*MX_BLOCK elements. DMA sets this grain, not the compute.
constexpr unsigned TILE_GRAIN = 1024;
// ROWS_PER_TILE: the largest row count whose tile is a whole number of grains.
// Not TILE_ELEMS / K -- for a large odd factor (768 = 32*24) the quotient is
// not a multiple of the grain. Zero means inadmissible; Rows asserts on it.
template <unsigned K, unsigned Candidate>
struct LargestValidRows {
  static constexpr unsigned value =
      ((Candidate * K) % TILE_GRAIN == 0)
          ? Candidate
          : LargestValidRows<K, Candidate - 1u>::value;
};
template <unsigned K>
struct LargestValidRows<K, 0u> {
  static constexpr unsigned value = 0u;
};

template <unsigned K>
struct RowsFor {
  static constexpr unsigned cap = TILE_ELEMS / K > 1u ? TILE_ELEMS / K : 1u;
  static constexpr unsigned value = LargestValidRows<K, cap>::value;
};

#ifdef __CCE_AICORE__
constexpr unsigned B16_LANES = 128;  // bf16 lanes in one vector register
// vcgmax on b16 groups 16 lanes, 8 results in lanes 0..7.
constexpr unsigned VCGMAX_B16_GROUP = 16;
constexpr unsigned VCGMAX_B16_RESULTS = B16_LANES / VCGMAX_B16_GROUP;
static_assert(VCGMAX_B16_RESULTS == 8, "block_abs_max stores with PAT_VL8");
// RULE: vsts needs a 32-byte-aligned UB address, else 507035. Tile refuses a
// sub-32-byte DMA, so the padding is squeezed out in UB, not on the way to GM.
constexpr unsigned VSTS_ALIGN = 32;
constexpr unsigned GROUP_PITCH_B16 = VSTS_ALIGN / 2u;  // in b16 elements
// RULE: vselr indices reach only the low 128 source bytes: 4 groups per gather.
constexpr unsigned GROUPS_PER_COMPACT = 4;
constexpr unsigned EVENT_SLOTS = 8;
static_assert(EVENT_SLOTS == 8, "extend buffer_free's initialiser first");
constexpr unsigned UB_ALIGN = 512;
// A5 has 256 KB.
constexpr unsigned UB_BYTES = PTO_UBUF_SIZE_BYTES;

// bf16 bit-field constants. bf16 is 1-8-7, so a magnitude's biased exponent is
// simply bits >> 7 once the sign is cleared.
constexpr uint16_t BF16_ABS = 0x7FFFu;  // clears the sign bit
constexpr int16_t BF16_MANT_BITS = 7;
constexpr int16_t E8M0_BIAS_ADJ = -2;  // byte = b - 2 (Algorithm 1, FLOOR)
constexpr int16_t RECIP_OFFSET = 256;  // 1/X exponent field = 256 - b
// b must stay in a window where 1/X is finite, non-subnormal bf16: field 256-b
// must land in [2, 254]. Clamp b, then derive BOTH outputs from the clamped b.
constexpr int16_t B_MIN = 2;
constexpr int16_t B_MAX = 254;

// RULE: a constexpr function cannot be called from [aicore] code.
template <unsigned Bytes>
struct RoundUp {
  static constexpr unsigned value = (Bytes + UB_ALIGN - 1) & ~(UB_ALIGN - 1);
};

// Every derived size for one instantiation.
template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
struct QuantShape {
  static constexpr unsigned tile_elems = Rows * K;
  static constexpr unsigned blocks = tile_elems / MX_BLOCK;
  static constexpr unsigned in_bytes = tile_elems * 2u;
  static constexpr unsigned q_bytes = tile_elems / 2u;
  static constexpr unsigned scale_bytes = blocks;

  // One "group" is one vcgmax: 8 blocks == 256 elements.
  static constexpr unsigned groups = blocks / VCGMAX_B16_RESULTS;
  // Round UP: the last bite may be partial. Safe only because the buffers
  // below are sized from these counts, and nothing reads what a partial bite
  // writes past `blocks`. The asserts at the end of this struct pin both.
  static constexpr unsigned compact_iters =
      (groups + GROUPS_PER_COMPACT - 1u) / GROUPS_PER_COMPACT;
  static constexpr unsigned b_iters = (blocks + B16_LANES - 1u) / B16_LANES;
  static constexpr unsigned c_iters = tile_elems / (2u * B16_LANES);

  // maxima_bytes carries a register of read-ahead: the gather reads one
  // register past its last input offset. The rest are sized from what the loops
  // WRITE, since the two rounded counts can exceed their data by one bite.
  static constexpr unsigned maxima_bytes =
      compact_iters * GROUPS_PER_COMPACT * GROUP_PITCH_B16 * 2u + B16_LANES;
  static constexpr unsigned packed_bytes =
      compact_iters * GROUPS_PER_COMPACT * VCGMAX_B16_RESULTS * 2u;
  static constexpr unsigned aligned_in = RoundUp<in_bytes>::value;
  static constexpr unsigned aligned_q = RoundUp<q_bytes>::value;
  static constexpr unsigned aligned_s = RoundUp<b_iters * B16_LANES>::value;
  static constexpr unsigned aligned_max = RoundUp<maxima_bytes>::value;
  static constexpr unsigned aligned_packed = RoundUp<packed_bytes>::value;
  static constexpr unsigned aligned_mult =
      RoundUp<b_iters * B16_LANES * 2u>::value;

  static constexpr unsigned slot_stride = aligned_in + aligned_q + aligned_s;
  static constexpr unsigned scratch_base = NBuffers * slot_stride;
  // every scratch region SlotOffset hands out, in the same order, so the two
  // cannot drift: omitting one here silently shrinks the UB-overflow guard
  static constexpr unsigned ub_needed =
      scratch_base + aligned_max + aligned_packed + aligned_mult;

#ifdef MXFP4_TQUANT
  // TQuant reads its per-block maxima a whole register at a time, so the span
  // rounds up to b_iters registers, and its reducer flushes one 32-byte block
  // past the last group.
  static constexpr unsigned tquant_max_elems = b_iters * B16_LANES;
  static constexpr unsigned tquant_max_bytes =
      tquant_max_elems * 2u + VSTS_ALIGN;
  static constexpr unsigned tquant_scaling_bytes = blocks * 2u;
  static_assert(tquant_max_bytes <= aligned_max,
                "TQuant maxima do not fit the maxima region");
  static_assert(tquant_scaling_bytes <= aligned_mult,
                "TQuant scaling does not fit the reciprocal region");
  // numGroups truncates inside TQuant, and a partial 8-group window takes a
  // different store path.
  static_assert(tile_elems % MX_BLOCK == 0, "TQuant would drop a group");
  static_assert(blocks % 8u == 0, "TQuant would take its vstus tail path");
#endif

  static_assert(Rows > 0, "no Rows makes Rows*K a whole TILE_GRAIN: bad K");
  static_assert(K % MX_BLOCK == 0, "a block may not straddle a row boundary");
  static_assert(TILE_GRAIN == VSTS_ALIGN * MX_BLOCK, "grain != scale DMA row");
  static_assert(tile_elems % TILE_GRAIN == 0, "tile is not a whole grain");
  static_assert(tile_elems % (2u * B16_LANES) == 0, "pack_nibbles wants 256");
  static_assert(scale_bytes % VSTS_ALIGN == 0, "scale row is not a legal DMA");
  static_assert(q_bytes % VSTS_ALIGN == 0, "nibble row is not a legal DMA");
  static_assert(in_bytes % VSTS_ALIGN == 0, "input row is not a legal DMA");
  static_assert(blocks % VCGMAX_B16_RESULTS == 0,
                "blocks != whole vcgmax groups");
  // the rounded-up passes run one bite past the data; prove they stay inside
  static_assert(b_iters * B16_LANES <= aligned_s, "scale tail overruns");
  static_assert(b_iters * B16_LANES * 2u <= aligned_mult,
                "recips tail overruns");
  static_assert(packed_bytes <= aligned_packed, "compaction tail overruns");
  static_assert(groups * VSTS_ALIGN <= aligned_max, "padded maxima overrun");
  static_assert(
      c_iters * VCGMAX_B16_RESULTS <= b_iters * B16_LANES,
      "pack_nibbles would index recips past what derive_scales wrote");
  static_assert(sizeof(bfloat16_t) == 2, "RowsFor assumes 2-byte elements");
  static_assert(ub_needed <= UB_BYTES, "UB overflow");
  static_assert(NBuffers <= EVENT_SLOTS, "NBUF exceeds the event-id array");
  static_assert(NPrefetch < NBuffers,
                "PREFETCH == NBUF deadlocks the pipeline");
};

// Byte offsets within a pipeline slot, plus shared scratch. Constexpr
// *variables* for the reason above, so the slot base is multiplied in at
// the use site.
template <typename Shape>
struct SlotOffset {
  static constexpr unsigned input = 0;
  static constexpr unsigned nibbles = Shape::aligned_in;
  static constexpr unsigned scales = Shape::aligned_in + Shape::aligned_q;
  static constexpr unsigned maxima = Shape::scratch_base;
  static constexpr unsigned packed = Shape::scratch_base + Shape::aligned_max;
  static constexpr unsigned reciprocal = packed + Shape::aligned_packed;
};

#ifdef __DAV_VEC__
// A flat run of Elems values in GM, and the matching UB tile, for one dtype.
template <typename T, unsigned Elems>
using GmShape = pto::Shape<1, 1, 1, 1, Elems>;
template <typename T, unsigned Elems>
using GmStride = pto::Stride<1, 1, 1, Elems, 1>;
template <typename T, unsigned Elems>
using UbTile = Tile<TileType::Vec, T, 1, Elems, BLayout::RowMajor, 1, Elems>;
// Same tile with a RUNTIME valid column count, zero-filling the rest in UB, for
// the one partial tile a batch can end on.
template <typename T, unsigned Elems>
using UbTilePart =
    Tile<TileType::Vec, T, 1, Elems, BLayout::RowMajor, 1, DYNAMIC,
         SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Zero>;

// ------------------------------------------------------- block_abs_max
// Per-32-element magnitude max. A 2:1 fold makes 16 lanes == one block,
// which is what vcgmax's group size requires. A 4:1 fold silently reports
// max(block 2j, block 2j+1) instead.
template <typename Shape>
__tf__ static AICORE void block_abs_max(__ubuf__ uint16_t *input,
                                        __ubuf__ uint16_t *maxima) {
  __VEC_SCOPE__ {
    MaskReg all_lanes = pset_b16(PAT_ALL);
    // PAT_VL8 matches VCGMAX_B16_RESULTS
    MaskReg low_eight = pset_b16(PAT_VL8);
    vector_u16 abs_mask;
    vdup(abs_mask, BF16_ABS, all_lanes, MODE_ZEROING);

    for (uint16_t group = 0; group < (uint16_t)Shape::groups; ++group) {
      const uint32_t base = (uint32_t)group * 256u;
      vector_u16 even, odd, folded, grouped;
      vlds(even, odd, input + base, 0,
           DINTLV_B16);  // lane i: elements 2i, 2i+1
      vand(even, even, abs_mask, all_lanes);
      vand(odd, odd, abs_mask, all_lanes);
      // sign cleared, so a signed max over the bit patterns IS a magnitude max
      vmax((vector_s16 &)folded, (vector_s16 &)even, (vector_s16 &)odd,
           all_lanes);
      vcgmax((vector_s16 &)grouped, (vector_s16 &)folded, all_lanes);
      // 32-byte pitch, not 16: see VSTS_ALIGN
      vsts(grouped, maxima + (uint32_t)group * GROUP_PITCH_B16, 0, NORM_B16,
           low_eight);
    }
    mem_bar(VST_VLD);
  }
}

// ------------------------------------------------------ compact_maxima
// Squeeze out the padding VSTS_ALIGN forces: output byte i takes input byte
// 2*(i & 0xF0) + (i & 0x0F).
template <typename Shape>
__tf__ static AICORE void compact_maxima(__ubuf__ uint16_t *padded,
                                         __ubuf__ uint16_t *packed) {
  __VEC_SCOPE__ {
    MaskReg all_byte_lanes = pset_b8(PAT_ALL);
    MaskReg low_32 = pset_b16(PAT_VL32);  // 32 b16 == 64 bytes
    vector_u8 byte_index, high_half, low_half, high_mask, low_mask;
    vci((vector_s8 &)byte_index, (int8_t)0, INC_ORDER);
    vdup(high_mask, (uint8_t)0xF0, all_byte_lanes, MODE_ZEROING);
    vdup(low_mask, (uint8_t)0x0F, all_byte_lanes, MODE_ZEROING);
    vand(high_half, byte_index, high_mask, all_byte_lanes);
    vand(low_half, byte_index, low_mask, all_byte_lanes);
    vadd((vector_s8 &)high_half, (vector_s8 &)high_half, (vector_s8 &)high_half,
         all_byte_lanes);  // 2*high_half
    vadd((vector_s8 &)byte_index, (vector_s8 &)high_half, (vector_s8 &)low_half,
         all_byte_lanes);

    for (uint16_t gather = 0; gather < (uint16_t)Shape::compact_iters;
         ++gather) {
      vector_u16 padded_chunk, packed_chunk;
      const uint32_t src_offset =
          (uint32_t)gather * GROUPS_PER_COMPACT * GROUP_PITCH_B16;
      vlds(padded_chunk, padded + src_offset, 0, NORM);
      vselr((vector_u8 &)packed_chunk, (vector_u8 &)padded_chunk, byte_index);
      vsts(packed_chunk,
           packed + (uint32_t)gather * GROUPS_PER_COMPACT * VCGMAX_B16_RESULTS,
           0, NORM_B16, low_32);
    }
    mem_bar(VST_VLD);
  }
}

// -------------------------------------------------------- derive_scales
// maxima -> E8M0 scale byte + one bf16 reciprocal per block. pack_nibbles
// reads this array with E2B_B16, whose x16 replication matches its
// pair-granular deinterleave exactly, so no duplication is needed here.
template <typename Shape>
__tf__ static AICORE void derive_scales(__ubuf__ uint16_t *maxima,
                                        __ubuf__ uint16_t *recips_out,
                                        __ubuf__ uint16_t *scale_out) {
  __VEC_SCOPE__ {
    MaskReg all_lanes = pset_b16(PAT_ALL);
    vector_u16 bias;
    vdup(bias, (uint16_t)RECIP_OFFSET, all_lanes, MODE_ZEROING);

    for (uint16_t chunk = 0; chunk < (uint16_t)Shape::b_iters; ++chunk) {
      vector_u16 block_max, exponent, scale_byte, reciprocal;
      vlds(block_max, maxima + (uint32_t)chunk * B16_LANES, 0, NORM);
      // bit 15 is already clear, so this shift alone yields the biased exponent
      vshrs(exponent, block_max, BF16_MANT_BITS, all_lanes, MODE_ZEROING);
      vmaxs(exponent, exponent, B_MIN, all_lanes);
      vmins(exponent, exponent, B_MAX, all_lanes);
      vadds(scale_byte, exponent, E8M0_BIAS_ADJ, all_lanes);
      vsts(scale_byte, scale_out + (uint32_t)chunk * 64u, 0, PK_B16, all_lanes);
      vsub(reciprocal, bias, exponent, all_lanes);
      vshls(reciprocal, reciprocal, BF16_MANT_BITS, all_lanes, MODE_ZEROING);
      vsts(reciprocal, recips_out + (uint32_t)chunk * B16_LANES, 0, NORM_B16,
           all_lanes);
    }
    mem_bar(VST_VLD);
  }
}

// --------------------------------------------------------- pack_nibbles
// Scale, cast, pack -- 256 elements per iteration, no gather.
// One vcvt puts 64 bytes at byte STRIDE 4, offset chosen by
// PART_P0..P3, so converting two halves into offsets 0 and 1, OR-ing, and
// storing with PK_B32 (keeps the low 2 bytes of each 4-byte group) writes 128
// CONTIGUOUS bytes. RULE: fp4 packs two elements per byte, so DINTLV_B16 would
// pair element 4k with 4k+2 -- deinterleave at b32 (pairs) to keep (4k, 4k+1)
// together. That also puts both b16 lanes of a half in block j/8, so E2B_B16's
// x16 replication is exact and one multiplier register serves both halves.
template <typename Shape>
__tf__ static AICORE void pack_nibbles(__ubuf__ uint16_t *input,
                                       __ubuf__ uint16_t *reciprocal,
                                       __ubuf__ uint8_t *nibble_out) {
  __VEC_SCOPE__ {
    MaskReg all_lanes = pset_b16(PAT_ALL);
    MaskReg all_byte_lanes = pset_b8(PAT_ALL);
    MaskReg all_b32_lanes = pset_b32(PAT_ALL);

    for (uint16_t chunk = 0; chunk < (uint16_t)Shape::c_iters; ++chunk) {
      vector_u16 recips;
      vector_u32 even, odd;
      vector_bf16 scaled_even, scaled_odd;
      vector_f4e2m1x2 packed_even, packed_odd, packed;
      vlds(recips, reciprocal + (uint32_t)chunk * VCGMAX_B16_RESULTS, 0,
           E2B_B16);
      vlds(even, odd, (__ubuf__ uint32_t *)input + (uint32_t)chunk * B16_LANES,
           0, DINTLV_B32);
      vmul(scaled_even, (vector_bf16 &)even, (vector_bf16 &)recips, all_lanes);
      vmul(scaled_odd, (vector_bf16 &)odd, (vector_bf16 &)recips, all_lanes);
      vcvt(packed_even, scaled_even, all_lanes, ROUND_R, PART_P0);
      vcvt(packed_odd, scaled_odd, all_lanes, ROUND_R, PART_P1);
      vor((vector_u8 &)packed, (vector_u8 &)packed_even,
          (vector_u8 &)packed_odd, all_byte_lanes);
      // 256 elements in, but PK_B32 keeps 2 of every 4 bytes: 128 bytes out
      vsts((vector_u16 &)packed,
           (__ubuf__ uint16_t *)(nibble_out + (uint32_t)chunk * B16_LANES), 0,
           PK_B32, all_b32_lanes);
    }
    mem_bar(VST_VLD);
  }
}

#ifdef MXFP4_TQUANT
// Requires PTO 9.1.0: 9.0.0 has no MXFP4 quantizer. Included here, not at file
// scope, because this region is inside the device-pass guard.
#include <pto/npu/a5/TQuant.hpp>

// ------------------------------------------------------- tquant_passes
// One vendor tile op in place of block_abs_max, compact_maxima, derive_scales
// and pack_nibbles. validCols is tile_elems even on the partial tile: the load
// already zero-fills the pad, and a short validCols would send TQuant's own
// ZeroPadSourceTile over the input slot. Offsets::packed is left allocated and
// unused, since reclaiming it would move slot_stride.
template <typename Shape>
inline AICORE void tquant_passes(uint32_t input_offset, uint32_t nibble_offset,
                                 uint32_t scale_offset) {
  static_assert(sizeof(float4_e2m1x2_t) == 1,
                "the nibble tile assumes one byte per float4_e2m1x2_t");
  static_assert(REPEAT_BYTE / sizeof(bfloat16_t) == B16_LANES,
                "tquant_max_elems assumes a 128-lane b16 vector");
  UbTile<bfloat16_t, Shape::tile_elems> source;
  UbTile<float4_e2m1x2_t, Shape::q_bytes> nibbles;
  UbTile<uint8_t, Shape::scale_bytes> scales;
  UbTile<bfloat16_t, Shape::tquant_max_elems> block_max;
  UbTile<bfloat16_t, Shape::blocks> reciprocal;
  TASSIGN(source, input_offset);
  TASSIGN(nibbles, nibble_offset);
  TASSIGN(scales, scale_offset);
  TASSIGN(block_max, SlotOffset<Shape>::maxima);
  TASSIGN(reciprocal, SlotOffset<Shape>::reciprocal);
  // TEMPLATE order is Out, Src, Exp, Max, Scaling; ARGUMENT order is dst, exp,
  // max, scaling, src. PTO 9.1.0 release inserted a `bool Exp2DStrided` second
  // template parameter that 9.1.0-beta.3 does not have; the tile types are in a
  // non-deduced position, so neither spelling can be dropped. benchmark.py
  // compiles both and keeps whichever the local headers accept.
#ifdef MXFP4_TQUANT_EXP2D
  TQuant_MXFP4_E2M1_Impl<QuantScaleAlg::OCP, false, decltype(nibbles),
                         decltype(source), decltype(scales),
                         decltype(block_max), decltype(reciprocal)>(
      nibbles.data(), scales.data(), block_max.data(), reciprocal.data(),
      source.data(), 1u, Shape::tile_elems);
#else
  TQuant_MXFP4_E2M1_Impl<QuantScaleAlg::OCP, decltype(nibbles),
                         decltype(source), decltype(scales),
                         decltype(block_max), decltype(reciprocal)>(
      nibbles.data(), scales.data(), block_max.data(), reciprocal.data(),
      source.data(), 1u, Shape::tile_elems);
#endif
}
#endif  // MXFP4_TQUANT

// Move one tile of `T` between GM and UB. Partial carries only `valid`
// elements: the load zero-fills the rest of the UB tile so the compute passes
// still see whole registers, and the store truncates so padding never reaches
// GM.
template <typename T, unsigned Elems, bool ToUb, bool Partial = false>
inline AICORE void move_tile(uint32_t tile_index, uint32_t ub_offset,
                             __gm__ void *gm_base, uint32_t valid = 0) {
  std::conditional_t<Partial, UbTilePart<T, Elems>, UbTile<T, Elems>> ub;
  TASSIGN(ub, ub_offset);
  if constexpr (Partial) ub.ColMaskInternal = (int)valid;
  GlobalTensor<T, GmShape<T, Elems>, GmStride<T, Elems>> gm(
      (__gm__ T *)gm_base + (uint64_t)tile_index * Elems, GmShape<T, Elems>());
  if constexpr (ToUb) {
    TLOAD(ub, gm);
  } else {
    TSTORE(gm, ub);
  }
}

// Start the async load of this core's nth tile, if it has one. A function,
// not a lambda: set_flag/wait_flag do not resolve inside a lambda.
template <typename Shape, unsigned Buffers>
inline AICORE void issue_tile_load(uint32_t nth_tile, uint32_t core_id,
                                   uint32_t core_count, uint32_t tiles,
                                   uint32_t full_tiles, uint32_t tail_elems,
                                   const event_t *buffer_free,
                                   __gm__ void *input_gm) {
  const uint32_t tile_index = core_id + nth_tile * core_count;
  if (tile_index >= tiles) return;
  const uint32_t buffer = nth_tile % Buffers;
  const uint32_t off = buffer * Shape::slot_stride + SlotOffset<Shape>::input;
  wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buffer]);
  // at most one tile is partial, and only when batch does not fill it
  if (tile_index == full_tiles) {
    move_tile<bfloat16_t, Shape::tile_elems, true, true>(tile_index, off,
                                                         input_gm, tail_elems);
  } else {
    move_tile<bfloat16_t, Shape::tile_elems, true>(tile_index, off, input_gm);
  }
  set_flag(PIPE_MTE2, PIPE_V, buffer_free[buffer]);
}
#endif  // __DAV_VEC__
#endif  // __CCE_AICORE__

// The pipeline: each core walks a strided subset of the tiles, keeping Prefetch
// loads in flight so DMA and the vector pipe overlap.
template <unsigned K, unsigned Rows, unsigned NBuffers, unsigned NPrefetch>
__global__ AICORE void mxfp4_quant(__gm__ void *input_gm,
                                   __gm__ void *nibble_gm,
                                   __gm__ void *scale_gm, uint32_t batch) {
#ifdef __DAV_VEC__
  using Shape = QuantShape<K, Rows, NBuffers, NPrefetch>;
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
#ifdef MXFP4_TQUANT
    tquant_passes<Shape>(slot_base + Offsets::input,
                         slot_base + Offsets::nibbles,
                         slot_base + Offsets::scales);
#else
    // name the UB regions once; inline casts are noise at every call site
    using B16 = __ubuf__ uint16_t *;
    B16 input_ub = (B16)(uintptr_t)(slot_base + Offsets::input);
    B16 scale_ub = (B16)(uintptr_t)(slot_base + Offsets::scales);
    B16 maxima_ub = (B16)(uintptr_t)Offsets::maxima;
    B16 packed_ub = (B16)(uintptr_t)Offsets::packed;
    B16 recips_ub = (B16)(uintptr_t)Offsets::reciprocal;
    __ubuf__ uint8_t *nibble_ub =
        (__ubuf__ uint8_t *)(uintptr_t)(slot_base + Offsets::nibbles);
    block_abs_max<Shape>(input_ub, maxima_ub);
    compact_maxima<Shape>(maxima_ub, packed_ub);
    derive_scales<Shape>(packed_ub, recips_ub, scale_ub);
    pack_nibbles<Shape>(input_ub, recips_ub, nibble_ub);
#endif
    set_flag(PIPE_V, PIPE_MTE3, buffer_free[buffer]);
    wait_flag(PIPE_V, PIPE_MTE3, buffer_free[buffer]);
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
    set_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[buffer]);
  }
  for (unsigned i = 0; i < NBuffers; ++i)  // drain
    wait_flag(PIPE_MTE3, PIPE_MTE2, buffer_free[i]);
#else
  (void)input_gm;
  (void)nibble_gm;
  (void)scale_gm;
  (void)batch;
#endif
}

// ---------------------------------------------------------------- entry points
// One .so serves every K: fold over SUPPORTED_K for the instantiation.
template <std::size_t... Idx>
inline void launch_for_k(uint32_t block_dim, void *stream, uint8_t *input,
                         uint8_t *nibbles, uint8_t *scales, uint32_t batch,
                         uint32_t k, std::index_sequence<Idx...>) {
  ((k == SUPPORTED_K[Idx]
        ? (void)(mxfp4_quant<SUPPORTED_K[Idx], RowsFor<SUPPORTED_K[Idx]>::value,
                             DEF_BUFFERS, DEF_PREFETCH>
                 <<<block_dim, nullptr, stream>>>(input, nibbles, scales,
                                                  batch))
        : (void)0),
   ...);
}

// An unsupported k is a silent no-op; the host validates
// (check_row_width).
extern "C" void call_mxfp4_quant(uint32_t block_dim, void *stream,
                                 uint8_t *input, uint8_t *nibbles,
                                 uint8_t *scales, uint32_t batch, uint32_t k) {
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

extern "C" uint32_t mxfp4_rows_for(uint32_t k) {
  return rows_for_k(k, std::make_index_sequence<SUPPORTED_COUNT>{});
}
