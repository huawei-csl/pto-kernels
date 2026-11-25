// MXFP4 x MXFP4 matmul on the A5 cube's microscaled path, TMATMUL_MX.
//
//   A: (M, K) E2M1 nibbles, two per byte, + one E8M0 scale per 32 along K
//   B: (K, N) the same, stored DN
//   y: (M, N) bf16, accumulated in fp32
//
// Layouts are asymmetric and not implied by the type names. Feeding
// row-major B compiles and computes something else.
//   A data   Layout::ND       A scale  Layout::MX_A_ND
//   B data   Layout::DN       B scale  Layout::MX_B_DN
//
// Scale tiles bind by ADDRESS, not by argument: TMATMUL_MX takes them for the
// type check but never forwards them to the mad_mx builtin
// (a5/TMatmul.hpp:259-271). The hardware reads them at
// GetScaleAddr(operand.data()), a >> 4 of the operand's L0 address
// (a5/utils.hpp:82-87). TASSIGNing them anywhere else compiles and multiplies
// by whatever is there.
//
// CheckMadMxValid asserts K a multiple of 64 elements, N a multiple of 64,
// M a multiple of 16, and an fp32 accumulator.
//
// Measurements and the sweeps behind the defaults are in README.md.
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

// Hardware limits. Each is named once and every static_assert below refers to
// the name, so a port states its constraints rather than repeating literals.
constexpr unsigned M_ALIGN = 16;  // TMATMUL_MX's row alignment
constexpr unsigned N_ALIGN = 64;  // fp4 column alignment
constexpr unsigned K_ALIGN = 64;  // scales pair along K, so K/32 must be even
constexpr uint32_t L1_BYTES = 512u * 1024u;
constexpr uint32_t L0A_BYTES = 64u * 1024u;
constexpr uint32_t L0B_BYTES = 64u * 1024u;
constexpr unsigned MAT_FRACTAL_BYTES = 512;
constexpr unsigned SCALE_FRACTAL_BYTES = 32;

constexpr uint32_t SCALE_FACTOR = 32;
constexpr uint32_t SHIFT_SCALE_FACTOR = 5;
constexpr uint32_t SHIFT_FP4 = 1;
constexpr unsigned BASE_K = 256;

// Three output-tile row counts against one shared N. The launcher picks by M:
// a taller tile re-fetches less L1 but halves the block count, and M_ALIGN is
// the floor the hardware imposes.
#ifndef MXMM_BASE_M
#define MXMM_BASE_M 128
#endif
#ifndef MXMM_BASE_N
#define MXMM_BASE_N 256
#endif
constexpr unsigned TILE_N = MXMM_BASE_N;
constexpr unsigned TILE_M = MXMM_BASE_M;
constexpr unsigned TILE_M_BIG = 2u * TILE_M;
constexpr unsigned TILE_M_MIN = M_ALIGN;

// The L1 slab's K extent, independent of the cube's BASE_K: the cube consumes
// BASE_K at a time from a sub-tile loop inside the slab.
#ifndef MXMM_K_L1
#define MXMM_K_L1 512
#endif
constexpr unsigned K_L1 = MXMM_K_L1;

// How far ahead the GM->L1 load runs. One set means no overlap and half the
// L1 free for a wider slab.
#ifndef MXMM_L1_SETS
#define MXMM_L1_SETS 2
#endif
constexpr uint32_t L1_SETS = (uint32_t)(MXMM_L1_SETS);
constexpr uint32_t LOOKAHEAD = L1_SETS - 1u;
static_assert(L1_SETS >= 1u && L1_SETS <= 2u, "One or two L1 sets.");

// Output tiles are walked kSwizzleGroup rows down before stepping across.
// 1 restores the plain row-major walk; -DMXMM_SWIZZLE overrides the default.
constexpr uint32_t SWIZZLE_WIDE_TILES = 48u;
constexpr uint32_t SWIZZLE_WIDE = 2u;
constexpr uint32_t SWIZZLE_NARROW = 8u;

// The shape this translation unit is built for.
#ifndef MXMM_TEST_K
#define MXMM_TEST_K 512
#endif
#ifndef MXMM_TEST_N
#define MXMM_TEST_N 512
#endif
constexpr unsigned TEST_M_MAX = 65536;
constexpr unsigned TEST_K = MXMM_TEST_K, TEST_N = MXMM_TEST_N;
// The block count the tile picker aims to fill. jit_util passes the device's
// cube_core_num; the default is what an A5 reports.
#ifndef MXMM_TARGET_BLOCKS
#define MXMM_TARGET_BLOCKS 32
#endif
constexpr uint32_t TARGET_BLOCKS = (uint32_t)(MXMM_TARGET_BLOCKS);
static_assert(TARGET_BLOCKS >= 1u, "At least one block.");

namespace {

// Every size the kernel derives from one output-tile choice, with the
// static_asserts that make an unsupported combination fail to instantiate
// rather than fault on device.
template <unsigned kMMax, unsigned kMatK, unsigned kMatN, unsigned kTileM,
          unsigned kTileN>
struct TileShape {
  static constexpr unsigned kRows = kTileM;
  static constexpr unsigned kCols = kTileN;
  static constexpr unsigned kMax = kMMax;
  static constexpr unsigned kK = kMatK;
  static constexpr unsigned kN = kMatN;

  static constexpr unsigned kScaleK = kMatK / SCALE_FACTOR;
  static constexpr unsigned kTileScaleK = BASE_K / SCALE_FACTOR;
  static constexpr uint32_t kNTiles = kMatN / kTileN;
  static constexpr uint32_t kCubeTiles = kMatK / BASE_K;
  static constexpr uint32_t kSlabs = kMatK / K_L1;
  static constexpr uint32_t kSubTiles = K_L1 / BASE_K;

  static constexpr uint32_t kSlabABytes = (kTileM * K_L1) >> SHIFT_FP4;
  static constexpr uint32_t kSlabBBytes = (K_L1 * kTileN) >> SHIFT_FP4;
  static constexpr uint32_t kSlabBytes = kSlabABytes + kSlabBBytes;
  static constexpr uint32_t kScaleBytes = kTileM * kScaleK + kScaleK * kTileN;
  static constexpr uint32_t kL0ABytes = (kTileM * BASE_K) >> SHIFT_FP4;
  static constexpr uint32_t kL0BBytes = (BASE_K * kTileN) >> SHIFT_FP4;

#ifdef MXMM_SWIZZLE
  static constexpr uint32_t kSwizzleGroup = (uint32_t)(MXMM_SWIZZLE);
#else
  static constexpr uint32_t kSwizzleGroup =
      kNTiles >= SWIZZLE_WIDE_TILES ? SWIZZLE_WIDE : SWIZZLE_NARROW;
#endif

  static_assert(kTileM % M_ALIGN == 0, "TMATMUL_MX needs M M_ALIGN-aligned.");
  static_assert(kTileN % N_ALIGN == 0, "fp4 needs N N_ALIGN-aligned.");
  static_assert(BASE_K % K_ALIGN == 0, "The cube K tile must be K_ALIGN.");
  static_assert(K_L1 % BASE_K == 0, "The slab must be whole cube K tiles.");
  static_assert(kSubTiles >= 1u, "At least one sub-tile per slab.");
  static_assert(kMatK % K_L1 == 0u, "K must be whole L1 slabs.");
  static_assert(kMatN % kTileN == 0u, "N must be whole output tiles.");
  static_assert(kSwizzleGroup >= 1u, "A group spans at least one tile row.");
  // Holding the scales for the whole K costs 16*K bytes, so past a width they
  // no longer fit beside the data sets.
  static_assert(L1_SETS * kSlabBytes + kScaleBytes <= L1_BYTES,
                "The data sets plus the whole-K scale pair must fit L1; "
                "reduce L1_SETS or K.");
  static_assert(2u * kL0ABytes <= L0A_BYTES, "Two A tiles must fit L0A.");
  static_assert(2u * kL0BBytes <= L0B_BYTES, "Two B tiles must fit L0B.");
};

#if defined(__DAV_CUBE__)
template <typename Shape>
AICORE void runMxfp4Matmul(__gm__ void *a_gm, __gm__ void *a_scale_gm,
                           __gm__ void *b_gm, __gm__ void *b_scale_gm,
                           __gm__ void *out_gm, uint32_t m_total,
                           uint32_t core_count) {
  using Fp4 = float4_e2m1x2_t;
  using E8m0 = float8_e8m0_t;

  // A static tile shape against a full-matrix BaseShape2D reads A's first 8
  // rows as zero. Scale shapes must come from TileShape2D besides: TLoad
  // asserts staticShape[4] == 2 or -1 for MX_*_ND/DN (a5/TLoad.hpp:85),
  // because scales are paired along K -- which is also why K must be a
  // multiple of 64 rather than of 32.
  using DynShape = pto::Shape<1, 1, 1, -1, -1>;
  using AGlobal =
      GlobalTensor<Fp4, DynShape,
                   BaseShape2D<Fp4, Shape::kMax, Shape::kK, Layout::ND>,
                   Layout::ND>;
  using BGlobal =
      GlobalTensor<Fp4, DynShape,
                   BaseShape2D<Fp4, Shape::kK, Shape::kN, Layout::DN>,
                   Layout::DN>;
  using AScaleGlobal = GlobalTensor<
      E8m0, DynShape,
      BaseShape2D<E8m0, Shape::kMax, Shape::kScaleK, Layout::MX_A_ND>,
      Layout::MX_A_ND>;
  using BScaleGlobal = GlobalTensor<
      E8m0, DynShape,
      BaseShape2D<E8m0, Shape::kScaleK, Shape::kN, Layout::MX_B_DN>,
      Layout::MX_B_DN>;
  using OutGlobal =
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, Shape::kRows, Shape::kCols>,
                   BaseShape2D<bfloat16_t, Shape::kMax, Shape::kN, Layout::ND>,
                   Layout::ND>;

  // The 512-byte fractal has to be explicit: omitting it reads A's first 8
  // rows as zero.
  using ATile = Tile<TileType::Mat, Fp4, Shape::kRows, K_L1, BLayout::ColMajor,
                     Shape::kRows, K_L1, SLayout::RowMajor, MAT_FRACTAL_BYTES>;
  using BTile = Tile<TileType::Mat, Fp4, K_L1, Shape::kCols, BLayout::RowMajor,
                     K_L1, Shape::kCols, SLayout::ColMajor, MAT_FRACTAL_BYTES>;
  using AScaleTile = Tile<TileType::Mat, E8m0, Shape::kRows, Shape::kScaleK,
                          BLayout::RowMajor, Shape::kRows, Shape::kScaleK,
                          SLayout::RowMajor, SCALE_FRACTAL_BYTES>;
  using BScaleTile = Tile<TileType::Mat, E8m0, Shape::kScaleK, Shape::kCols,
                          BLayout::ColMajor, Shape::kScaleK, Shape::kCols,
                          SLayout::ColMajor, SCALE_FRACTAL_BYTES>;

  using LeftTile = TileLeft<Fp4, Shape::kRows, BASE_K, Shape::kRows, BASE_K>;
  using RightTile = TileRight<Fp4, BASE_K, Shape::kCols, BASE_K, Shape::kCols>;
  using LeftScaleTile = TileLeftScale<E8m0, Shape::kRows, Shape::kTileScaleK,
                                      Shape::kRows, Shape::kTileScaleK>;
  using RightScaleTile = TileRightScale<E8m0, Shape::kTileScaleK, Shape::kCols,
                                        Shape::kTileScaleK, Shape::kCols>;
  using AccTile =
      TileAcc<float, Shape::kRows, Shape::kCols, Shape::kRows, Shape::kCols>;

  ATile aL1, aL1b;
  BTile bL1, bL1b;
  AScaleTile aScaleL1;
  BScaleTile bScaleL1;
  LeftTile aL0, aL0b;
  RightTile bL0, bL0b;
  LeftScaleTile aScaleL0, aScaleL0b;
  RightScaleTile bScaleL0, bScaleL0b;
  AccTile accL0;

  TASSIGN(aL1, 0x0);
  TASSIGN(bL1, Shape::kSlabABytes);
  TASSIGN(aL1b, Shape::kSlabBytes);
  TASSIGN(bL1b, Shape::kSlabBytes + Shape::kSlabABytes);
  TASSIGN(aScaleL1, L1_SETS * Shape::kSlabBytes);
  TASSIGN(bScaleL1,
          L1_SETS * Shape::kSlabBytes + Shape::kRows * Shape::kScaleK);
  TASSIGN(aL0, 0x0);
  TASSIGN(bL0, 0x0);
  TASSIGN(aL0b, Shape::kL0ABytes);
  TASSIGN(bL0b, Shape::kL0BBytes);
  TASSIGN(accL0, 0x0);
  TASSIGN(aScaleL0, GetScaleAddr(aL0.data()));
  TASSIGN(bScaleL0, GetScaleAddr(bL0.data()));
  TASSIGN(aScaleL0b, GetScaleAddr(aL0b.data()));
  TASSIGN(bScaleL0b, GetScaleAddr(bL0b.data()));

  const uint32_t m_tiles = m_total / Shape::kRows;
  const uint32_t out_tiles = m_tiles * Shape::kNTiles;
  const uint32_t per_group = Shape::kSwizzleGroup * Shape::kNTiles;

  // Every pipe is in order, so one counter per direction suffices.
  //
  //   MTE2 -> MTE1  ID0  a data slab has landed in L1
  //                 ID1  the whole-K scale pair has landed
  //   MTE1 -> MTE2  ID2  an L1 data set is free to refill
  //                 ID3  the scale buffer is free to refill
  //   MTE1 -> M     ID4  an L0 extract is done
  //   M    -> MTE1  ID5  the cube has released an L0 set
  //   M    -> FIX   ID0  the accumulator is complete
  //   FIX  -> M     ID6  the store has read the accumulator
  //
  // MTE2 runs ahead of MTE1, so without ID3 a tile's scale load lands on top
  // of scales the previous tile is still extracting.
  bool scales_held = false;
  for (uint32_t t = get_block_idx(); t < out_tiles; t += core_count) {
    const uint32_t in_group = t % per_group;
    const uint32_t first_mt = (t / per_group) * Shape::kSwizzleGroup;
    const uint32_t group_mts = m_tiles - first_mt < Shape::kSwizzleGroup
                                   ? m_tiles - first_mt
                                   : Shape::kSwizzleGroup;
    const uint32_t mt = first_mt + in_group % group_mts;
    const uint32_t nt = in_group / group_mts;

#define MXMM_TILE_GM(kk)                                                 \
  AGlobal aGlobal(                                                       \
      (__gm__ Fp4 *)a_gm +                                               \
          (((uint64_t)mt * Shape::kRows * Shape::kK + (uint64_t)(kk)) >> \
           SHIFT_FP4),                                                   \
      DynShape(Shape::kRows, K_L1));                                     \
  BGlobal bGlobal(                                                       \
      (__gm__ Fp4 *)b_gm +                                               \
          (((uint64_t)nt * Shape::kCols * Shape::kK + (uint64_t)(kk)) >> \
           SHIFT_FP4),                                                   \
      DynShape(K_L1, Shape::kCols));                                     \
  (void)0

#define MXMM_FILL_SCALES()                                                     \
  do {                                                                         \
    AScaleGlobal aScaleGlobal(                                                 \
        (__gm__ E8m0 *)a_scale_gm +                                            \
            (((uint64_t)mt * Shape::kRows * Shape::kK) >> SHIFT_SCALE_FACTOR), \
        DynShape(Shape::kRows, Shape::kScaleK));                               \
    BScaleGlobal bScaleGlobal(                                                 \
        (__gm__ E8m0 *)b_scale_gm +                                            \
            (((uint64_t)nt * Shape::kCols * Shape::kK) >> SHIFT_SCALE_FACTOR), \
        DynShape(Shape::kScaleK, Shape::kCols));                               \
    TLOAD<AScaleTile, AScaleGlobal>(aScaleL1, aScaleGlobal);                   \
    TLOAD<BScaleTile, BScaleGlobal>(bScaleL1, bScaleGlobal);                   \
  } while (0)

#define MXMM_DRAIN_TO_L0(a1, b1, a0, b0, as0, bs0, sub, tile)       \
  do {                                                              \
    const uint16_t koff_ = (uint16_t)((sub) * BASE_K);              \
    const uint16_t soff_ = (uint16_t)((tile) * Shape::kTileScaleK); \
    TEXTRACT(a0, a1, 0, koff_);                                     \
    TEXTRACT(b0, b1, koff_, 0);                                     \
    TEXTRACT(as0, aScaleL1, 0, soff_);                              \
    TEXTRACT(bs0, bScaleL1, soff_, 0);                              \
  } while (0)

#define MXMM_LOAD(sl, sel)                     \
  do {                                         \
    MXMM_TILE_GM((uint64_t)(sl) * K_L1);       \
    if ((sel) == 0u) {                         \
      TLOAD(aL1, aGlobal);                     \
      TLOAD(bL1, bGlobal);                     \
    } else {                                   \
      TLOAD(aL1b, aGlobal);                    \
      TLOAD(bL1b, bGlobal);                    \
    }                                          \
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0); \
  } while (0)

#define MXMM_EXTRACT(e, a0, b0, as0, bs0)                       \
  do {                                                          \
    const uint32_t e_ = (e);                                    \
    const uint32_t slab_ = e_ / Shape::kSubTiles;               \
    const uint32_t sub_ = e_ % Shape::kSubTiles;                \
    if (sub_ == 0u) {                                           \
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);               \
    }                                                           \
    if (slab_ % L1_SETS == 0u) {                                \
      MXMM_DRAIN_TO_L0(aL1, bL1, a0, b0, as0, bs0, sub_, e_);   \
    } else {                                                    \
      MXMM_DRAIN_TO_L0(aL1b, bL1b, a0, b0, as0, bs0, sub_, e_); \
    }                                                           \
    if (sub_ + 1u == Shape::kSubTiles) {                        \
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);                \
    }                                                           \
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);                     \
  } while (0)

    if (scales_held) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    }
    MXMM_FILL_SCALES();
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    scales_held = true;

    for (uint32_t j = 0u; j < LOOKAHEAD && j < Shape::kSlabs; ++j) {
      MXMM_LOAD(j, j % L1_SETS);
    }
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    for (uint32_t kt = 0; kt < Shape::kCubeTiles; ++kt) {
      const bool odd = (kt & 1u) != 0u;
      if (kt % Shape::kSubTiles == 0u) {
        const uint32_t slab_ahead = kt / Shape::kSubTiles + LOOKAHEAD;
        if (slab_ahead < Shape::kSlabs) {
          if (slab_ahead >= L1_SETS) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          }
          MXMM_LOAD(slab_ahead, slab_ahead % L1_SETS);
        }
      }
      if (kt == 0u) {
        MXMM_EXTRACT(0u, aL0, bL0, aScaleL0, bScaleL0);
      }
      if (kt + 1u < Shape::kCubeTiles) {
        if (kt >= 1u) {
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
        }
        if (odd) {
          MXMM_EXTRACT(kt + 1u, aL0, bL0, aScaleL0, bScaleL0);
        } else {
          MXMM_EXTRACT(kt + 1u, aL0b, bL0b, aScaleL0b, bScaleL0b);
        }
      }
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      if (odd) {
        TMATMUL_MX(accL0, accL0, aL0b, aScaleL0b, bL0b, bScaleL0b);
      } else if (kt == 0u) {
        TMATMUL_MX(accL0, aL0, aScaleL0, bL0, bScaleL0);
      } else {
        TMATMUL_MX(accL0, accL0, aL0, aScaleL0, bL0, bScaleL0);
      }
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    }
    for (uint32_t d = 0u; d < L1_SETS && d < Shape::kSlabs; ++d) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    }
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    if constexpr (Shape::kCubeTiles > 1u) {
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    }
#undef MXMM_TILE_GM
#undef MXMM_FILL_SCALES
#undef MXMM_DRAIN_TO_L0
#undef MXMM_LOAD
#undef MXMM_EXTRACT
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    OutGlobal outGlobal((__gm__ bfloat16_t *)out_gm +
                        ((uint64_t)mt * Shape::kRows * Shape::kN +
                         (uint64_t)nt * Shape::kCols));
    TSTORE(outGlobal, accL0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  }

  if (scales_held) {
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  }
}
#endif

inline uint32_t pickMTile(uint32_t m) {
  if (m % TILE_M_BIG == 0u &&
      (m / TILE_M_BIG) * (TEST_N / TILE_N) >= TARGET_BLOCKS) {
    return TILE_M_BIG;
  }
  if (m % TILE_M == 0u) {
    return TILE_M;
  }
  return TILE_M_MIN;
}

inline uint32_t pickMRound(uint32_t m) {
  if (m == 0u) return TILE_M_MIN;
  if (m <= TILE_M_MIN) return TILE_M_MIN;
  return ((m + TILE_M - 1u) / TILE_M) * TILE_M;
}

}  // namespace

template <unsigned kMMax, unsigned kK, unsigned kN, unsigned kBaseM,
          unsigned kBaseN>
__global__ AICORE void mxfp4_matmul(__gm__ void *a_gm, __gm__ void *a_scale_gm,
                                    __gm__ void *b_gm, __gm__ void *b_scale_gm,
                                    __gm__ void *out_gm, uint32_t m_total,
                                    uint32_t core_count) {
#if defined(__DAV_CUBE__)
  runMxfp4Matmul<TileShape<kMMax, kK, kN, kBaseM, kBaseN>>(
      a_gm, a_scale_gm, b_gm, b_scale_gm, out_gm, m_total, core_count);
#else
  (void)a_gm;
  (void)a_scale_gm;
  (void)b_gm;
  (void)b_scale_gm;
  (void)out_gm;
  (void)m_total;
  (void)core_count;
#endif
}

namespace {

inline void launchMatmul(uint32_t blockDim, void *stream, uint8_t *a_gm,
                         uint8_t *a_scale_gm, uint8_t *b_gm,
                         uint8_t *b_scale_gm, uint8_t *out_gm, uint32_t m,
                         uint32_t k, uint32_t n) {
  // The host must reject what has no instantiation: returning quietly hands
  // the caller an untouched output buffer back.
  if (blockDim == 0u) return;
  if (k != TEST_K || n != TEST_N) return;
  if (m == 0u || m % TILE_M_MIN != 0u || m > TEST_M_MAX) return;
  // NEVER guard this launch with a device-pass macro.
  const uint32_t tile = pickMTile(m);
  if (tile == TILE_M_BIG) {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, TILE_M_BIG, TILE_N>
        <<<blockDim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                        out_gm, m, blockDim);
  } else if (tile == TILE_M) {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, TILE_M, TILE_N>
        <<<blockDim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                        out_gm, m, blockDim);
  } else {
    mxfp4_matmul<TEST_M_MAX, TEST_K, TEST_N, TILE_M_MIN, TILE_N>
        <<<blockDim, nullptr, stream>>>(a_gm, a_scale_gm, b_gm, b_scale_gm,
                                        out_gm, m, blockDim);
  }
}

}  // namespace

extern "C" void call_mxfp4_matmul(uint32_t blockDim, void *stream,
                                  uint8_t *a_gm, uint8_t *a_scale_gm,
                                  uint8_t *b_gm, uint8_t *b_scale_gm,
                                  uint8_t *out_gm, uint32_t m, uint32_t k,
                                  uint32_t n) {
  launchMatmul(blockDim, stream, a_gm, a_scale_gm, b_gm, b_scale_gm, out_gm, m,
               k, n);
}

extern "C" uint32_t mxfp4_matmul_m_tile() { return TILE_M_MIN; }
extern "C" uint32_t mxfp4_matmul_m_round(uint32_t m) { return pickMRound(m); }
extern "C" uint32_t mxfp4_matmul_m_tile_for(uint32_t m) { return pickMTile(m); }
extern "C" uint32_t mxfp4_matmul_n_tile_for(uint32_t m) {
  (void)m;  // all three tiles share one N
  return TILE_N;
}
extern "C" uint32_t mxfp4_matmul_m_max() { return TEST_M_MAX; }
extern "C" uint32_t mxfp4_matmul_k() { return TEST_K; }
extern "C" uint32_t mxfp4_matmul_n() { return TEST_N; }
