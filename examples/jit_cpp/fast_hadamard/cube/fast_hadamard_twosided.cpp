#include <pto/pto-inst.hpp>

using namespace pto;

// Radix-sqrt(N) Hadamard transform on the cube unit, as two matrix products.
//
// H_N = H_R (x) H_C for N = R * C in Sylvester (natural) ordering, so for a
// sample read as an R x C row-major matrix X, Y = H_R * X * H_C read back
// row-major is the length-N transform in natural order.
//
// A tile is S consecutive samples, in GM one contiguous (S*R x C) matrix.
// Neither multiply mixes the other's axis, so two TMATMULs transform the tile
// whatever S is. Between them the per-sample blocks move from a vertical stack
// to a horizontal one: free at C == 16 (one NZ block-column, the same bytes),
// else S TEXTRACTs into one (R x S*C) L1 tile plus S per-sample stores.
//
// Both GM legs may read through a Layout::NZ view (Layout::kNzGm), one
// contiguous burst instead of the ND2NZ gather. That is correct at C == 16,
// where the NZ image IS the row-major bytes, and at S == 1, where ND and NZ
// differ by a permutation of the BIT FIELDS of the byte index, under which
// H[i][j] = (-1)^popcount(i & j) is invariant, so an NZ load and an NZ store
// cancel. S > 1 at C > 16 keeps the gather; buying the burst there costs more
// MACs than the gather costs, measured.
//
// Every tuning constant below is a measured value rather than a derived
// one -- resweep it after any change to the tile geometry or the case
// table.

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

// Samples per tile at N=256. S*N <= 16384 keeps both L0C accumulators inside
// 128 KB; the rest of the table is fixed by the same bound.
#ifndef PTO_TS_S256
#define PTO_TS_S256 32
#endif

// The S each size carries below is its TALL tile. A tile is S samples, so the
// tile height caps usable cores at ceil(batch / S), and below that cap the tall
// tile is the wrong one. Every size therefore carries S, S/2 and S/4 and picks
// between them on the tile count the batch yields: S above PTO_TS_TILES_TALL
// tall tiles, else S/2 above PTO_TS_TILES_MID mid tiles, else S/4. Sizes where
// a level repeats a value compile the pick away with `if constexpr`.
#ifndef PTO_TS_TILES_TALL
#define PTO_TS_TILES_TALL 16
#endif
#ifndef PTO_TS_TILES_MID
// Not the same count as the upper crossover, which is why one constant cannot
// serve both levels.
#define PTO_TS_TILES_MID 6
#endif

// Apply a stage's factor as H_f = H_acc (x) H_mat with H_mat of size M, so the
// matmul is M-sized and the sum over blocks runs in the L0C accumulator. Same
// MAC count, smaller per-matmul slot, and slower -- used only where there is no
// dense form at all: N=32768, whose H_C alone fills L0B. M must divide its
// factor and be a multiple of 16, the cube's fractal size.
#ifndef PTO_TS_BLOCK
#define PTO_TS_BLOCK 64
#endif

#define FAST_HADAMARD_TWOSIDED_CASES(X) \
  X(256, 16, 16, PTO_TS_S256)           \
  X(512, 32, 16, 16)                    \
  X(1024, 64, 16, 8)                    \
  X(2048, 32, 64, 4)                    \
  X(4096, 64, 64, 2)                    \
  X(8192, 64, 128, 1)                   \
  X(16384, 128, 128, 1)                 \
  X(32768, 128, 256, 1)

#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

namespace {

constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t L0AB_BYTES = 64 * 1024;
constexpr uint32_t L0C_BYTES = 128 * 1024;

// Distinct ids for the cross-tile edges; the intra-tile Sync pairs use id 0.
// Macros, not constexpr: bisheng accepts only a literal EVENT_ID* token or a
// template non-type argument as the third set_flag / wait_flag argument, which
// is also why Slot and parity are template arguments throughout.
//
// A credit on (src pipe, dst pipe, id) says either that data has landed or that
// a resource is free again. One id must never carry two set_flag channels out
// of the same SOURCE pipe, or the wrong waiter takes a mid-loop credit. The
// shallow path, one id per channel:
//
//   EV_X0/X1   MTE2->MTE1  an L1 X slot is filled
//              MTE1->MTE2  that same slot is free again
//   EV_ACC     FIX->M      the output accumulator is free, the stores read it
//   EV_SWB     M->MTE1     the left multiply's L1 T slice is free
//   EV_SWA0/1  M->MTE1     the right multiply's L0A slot is free
//   EV_SRA0/1  MTE1->M     that same L0A slot is staged
#define EV_X0 EVENT_ID1
#define EV_X1 EVENT_ID2
#define EV_ACC EVENT_ID3
// Shallow blocked bodies. EV_SWB shares id 3 with EV_ACC, but leaves M rather
// than FIX, so they are different channels.
#define EV_SWB EVENT_ID3
#define EV_SWA0 EVENT_ID4
#define EV_SWA1 EVENT_ID5
#define EV_SRA0 EVENT_ID6
#define EV_SRA1 EVENT_ID7

// The deep ring, read the same way. Phase A's id carries three channels and
// phase B's two, each out of a different source pipe, so the rule holds. The
// suffix is the tile parity -- two tiles are in flight.
//
//   EV_D0..3   MTE2->MTE1  a ring slot is filled
//              MTE1->MTE2  that same slot is free again
//   EV_PA0/1   MTE1->M     the X tile is staged in L0A
//              M->FIX      the accumulator holds X * H_C
//              FIX->MTE1   T is parked in L1
//   EV_PB0/1   MTE1->M     T is staged in L0B
//              M->FIX      the accumulator holds H_R * T
#define EV_D0 EVENT_ID0
#define EV_D1 EVENT_ID1
#define EV_D2 EVENT_ID2
#define EV_D3 EVENT_ID3
#define EV_PA0 EVENT_ID4
#define EV_PA1 EVENT_ID5
#define EV_PB0 EVENT_ID6
#define EV_PB1 EVENT_ID7

// Intra-tile sequencing. Id 0 is correct because every Sync below sits in a
// shallow body or in loadHadamardFactors, and those spend ids 1-7 only.
#define EV_SEQ EVENT_ID0

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void Sync(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

// The only Mat layout TLOAD / TMOV / TEXTRACT accept on A2/A3.
template <uint32_t Rows, uint32_t Cols>
using MatTile = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor, Rows,
                     Cols, SLayout::RowMajor, 512>;

// NZ view of a tile: block-column major over 16x16 row-major fractals.
template <uint32_t Rows, uint32_t Cols>
using GmNz =
    GlobalTensor<half, TileShape2D<half, Rows, Cols, Layout::NZ>,
                 BaseShape2D<half, Rows, Cols, Layout::NZ>, Layout::NZ>;

template <uint32_t Rows, uint32_t Cols>
using GmTile =
    GlobalTensor<half, TileShape2D<half, Rows, Cols, Layout::ND>,
                 BaseShape2D<half, Rows, Cols, Layout::ND>, Layout::ND>;

// A tile's Rows is a template parameter but the TRANSFERS around it are not,
// so a full-size tile can load and store fewer rows than it computes -- which
// is how one masked ring tile carries a core's 1..S-1 sample remainder. A row
// count is S' * R, so it stays 16-aligned whatever the sample count is.
template <uint32_t Rows, uint32_t Cols, int RowValid>
using MatTileV = Tile<TileType::Mat, half, Rows, Cols, BLayout::ColMajor,
                      RowValid, Cols, SLayout::RowMajor, 512>;

template <uint32_t Rows, uint32_t Cols>
using GmTileDyn =
    GlobalTensor<half, TileShape2D<half, DYNAMIC, Cols, Layout::ND>,
                 BaseShape2D<half, Rows, Cols, Layout::ND>, Layout::ND>;

// Masked NZ view, so a short tile keeps the burst. The two layouts index their
// shape differently: ND is Shape<1,1,1,rows,cols> so the row count is DIM_3,
// while NZ is Shape<1,cols/C0,rows/16,16,C0> so it is DIM_2 and counts
// FRACTALS.
template <uint32_t Rows, uint32_t Cols>
using GmNzDyn =
    GlobalTensor<half, TileShape2D<half, DYNAMIC, Cols, Layout::NZ>,
                 BaseShape2D<half, Rows, Cols, Layout::NZ>, Layout::NZ>;

// Buffer map. The factor addresses must not depend on S: the factors are
// loaded once and reused by every tile size the dispatch picks.
template <uint32_t R, uint32_t C, uint32_t S>
struct Layout {
  static constexpr uint32_t kN = R * C;
  static constexpr uint32_t kStackRows = S * R;
  static constexpr uint32_t kTileElems = S * R * C;
  static constexpr unsigned kTileBytes = kTileElems * sizeof(half);
  static constexpr unsigned kAccBytes = kTileElems * sizeof(float);

  // Does the DENSE SHALLOW path fit? That is the fallback every size has, so
  // it -- not the deep path -- decides whether blocking is needed at all.
  static constexpr bool kDenseShallowFits =
      R * R * sizeof(half) + kTileBytes <= L0AB_BYTES &&
      C * C * sizeof(half) + kTileBytes <= L0AB_BYTES &&
      2 * kAccBytes <= L0C_BYTES;
  static constexpr bool kBlockIt = !kDenseShallowFits;

  // Both GM legs use the NZ view or neither -- ONE predicate read by every load
  // and store site, because at C > 16 the two only cancel as a pair. Never on
  // the blocked path, whose stores are row SUB-BLOCKS: TStoreAccNz2nz takes the
  // block-column stride from the shape, so it can only write a whole NZ tile.
  static constexpr bool kNzGm = (C == 16) || (S == 1 && !kBlockIt);

  static constexpr uint32_t kMa =
      (!kBlockIt || PTO_TS_BLOCK == 0 || PTO_TS_BLOCK >= R)
          ? R
          : (uint32_t)PTO_TS_BLOCK;
  static constexpr uint32_t kMb =
      (!kBlockIt || PTO_TS_BLOCK == 0 || PTO_TS_BLOCK >= C)
          ? C
          : (uint32_t)PTO_TS_BLOCK;
  // Blocks of each factor, and whether that makes a stage blocked: phase A is
  // the right multiply by H_C, phase B the left multiply by H_R.
  static constexpr uint32_t kR1 = R / kMa;
  static constexpr uint32_t kC1 = C / kMb;
  static_assert(kR1 * kMa == R && kC1 * kMb == C, "M must divide its factor");
  static_assert(kMa % 16 == 0 && kMb % 16 == 0, "M must be 16-aligned");
  static constexpr bool kBlkA = (kC1 > 1);
  static constexpr bool kBlkB = (kR1 > 1);

  // R == C makes H_R and H_C the SAME Sylvester matrix, so ONE L1 copy feeds
  // both L0A and L0B and the second GM read goes away. H is symmetric, so the
  // left operand needs no transpose. Unblocked only: a blocked side reads
  // +-H_M from its own offset instead.
  static constexpr bool kShareHf = (R == C) && !kBlkA && !kBlkB;

  // At C == 16 each sample's (R x 16) block is ONE NZ block-column of R/16
  // fractals, so the vertical stack of S samples IS their horizontal
  // concatenation, for ANY R. That is what lets phase B read the turnaround
  // "wide" as (R x S*C) and store the tile whole.
  static_assert(C != 16 || R % 16 == 0, "C == 16 needs a 16-aligned R");

  // Only where blocking is active, so every unblocked size keeps its offsets.
  static constexpr bool kAlias = (kC1 > 1 || kR1 > 1);

  // Operands, resident for the whole kernel: +-H_M when blocked, else H itself.
  static constexpr unsigned kOpABytes =
      kBlkB ? 2 * kMa * kMa * sizeof(half) : R * R * sizeof(half);
  static constexpr unsigned kOpBBytes =
      kBlkA ? 2 * kMb * kMb * sizeof(half) : C * C * sizeof(half);
  // Per-matmul data slices: phase A's lives in L0A, phase B's in L0B.
  static constexpr unsigned kSliceA = S * R * kMb * sizeof(half);
  static constexpr unsigned kSliceB = kMa * S * C * sizeof(half);

  // Two L0A slices let MTE1 stage block i+1 under M's block i. Phase A only:
  // two of phase B's wider slices plus the operand overflow L0B.
  static constexpr uint32_t kSlotsA =
      (kBlkA && kOpABytes + 2 * kSliceA <= L0AB_BYTES) ? 2u : 1u;

  // Two of every intermediate plus a 4-deep X ring. `!kBlockIt` is
  // load-bearing: the deep path is dense, so the sizes measured here are too.
  static constexpr bool kDbuf =
      !kBlockIt && R * R * sizeof(half) + 2 * kTileBytes <= L0AB_BYTES &&
      C * C * sizeof(half) + 2 * kTileBytes <= L0AB_BYTES &&
      4 * kAccBytes <= L0C_BYTES &&
      (R * R + C * C) * sizeof(half) + 6 * kTileBytes <= L1_BYTES;

  static constexpr unsigned kXSlots = kDbuf ? 4 : 2;
  static constexpr unsigned kBuf = kDbuf ? 2 : 1;

  // Cannot fire today -- M is clamped to the factor at C == 16 -- and is here
  // to catch a future case-table entry that breaks it.
  static_assert(!(kC1 > 1 || kR1 > 1) || C != 16,
                "the C == 16 bodies are dense and would read +-H_M as H");

  // L1: the two factors, the X ring, then the T turnaround slots.

  static constexpr unsigned kL1Hl = 0;
  static constexpr unsigned kL1Hc = kL1Hl + kOpABytes;
  static constexpr unsigned kL1X = kL1Hc + kOpBBytes;
  static constexpr unsigned kL1XSlot = kTileBytes;
  static constexpr unsigned kL1T = kL1X + kXSlots * kL1XSlot;
  static_assert(kL1T + kBuf * kTileBytes <= L1_BYTES, "L1 overflow");

  static constexpr unsigned xSlot(uint32_t slot) {
    return kL1X + slot * kL1XSlot;
  }
  static constexpr unsigned l1T(bool par) {
    return kL1T + (par ? kTileBytes : 0);
  }

  // L0A: H_R, then the stacked X tile
  static constexpr unsigned kL0aHl = 0;
  static constexpr unsigned kL0aX = kL0aHl + kOpABytes;
  static constexpr unsigned kL0aSlot = (C == 16) ? kTileBytes : kSliceA;
  static_assert(kL0aX + kBuf * kL0aSlot <= L0AB_BYTES, "L0A overflow");
  static constexpr unsigned l0aX(bool par) {
    return kL0aX + (par ? kL0aSlot : 0);
  }
  // Operand slot: 0 is +H_Ma, 1 is -H_Ma (and 0 is H_R when dense).
  static constexpr unsigned l0aOp(uint32_t neg) {
    return kL0aHl + (kBlkB ? neg * kMa * kMa * sizeof(half) : 0u);
  }

  // L0B: H_C, then the concatenated T tile
  static constexpr unsigned kL0bHc = 0;
  static constexpr unsigned kL0bT = kL0bHc + kOpBBytes;
  static constexpr unsigned kL0bSlot = (C == 16) ? kTileBytes : kSliceB;
  static_assert(kL0bT + kBuf * kL0bSlot <= L0AB_BYTES, "L0B overflow");
  static constexpr unsigned l0bT(bool par) {
    return kL0bT + (par ? kL0bSlot : 0);
  }
  static constexpr unsigned l0bOp(uint32_t neg) {
    return kL0bHc + (kBlkA ? neg * kMb * kMb * sizeof(half) : 0u);
  }

  // L0C: the stacked accumulator then the concatenated one, as a pair, so the
  // deep path's two tiles in flight get [stack|cat][stack|cat].
  static constexpr unsigned kL0cStack = 0;
  static constexpr unsigned kL0cCat = kL0cStack + kAccBytes;
  // With kAlias the cat accumulator shares the stack one's space, halving L0C.
  static constexpr unsigned kAccPair = (kAlias ? 1u : 2u) * kAccBytes;
  static_assert(kBuf * kAccPair <= L0C_BYTES, "L0C overflow");
  static constexpr unsigned l0cStack(bool par) { return (par ? kAccPair : 0); }
  static constexpr unsigned l0cCat(bool par) {
    return (par ? kAccPair : 0) + (kAlias ? 0u : kAccBytes);
  }
  // Output block a of the blocked phase B: an L0C tile is block-COLUMN-major,
  // so kR1 separate (Ma x S*C) accumulators are contiguous where a row slice of
  // one tall tile would be strided.
  static constexpr unsigned l0cCatBlock(bool par, uint32_t a) {
    return l0cCat(par) + a * kMa * S * C * sizeof(float);
  }
  // Column block j of phase A's accumulator: whole 16-column groups, which are
  // contiguous in an L0C tile.
  static constexpr unsigned l0cStackBlock(bool par, uint32_t j) {
    return l0cStack(par) + j * kMb * kStackRows * sizeof(float);
  }
};

// GM -> L1 -> L0A/L0B for the two Hadamard factors, once per core; both stay
// resident. Both legs read through Layout::NZ and the HOST stores the operands
// pre-fractalised to match (see _to_nz); an ND read of a (W x W) factor is a
// 2W-byte-per-row gather that every core pays before its first tile.
//
// S is a template parameter although no address here depends on it: the
// blocking decision does, and loading the factors under a different decision
// from the tile bodies' would stage dense H where they expect +-H_M.
template <uint32_t R, uint32_t C, uint32_t S>
AICORE void loadHadamardFactors(__gm__ half *h_left, __gm__ half *h_right) {
  using L = Layout<R, C, S>;

  // Each buffer is H_f followed by +H_M and -H_M (see _factor_buffer on the
  // host); a stage reads one or the other, never both.
  if constexpr (L::kBlkB) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      MatTile<L::kMa, L::kMa> hl_l1;
      TASSIGN(hl_l1, L::kL1Hl + neg * L::kMa * L::kMa * sizeof(half));
      __gm__ half *src = h_left + R * R + neg * L::kMa * L::kMa;
      GmNz<L::kMa, L::kMa> hl_gm(src);
      TASSIGN(hl_gm, src);
      TLOAD(hl_l1, hl_gm);
    }
  } else {
    MatTile<R, R> hl_l1;
    TASSIGN(hl_l1, L::kL1Hl);
    GmNz<R, R> hl_gm(h_left);
    TASSIGN(hl_gm, h_left);
    TLOAD(hl_l1, hl_gm);
  }
  if constexpr (L::kBlkA) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      MatTile<L::kMb, L::kMb> hc_l1;
      TASSIGN(hc_l1, L::kL1Hc + neg * L::kMb * L::kMb * sizeof(half));
      __gm__ half *src = h_right + C * C + neg * L::kMb * L::kMb;
      GmNz<L::kMb, L::kMb> hc_gm(src);
      TASSIGN(hc_gm, src);
      TLOAD(hc_l1, hc_gm);
    }
  } else if constexpr (!L::kShareHf) {
    MatTile<C, C> hc_l1;
    TASSIGN(hc_l1, L::kL1Hc);
    GmNz<C, C> hc_gm(h_right);
    TASSIGN(hc_gm, h_right);
    TLOAD(hc_l1, hc_gm);
  }
  Sync<PIPE_MTE2, PIPE_MTE1>(EV_SEQ);

  if constexpr (L::kBlkB) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      MatTile<L::kMa, L::kMa> hl_l1;
      TileLeft<half, L::kMa, L::kMa> hl_l0;
      TASSIGN(hl_l1, L::kL1Hl + neg * L::kMa * L::kMa * sizeof(half));
      TASSIGN(hl_l0, L::l0aOp(neg));
      TMOV(hl_l0, hl_l1);
    }
  } else {
    MatTile<R, R> hl_l1;
    TileLeft<half, R, R> hl_l0;
    TASSIGN(hl_l1, L::kL1Hl);
    TASSIGN(hl_l0, L::kL0aHl);
    TMOV(hl_l0, hl_l1);
  }
  if constexpr (L::kBlkA) {
    for (uint32_t neg = 0; neg < 2; ++neg) {
      MatTile<L::kMb, L::kMb> hc_l1;
      TileRight<half, L::kMb, L::kMb> hc_l0;
      TASSIGN(hc_l1, L::kL1Hc + neg * L::kMb * L::kMb * sizeof(half));
      TASSIGN(hc_l0, L::l0bOp(neg));
      TMOV(hc_l0, hc_l1);
    }
  } else {
    MatTile<C, C> hc_l1;
    TileRight<half, C, C> hc_l0;
    TASSIGN(hc_l1, L::kShareHf ? L::kL1Hl : L::kL1Hc);
    TASSIGN(hc_l0, L::kL0bHc);
    TMOV(hc_l0, hc_l1);
  }
  Sync<PIPE_MTE1, PIPE_M>(EV_SEQ);
}

// One column block of the blocked shallow right multiply.
template <uint32_t R, uint32_t C, uint32_t S, uint32_t Slot, uint32_t P,
          event_t EvRdy, event_t EvWar>
AICORE void shallowBlockA(uint32_t i);

// H_f[i,j] = (-1)^popcount(i & j) for the Sylvester matrix, so a block's sign
// is a parity and H_acc itself is never built or read. Returns 1 for -H_M and
// 0 for +H_M, which are also their operand slot indices.
AICORE inline uint32_t blockSign(uint32_t i, uint32_t j) {
  uint32_t v = i & j;
  v ^= v >> 4;
  v ^= v >> 2;
  v ^= v >> 1;
  return v & 1u;
}

// GM -> L1 for one tile, signalled MTE2->MTE1 on EvX; `waitSlot` takes the
// slot's free-again credit first.
template <uint32_t R, uint32_t C, uint32_t S, uint32_t Slot, event_t EvX>
AICORE void issueTileLoad(__gm__ half *x, uint32_t sample, uint32_t rows,
                          bool waitSlot) {
  using L = Layout<R, C, S>;

  if (waitSlot) wait_flag(PIPE_MTE1, PIPE_MTE2, EvX);

  __gm__ half *x_tile = x + sample * L::kN;

  // At most ONE tile per core is ever short, so this runs once a launch.
  if constexpr (S > 1) {
    if (rows != S) {
      // Masked, but on the SAME layout the full tile uses.
      MatTileV<L::kStackRows, C, DYNAMIC> x_l1;
      TASSIGN(x_l1, L::xSlot(Slot));
      x_l1.SetValidRow(rows * R);
      if constexpr (L::kNzGm) {
        GmNzDyn<L::kStackRows, C> x_gm(x_tile);
        TASSIGN(x_gm, x_tile);
        x_gm.template SetShape<GlobalTensorDim::DIM_2>(rows * R / 16);
        TLOAD(x_l1, x_gm);
      } else {
        GmTileDyn<L::kStackRows, C> x_gm(x_tile);
        TASSIGN(x_gm, x_tile);
        x_gm.template SetShape<GlobalTensorDim::DIM_3>(rows * R);
        TLOAD(x_l1, x_gm);
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, EvX);
      return;
    }
  }

  MatTile<L::kStackRows, C> x_l1;
  TASSIGN(x_l1, L::xSlot(Slot));

  if constexpr (L::kNzGm) {
    GmNz<L::kStackRows, C> x_gm(x_tile);
    TASSIGN(x_gm, x_tile);
    TLOAD(x_l1, x_gm);
  } else {
    GmTile<L::kStackRows, C> x_gm(x_tile);
    TASSIGN(x_gm, x_tile);
    TLOAD(x_l1, x_gm);
  }
  set_flag(PIPE_MTE2, PIPE_MTE1, EvX);
}

// Y[sample .. sample+S) = H_R * X * H_C, for a tile already loaded into slot
// Pong.
template <uint32_t R, uint32_t C, uint32_t S, bool Pong, event_t EvX>
AICORE void hadamardTile(__gm__ half *y, uint32_t sample, bool freeSlot,
                         bool waitAcc, bool freeAcc) {
  using L = Layout<R, C, S>;
  constexpr uint32_t kStackRows = L::kStackRows;

  MatTile<kStackRows, C> x_l1;
  TASSIGN(x_l1, L::xSlot(Pong ? 1u : 0u));
  wait_flag(PIPE_MTE2, PIPE_MTE1, EvX);

  // Every C == 16 row has kDbuf and takes the deep ring, so the cat body lives
  // only in tilePhaseA/tilePhaseB and adding such a size must not compile.
  static_assert(C != 16, "a shallow C == 16 size would need the cat body here");
  if constexpr (L::kBlkA || L::kBlkB) {
    // Blocked shallow body: the same two multiplies as below, applied as blocks
    // against a resident +-H_M so neither factor is ever in L0A or L0B whole.
    constexpr uint32_t kSlot = Pong ? 1u : 0u;

    if constexpr (L::kAlias) {
      // The two accumulators share one L0C range here, so the previous tile's
      // stores must drain before this tile's FIRST matmul.
      if (waitAcc) wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
    }

    // --- right multiply: X * H_C, as kC1 column blocks summed in L0C ---
    for (uint32_t i = 0; i < L::kC1; ++i) {
      if constexpr (L::kSlotsA == 2) {
        if (i & 1u) {
          shallowBlockA<R, C, S, kSlot, 1, EV_SRA1, EV_SWA1>(i);
        } else {
          shallowBlockA<R, C, S, kSlot, 0, EV_SRA0, EV_SWA0>(i);
        }
      } else {
        shallowBlockA<R, C, S, kSlot, 0, EV_SRA0, EV_SWA0>(i);
      }
    }
    if (freeSlot) set_flag(PIPE_MTE1, PIPE_MTE2, EvX);
    Sync<PIPE_M, PIPE_FIX>(EV_SEQ);

    // --- vertical stack -> horizontal concatenation, in L1 ---
    // Grouped by the left multiply's ROW blocks, so each of its kR1 slices is
    // contiguous.
    TileAcc<float, kStackRows, C> acc_stack;
    TASSIGN(acc_stack, L::l0cStack(false));
    for (uint32_t a = 0; a < L::kR1; ++a) {
      for (uint32_t sm = 0; sm < S; ++sm) {
        MatTile<L::kMa, C> t_block;
        TASSIGN(t_block, L::kL1T + (a * S + sm) * L::kMa * C * sizeof(half));
        TEXTRACT(t_block, acc_stack, sm * R + a * L::kMa, 0);
      }
    }
    Sync<PIPE_FIX, PIPE_MTE1>(EV_SEQ);

    // --- left multiply: H_R * T, as kR1 row blocks summed in L0C ---
    // Under kAlias this writes over acc_stack, which the extracts have drained;
    // M is ordered behind them transitively, FIX -> MTE1 by the Sync just made
    // then MTE1 -> M by the one inside the loop.
    TileRight<half, L::kMa, S * C> t_l0;
    TASSIGN(t_l0, L::kL0bT);
    for (uint32_t ap = 0; ap < L::kR1; ++ap) {
      MatTile<L::kMa, S * C> t_l1_ap;
      TASSIGN(t_l1_ap, L::kL1T + ap * S * L::kMa * C * sizeof(half));
      if (ap > 0) {
        wait_flag(PIPE_M, PIPE_MTE1, EV_SWB);
      }
      TMOV(t_l0, t_l1_ap);
      Sync<PIPE_MTE1, PIPE_M>(EV_SEQ);
      if constexpr (!L::kAlias) {
        if (ap == 0) {
          if (waitAcc) wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
        }
      }
      for (uint32_t a = 0; a < L::kR1; ++a) {
        TileAcc<float, L::kMa, S * C> acc_a;
        TASSIGN(acc_a, L::l0cCatBlock(false, a));
        TileLeft<half, L::kMa, L::kMa> h_a;
        TASSIGN(h_a, L::l0aOp(blockSign(ap, a)));
        if (ap == 0) {
          TMATMUL(acc_a, h_a, t_l0);
        } else {
          TMATMUL_ACC(acc_a, h_a, t_l0);
        }
      }
      if (ap + 1 < L::kR1) {
        set_flag(PIPE_M, PIPE_MTE1, EV_SWB);
      }
    }
    Sync<PIPE_M, PIPE_FIX>(EV_SEQ);

    // Row sub-blocks, so these must stay ND whatever C is -- see kNzGm.
    for (uint32_t a = 0; a < L::kR1; ++a) {
      for (uint32_t sm = 0; sm < S; ++sm) {
        TileAcc<float, L::kMa, C> acc_block;
        TASSIGN(acc_block,
                L::l0cCatBlock(false, a) + sm * L::kMa * C * sizeof(float));
        __gm__ half *y_row = y + (sample + sm) * L::kN + a * L::kMa * C;
        GmTile<L::kMa, C> y_gm(y_row);
        TASSIGN(y_gm, y_row);
        TSTORE(y_gm, acc_block);
      }
    }
    if (freeAcc) set_flag(PIPE_FIX, PIPE_M, EV_ACC);
  } else {
    // --- right multiply: X * H_C, S samples stacked vertically ---
    TileLeft<half, kStackRows, C> x_l0;
    TileRight<half, C, C> hc_l0;
    TileAcc<float, kStackRows, C> acc_stack;
    TASSIGN(x_l0, L::kL0aX);
    TASSIGN(hc_l0, L::kL0bHc);
    TASSIGN(acc_stack, L::kL0cStack);

    TMOV(x_l0, x_l1);
    if (freeSlot) set_flag(PIPE_MTE1, PIPE_MTE2, EvX);
    Sync<PIPE_MTE1, PIPE_M>(EV_SEQ);
    TMATMUL(acc_stack, x_l0, hc_l0);
    Sync<PIPE_M, PIPE_FIX>(EV_SEQ);

    // --- vertical stack -> horizontal concatenation, in L1 ---
    // Block-columns are contiguous in NZ, so sample s's (R x C) result lands at
    // s * R * C and the S writes read back as one (R x S*C) tile.
    for (uint32_t s = 0; s < S; ++s) {
      MatTile<R, C> t_block;
      TASSIGN(t_block, L::kL1T + s * R * C * sizeof(half));
      TEXTRACT(t_block, acc_stack, s * R, 0);
    }
    Sync<PIPE_FIX, PIPE_MTE1>(EV_SEQ);

    // --- left multiply: H_R * T, S samples side by side ---
    MatTile<R, S * C> t_l1;
    TileRight<half, R, S * C> t_l0;
    TASSIGN(t_l1, L::kL1T);
    TASSIGN(t_l0, L::kL0bT);

    TMOV(t_l0, t_l1);
    Sync<PIPE_MTE1, PIPE_M>(EV_SEQ);

    TileLeft<half, R, R> hl_l0;
    TileAcc<float, R, S * C> acc_cat;
    TASSIGN(hl_l0, L::kL0aHl);
    TASSIGN(acc_cat, L::kL0cCat);

    if (waitAcc) wait_flag(PIPE_FIX, PIPE_M, EV_ACC);
    TMATMUL(acc_cat, hl_l0, t_l0);
    Sync<PIPE_M, PIPE_FIX>(EV_SEQ);

    // Each sample is a contiguous (R x C) block-column range of acc_cat, but
    // the fixpipe L0C->GM path is 2D-to-2D only, so they cannot be folded into
    // one strided descriptor -- a 5D scatter faults the aicore.
    for (uint32_t s = 0; s < S; ++s) {
      TileAcc<float, R, C> acc_block;
      TASSIGN(acc_block, L::kL0cCat + s * R * C * sizeof(float));

      __gm__ half *y_row = y + (sample + s) * L::kN;
      if constexpr (L::kNzGm) {
        GmNz<R, C> y_gm(y_row);
        TASSIGN(y_gm, y_row);
        TSTORE(y_gm, acc_block);
      } else {
        GmTile<R, C> y_gm(y_row);
        TASSIGN(y_gm, y_row);
        TSTORE(y_gm, acc_block);
      }
    }
    if (freeAcc) set_flag(PIPE_FIX, PIPE_M, EV_ACC);
  }
}

template <uint32_t R, uint32_t C, uint32_t S, uint32_t Slot, uint32_t P,
          event_t EvRdy, event_t EvWar>
AICORE void shallowBlockA(uint32_t i) {
  using L = Layout<R, C, S>;
  constexpr uint32_t kStackRows = L::kStackRows;

  TileLeft<half, kStackRows, L::kMb> x_l0;
  TASSIGN(x_l0, L::kL0aX + P * L::kSliceA);
  MatTile<kStackRows, L::kMb> x_l1_i;
  TASSIGN(x_l1_i, L::xSlot(Slot) + i * L::kMb * kStackRows * sizeof(half));

  // Only the block kSlotsA back shared this slot, so with two slots MTE1
  // stages block i+1 while M is still on block i.
  if (i >= L::kSlotsA) {
    wait_flag(PIPE_M, PIPE_MTE1, EvWar);
  }
  TMOV(x_l0, x_l1_i);
  set_flag(PIPE_MTE1, PIPE_M, EvRdy);

  wait_flag(PIPE_MTE1, PIPE_M, EvRdy);
  for (uint32_t j = 0; j < L::kC1; ++j) {
    TileAcc<float, kStackRows, L::kMb> acc_j;
    TASSIGN(acc_j, L::l0cStackBlock(false, j));
    TileRight<half, L::kMb, L::kMb> h_j;
    TASSIGN(h_j, L::l0bOp(blockSign(i, j)));
    if (i == 0) {
      TMATMUL(acc_j, x_l0, h_j);
    } else {
      TMATMUL_ACC(acc_j, x_l0, h_j);
    }
  }
  if (i + L::kSlotsA < L::kC1) {
    set_flag(PIPE_M, PIPE_MTE1, EvWar);
  }
}

// --- deep pipeline --------------------------------------------------------
//
// The straight-line tile above, split at the L1 turnaround. Phase A takes the
// loaded X slot to L0A, right-multiplies by H_C and parks the result in L1;
// phase B takes it to L0B, left-multiplies by H_R and stores. The driver issues
// A(i+1) before B(i), so each pipe always has the next tile's work queued
// behind the current one's.
//
// With two slots per intermediate and that issue order, every cross-tile WAR
// edge is already covered transitively -- A(i+2)'s reuse of a slot sits behind
// B(i)'s MTE1, which waits on A(i)'s FIX -- so this path needs no WAR flags and
// no EV_ACC.

// X slot -> L0A, X * H_C -> L0C, L0C -> L1 T slot. Releases the X slot and
// leaves t_l1(Par) ready, signalled FIX->MTE1 on EvPA.
template <uint32_t R, uint32_t C, uint32_t S, uint32_t Slot, bool Par,
          event_t EvX, event_t EvPA>
AICORE void tilePhaseA(bool freeSlot) {
  using L = Layout<R, C, S>;

  MatTile<L::kStackRows, C> x_l1;
  TileLeft<half, L::kStackRows, C> x_l0a;
  TileRight<half, C, C> hc_l0;
  TileAcc<float, L::kStackRows, C> acc_stack;
  TASSIGN(x_l1, L::xSlot(Slot));
  TASSIGN(x_l0a, L::l0aX(Par));
  TASSIGN(hc_l0, L::kL0bHc);
  TASSIGN(acc_stack, L::l0cStack(Par));

  wait_flag(PIPE_MTE2, PIPE_MTE1, EvX);
  TMOV(x_l0a, x_l1);
  if (freeSlot) set_flag(PIPE_MTE1, PIPE_MTE2, EvX);
  set_flag(PIPE_MTE1, PIPE_M, EvPA);

  wait_flag(PIPE_MTE1, PIPE_M, EvPA);
  TMATMUL(acc_stack, x_l0a, hc_l0);
  set_flag(PIPE_M, PIPE_FIX, EvPA);

  // Park T in L1 for phase B, which reads it WIDE as (R x S*C). At C == 16 that
  // view is the same fractals in the same order as this (S*R x C) accumulator,
  // so one TMOV moves the whole tile; above 16 it is one descriptor per sample.
  wait_flag(PIPE_M, PIPE_FIX, EvPA);
  if constexpr (C == 16) {
    MatTile<L::kStackRows, C> t_l1;
    TASSIGN(t_l1, L::l1T(Par));
    TMOV(t_l1, acc_stack);
  } else {
    for (uint32_t s = 0; s < S; ++s) {
      MatTile<R, C> t_block;
      TASSIGN(t_block, L::l1T(Par) + s * R * C * sizeof(half));
      TEXTRACT(t_block, acc_stack, s * R, 0);
    }
  }
  set_flag(PIPE_FIX, PIPE_MTE1, EvPA);
}

// L1 T slot -> L0B, H_R * T -> L0C, L0C -> GM.
template <uint32_t R, uint32_t C, uint32_t S, bool Par, event_t EvPA,
          event_t EvPB>
AICORE void tilePhaseB(__gm__ half *y, uint32_t sample, uint32_t rows) {
  using L = Layout<R, C, S>;

  MatTile<R, S * C> t_l1;
  TileRight<half, R, S * C> t_l0b;
  TileLeft<half, R, R> hl_l0;
  TileAcc<float, R, S * C> acc_cat;
  TASSIGN(t_l1, L::l1T(Par));
  TASSIGN(t_l0b, L::l0bT(Par));
  TASSIGN(hl_l0, L::kL0aHl);
  TASSIGN(acc_cat, L::l0cCat(Par));

  wait_flag(PIPE_FIX, PIPE_MTE1, EvPA);
  TMOV(t_l0b, t_l1);
  set_flag(PIPE_MTE1, PIPE_M, EvPB);

  wait_flag(PIPE_MTE1, PIPE_M, EvPB);
  TMATMUL(acc_cat, hl_l0, t_l0b);
  set_flag(PIPE_M, PIPE_FIX, EvPB);

  wait_flag(PIPE_M, PIPE_FIX, EvPB);
  __gm__ half *y_tile = y + sample * L::kN;

  // Phase A's turnaround again, on the output: at C == 16 this (R x S*C)
  // accumulator IS the (S*R x C) stack the GM tile wants, so one whole-tile
  // store does every sample -- masked when the last tile is short.
  if constexpr (C == 16) {
    if constexpr (S > 1) {
      if (rows != S) {
        TileAcc<float, L::kStackRows, C, DYNAMIC> acc_short;
        TASSIGN(acc_short, L::l0cCat(Par));
        acc_short.SetValidRow(rows * R);
        if constexpr (L::kNzGm) {
          GmNzDyn<L::kStackRows, C> y_gm(y_tile);
          TASSIGN(y_gm, y_tile);
          y_gm.template SetShape<GlobalTensorDim::DIM_2>(rows * R / 16);
          TSTORE(y_gm, acc_short);
        } else {
          GmTileDyn<L::kStackRows, C> y_gm(y_tile);
          TASSIGN(y_gm, y_tile);
          y_gm.template SetShape<GlobalTensorDim::DIM_3>(rows * R);
          TSTORE(y_gm, acc_short);
        }
        return;
      }
    }
    TileAcc<float, L::kStackRows, C> acc_out;
    TASSIGN(acc_out, L::l0cCat(Par));
    if constexpr (L::kNzGm) {
      GmNz<L::kStackRows, C> y_gm(y_tile);
      TASSIGN(y_gm, y_tile);
      TSTORE(y_gm, acc_out);
    } else {
      GmTile<L::kStackRows, C> y_gm(y_tile);
      TASSIGN(y_gm, y_tile);
      TSTORE(y_gm, acc_out);
    }
    return;
  }

  // Above 16 the samples sit side by side in L0C, so each stores separately.
  // The cube ran all S whatever `rows` says, but the transform is per-sample,
  // so the surplus reaches only output rows this loop skips.
  for (uint32_t s = 0; s < rows; ++s) {
    TileAcc<float, R, C> acc_block;
    TASSIGN(acc_block, L::l0cCat(Par) + s * R * C * sizeof(float));
    __gm__ half *y_row = y + (sample + s) * L::kN;
    if constexpr (L::kNzGm) {
      GmNz<R, C> y_gm(y_row);
      TASSIGN(y_gm, y_row);
      TSTORE(y_gm, acc_block);
    } else {
      GmTile<R, C> y_gm(y_row);
      TASSIGN(y_gm, y_row);
      TSTORE(y_gm, acc_block);
    }
  }
}

// One loop iteration: prefetch tile i+3 into the ring slot it is about to
// vacate, start phase A of tile i+1, finish phase B of tile i.
template <uint32_t R, uint32_t C, uint32_t S, uint32_t SlotLoad, uint32_t SlotA,
          bool ParA, bool ParB, event_t EvXLoad, event_t EvXA, event_t EvPA_A,
          event_t EvPA_B, event_t EvPB_B>
AICORE void deepStep(__gm__ half *x, __gm__ half *y, uint32_t base, uint32_t i,
                     uint32_t n, uint32_t rest) {
  // Tile j's load reuses tile j-4's slot, so it waits iff j >= 4; here j = i+3.
  // Only the LAST tile can be short, so a row count picks `rest` or S.
  if (i + 3 < n) {
    issueTileLoad<R, C, S, SlotLoad, EvXLoad>(
        x, base + (i + 3) * S, (i + 3 == n - 1) ? rest : S, i >= 1);
  }
  // Tile j frees its slot iff tile j+4 exists to reuse it; here j = i+1.
  if (i + 1 < n) {
    tilePhaseA<R, C, S, SlotA, ParA, EvXA, EvPA_A>(i + 5 < n);
  }
  tilePhaseB<R, C, S, ParB, EvPA_B, EvPB_B>(y, base + i * S,
                                            (i == n - 1) ? rest : S);
}

template <uint32_t R, uint32_t C, uint32_t S>
AICORE void runTilesDeep(__gm__ half *x, __gm__ half *y, uint32_t &sample,
                         uint32_t end) {
  const uint32_t base = sample;
  const uint32_t span = end - base;
  // ceil, not floor: a core's remainder is the last tile of this ring, masked
  // to the rows it really holds, not a separate smaller tile.
  const uint32_t n = DIV_ROUNDUP(span, S);
  const uint32_t rest = span - (n - 1) * S;

  // No primed credits and none drained: tile j reuses tile j-4's slot, so its
  // load waits iff j >= 4 and tile j frees its slot iff tile j+4 exists. The
  // pairing is exact, and tile j's load is issued an iteration after tile j-4
  // was consumed, so the credit always precedes the wait.
  //
  // So the ring does NOT drain itself, and nothing follows it today. Anything
  // placed after it that reuses an L1 slot, L0A or L0C must first take the
  // buffers over on MTE1 -> MTE2, M -> MTE1 and FIX -> M.

  // Fill the ring three deep and open phase A of tile 0, so the first loop
  // iteration already has MTE2 two tiles ahead of MTE1.
  issueTileLoad<R, C, S, 0, EV_D0>(x, base, n == 1 ? rest : S, false);
  if (n > 1) {
    issueTileLoad<R, C, S, 1, EV_D1>(x, base + S, n == 2 ? rest : S, false);
  }
  if (n > 2) {
    issueTileLoad<R, C, S, 2, EV_D2>(x, base + 2 * S, n == 3 ? rest : S, false);
  }
  tilePhaseA<R, C, S, 0, false, EV_D0, EV_PA0>(4 < n);

  // Slot and parity follow from i mod 4:
  //
  //   i mod 4   load slot   A slot   A parity   B parity
  //   0         3           1        odd        even
  //   1         0           2        even       odd
  //   2         1           3        odd        even
  //   3         2           0        even       odd
  for (uint32_t i = 0; i < n; ++i) {
    switch (i & 3u) {
      case 0:
        deepStep<R, C, S, 3, 1, true, false, EV_D3, EV_D1, EV_PA1, EV_PA0,
                 EV_PB0>(x, y, base, i, n, rest);
        break;
      case 1:
        deepStep<R, C, S, 0, 2, false, true, EV_D0, EV_D2, EV_PA0, EV_PA1,
                 EV_PB1>(x, y, base, i, n, rest);
        break;
      case 2:
        deepStep<R, C, S, 1, 3, true, false, EV_D1, EV_D3, EV_PA1, EV_PA0,
                 EV_PB0>(x, y, base, i, n, rest);
        break;
      default:
        deepStep<R, C, S, 2, 0, false, true, EV_D2, EV_D0, EV_PA0, EV_PA1,
                 EV_PB1>(x, y, base, i, n, rest);
        break;
    }
  }
  sample = end;
}

// One tile, standalone. It has no neighbour to wait for or signal to, so every
// credit is skipped. Not the ring: that would instantiate its loop and all four
// slot/parity combinations at every size runShortSpan can reach.
template <uint32_t R, uint32_t C, uint32_t S>
AICORE void oneTile(__gm__ half *x, __gm__ half *y, uint32_t sample,
                    uint32_t rows) {
  if constexpr (Layout<R, C, S>::kDbuf) {
    issueTileLoad<R, C, S, 0, EV_D0>(x, sample, rows, false);
    tilePhaseA<R, C, S, 0, false, EV_D0, EV_PA0>(false);
    tilePhaseB<R, C, S, false, EV_PA0, EV_PB0>(y, sample, rows);
  } else {
    issueTileLoad<R, C, S, 0, EV_X0>(x, sample, rows, false);
    hadamardTile<R, C, S, false, EV_X0>(y, sample, false, false, false);
  }
}

// A span of 1..S-1 samples, as ONE tile of the smallest size that covers it.
// Masking is free on the TRANSFERS but not on the tile -- the cube runs all S
// samples regardless -- and a lone tile, unlike the ring's last tile, has no
// predecessor to hide the waste behind. Exactly one tile ever runs.
template <uint32_t R, uint32_t C, uint32_t S>
AICORE void runShortSpan(__gm__ half *x, __gm__ half *y, uint32_t sample,
                         uint32_t span) {
  if constexpr (S > 1) {
    if (span <= S / 2) {
      runShortSpan<R, C, S / 2>(x, y, sample, span);
      return;
    }
  }
  oneTile<R, C, S>(x, y, sample, span);
}

// Consume [sample, end) as ONE unit of work: a single ring of S-sample tiles
// whose last tile is masked to whatever is left, so there is no level boundary
// to hand buffers over at and the remainder overlaps like any other tile.
template <uint32_t R, uint32_t C, uint32_t S>
AICORE void runTiles(__gm__ half *x, __gm__ half *y, uint32_t &sample,
                     uint32_t end) {
  if (sample >= end) {
    return;
  }
  // A core with no whole tile at all: one masked tile at its own size.
  if constexpr (S > 1) {
    if (end - sample < S) {
      runShortSpan<R, C, S>(x, y, sample, end - sample);
      sample = end;
      return;
    }
  }
  if constexpr (Layout<R, C, S>::kDbuf) {
    runTilesDeep<R, C, S>(x, y, sample, end);
  } else {
    // kDbuf is monotone in S and every shallow size in the case table has
    // S == 1, so a shallow tile is one sample and can never be short.
    static_assert(S == 1,
                  "a shallow tile size > 1 would need a masked last tile");
    // Same lazy credits as the deep ring, at depth 2: tile i's load waits iff
    // i >= 2, tile i frees its slot iff tile i+2 exists, and the EV_ACC edge
    // runs between neighbours only, so tile 0 does not wait on it and the last
    // tile does not set it.
    const uint32_t last = (end - sample) / S - 1;

    uint32_t cur = sample;
    issueTileLoad<R, C, S, 0, EV_X0>(x, cur, S, false);

    for (uint32_t i = 0;; ++i) {
      const uint32_t next = cur + S;
      const bool has_next = (i < last);
      const bool free_slot = (i + 2 <= last);
      const bool wait_acc = (i >= 1);
      const bool free_acc = (i < last);
      if (i & 1u) {
        if (has_next) {
          issueTileLoad<R, C, S, 0, EV_X0>(x, next, S, i >= 1);
        }
        hadamardTile<R, C, S, true, EV_X1>(y, cur, free_slot, wait_acc,
                                           free_acc);
      } else {
        if (has_next) {
          issueTileLoad<R, C, S, 1, EV_X1>(x, next, S, i >= 1);
        }
        hadamardTile<R, C, S, false, EV_X0>(y, cur, free_slot, wait_acc,
                                            free_acc);
      }
      if (!has_next) {
        break;
      }
      cur = next;
    }
    sample = cur + S;
  }
}

template <uint32_t N, uint32_t R, uint32_t C, uint32_t S>
AICORE void runTwosided(__gm__ half *x, __gm__ half *y, __gm__ half *h_left,
                        __gm__ half *h_right, uint32_t batch) {
  static_assert(R * C == N, "R * C must equal N");
  static_assert(R % 16 == 0 && C % 16 == 0, "cube tiles need 16-aligned dims");

  const uint32_t num_cores = block_num;
  const uint32_t core_id = get_block_idx();

  // Split by TILES, not samples: an even sample split leaves most cores on the
  // short-span path, paying a full tile's handshakes for a fraction of a tile's
  // work. The price is idling the last cores.
  const uint32_t tiles_per_core = DIV_ROUNDUP(DIV_ROUNDUP(batch, S), num_cores);
  const uint32_t begin = tiles_per_core * S * core_id;
  if (begin >= batch) {
    return;
  }
  uint32_t end = begin + tiles_per_core * S;
  if (end > batch) {
    end = batch;
  }

  loadHadamardFactors<R, C, S>(h_left, h_right);

  uint32_t sample = begin;
  runTiles<R, C, S>(x, y, sample, end);
}

AICORE void dispatchTwosided(__gm__ half *x, __gm__ half *y,
                             __gm__ half *h_left, __gm__ half *h_right,
                             uint32_t batch, uint32_t n) {
  switch (n) {
#define FAST_HADAMARD_TWOSIDED_DISPATCH(N, R, C, S)               \
  case N: {                                                       \
    constexpr uint32_t kMid = ((S) > 1) ? (S) / 2 : 1;            \
    constexpr uint32_t kShort = (kMid > 1) ? kMid / 2 : 1;        \
    if constexpr ((S) == 1) {                                     \
      runTwosided<N, R, C, S>(x, y, h_left, h_right, batch);      \
    } else if (batch >= (uint32_t)(PTO_TS_TILES_TALL) * (S)) {    \
      runTwosided<N, R, C, S>(x, y, h_left, h_right, batch);      \
    } else if constexpr (kMid == kShort) {                        \
      runTwosided<N, R, C, kMid>(x, y, h_left, h_right, batch);   \
    } else if (batch >= (uint32_t)(PTO_TS_TILES_MID) * kMid) {    \
      runTwosided<N, R, C, kMid>(x, y, h_left, h_right, batch);   \
    } else {                                                      \
      runTwosided<N, R, C, kShort>(x, y, h_left, h_right, batch); \
    }                                                             \
    return;                                                       \
  }
    FAST_HADAMARD_TWOSIDED_CASES(FAST_HADAMARD_TWOSIDED_DISPATCH)
#undef FAST_HADAMARD_TWOSIDED_DISPATCH
    default:
      return;
  }
}

}  // namespace

#endif

__global__ AICORE void fast_hadamard_twosided_fp16(__gm__ void *x,
                                                   __gm__ void *y,
                                                   __gm__ void *h_left,
                                                   __gm__ void *h_right,
                                                   uint32_t batch, uint32_t n) {
#if defined(__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  dispatchTwosided((__gm__ half *)x, (__gm__ half *)y, (__gm__ half *)h_left,
                   (__gm__ half *)h_right, batch, n);
#endif
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                            uint8_t *y, uint8_t *h_left, uint8_t *h_right,
                            uint32_t batch, uint32_t n) {
  fast_hadamard_twosided_fp16<<<blockDim, nullptr, stream>>>(x, y, h_left,
                                                             h_right, batch, n);
}
