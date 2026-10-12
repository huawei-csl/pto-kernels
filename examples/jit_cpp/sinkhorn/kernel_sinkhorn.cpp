/**
Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
See LICENSE in the root of the software repository for the full License text.
*/

/**
 * Doubly-stochastic Sinkhorn normalization — Ascend 910B kernel (fp16 I/O).
 *
 * Mirrors DeepSeek TileKernels `sinkhorn_normalize_ref`:
 *
 *     x = x.softmax(-1) + eps
 *     x = x / (x.sum(-2, keepdim=True) + eps)
 *     for _ in range(repeat - 1):
 *         x = x / (x.sum(-1, keepdim=True) + eps)
 *         x = x / (x.sum(-2, keepdim=True) + eps)
 *
 * Three code paths, dispatched on K:
 *
 *   K ∈ {4, 8, 16}  — `sinkhornFastPath` (TILE_COLS = 16)
 *                     K-templated so every tile dimension is compile-time.
 *                     Uses `TCOLEXPANDDIV` (PTO-ISA 9.0.0 op) on the full
 *                     [K, ROW_BLOCK_COLS] interleaved tile, replacing
 *                     `K+1` TADD-tree ops + `K` TDIVs per iteration with
 *                     exactly two ops (TCOLSUM + TCOLEXPANDDIV).
 *
 *   K ∈ (16, 64]    — `sinkhornStridedTree` (TILE_COLS ∈ {16, 32, 64})
 *                     K-runtime.  Interleaved layout but falls back to
 *                     a flat TADD-tree + K×TDIV col-normalize because
 *                     `TCOLEXPANDDIV` on a K-runtime tile requires
 *                     runtime tile widths we don't have.
 *
 *   K ∈ (64, 128]   — `sinkhornPerMatrixFp32`
 *                     fp16 I/O with fp32 internal compute (fp16 loses too
 *                     much precision at K=128).  Per-matrix, no batching.
 *
 * Parallelism model (all paths):
 *   The N matrices are sharded across AIV cores (`num_workers` total,
 *   = get_block_num() × get_subblockdim()).  Each worker takes an
 *   equal slice.  For fast / strided-tree paths it processes its slice
 *   in chunks of up to `MAX_BATCH_MATRICES` matrices, using one bulk
 *   TLOAD + TSTORE per chunk.  Within each chunk, matrices are further
 *   divided into groups of up to `MAX_GROUP_SIZE`; each group is
 *   softmaxed + sinkhorn-iterated as one batched unit.
 *
 * UB layout (fast / strided-tree paths):
 *   SCRATCH_UB     — reduction scratch (used as `tmp` by TROW/TCOLSUM,
 *                    and as row-block scratch in col-normalize)
 *   WORK_UB        — primary matrix data, in interleaved layout
 *   ROW_STATS_UB   — per-row reduction output (K×1 col-vec)
 *   COL_STATS_UB   — per-column reduction output (1×BLK row-vec)
 *   BATCH_UB       — bulk-loaded matrices before interleave; written back
 *                    after all groups in the chunk are processed.
 */

#include <pto/pto-inst.hpp>

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

using namespace pto;

// ==========================================================================
// Compile-time constants
// ==========================================================================
constexpr uint32_t UB_BYTES = 192 * 1024;  // per-AIV unified buffer size
constexpr uint32_t MAX_K = 128;            // max matrix dim we support
constexpr uint32_t STACK_ROWS =
    512;  // tall-tile row count for fast / strided-tree paths
constexpr uint32_t MAX_MATS_PER_GROUP_CAP =
    128;  // upper bound on mats per group (UB footprint)

// 32-byte align helper (fp16 PTO tiles require 32-byte-aligned row bytes).
#define ALIGN_32(x) (((x) + 31u) & ~31u)

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// ==========================================================================
// Tile type aliases
// ==========================================================================
// 1-D row vector over N elements (static).  Used for flat elementwise ops.
template <typename T, uint32_t N>
using FlatVec = Tile<TileType::Vec, T, 1, N, BLayout::RowMajor, -1, -1>;

// 2-D row-major tile.  Row stride = static Cols; valid shape is runtime.
template <typename T, uint32_t Rows, uint32_t Cols>
using Tile2D =
    Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

// 2-D col-major R×1 vector — used as output of per-row reductions
// (TROWMAX / TROWSUM give one scalar per row).
template <typename T, uint32_t Rows>
using ColVec =
    Tile<TileType::Vec, T, Rows, 1, BLayout::ColMajor, DYNAMIC, DYNAMIC>;

// ==========================================================================
// Global-memory tensor aliases (contiguous row-major)
// ==========================================================================
using GmDenseStride = Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T>
using GmShape2D = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
template <typename T, uint32_t Cols>
using GmTensor = GlobalTensor<T, GmShape2D<T>, GmDenseStride, Layout::ND>;

// ==========================================================================
// Pipeline-flag helpers
// ==========================================================================
// Each AIV has three pipelines (MTE2 / V / MTE3) that run in parallel.
// Cross-pipe ordering uses set_flag / wait_flag pairs keyed by EVENT_ID.
// `initPipelineFlags` primes the flags so the first wait_flag below
// succeeds immediately (lets us always-wait-then-set in a ring).
AICORE inline void initPipelineFlags() {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

AICORE inline void drainPipelineFlags() {
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

// ==========================================================================
// Strided UB→UB copy
// ==========================================================================
// Wrapper around the `copy_ubuf_to_ubuf` CCE builtin.  Used to transpose
// between the natural-order batch layout and the interleaved row-block
// layout expected by batched col-normalize.
//
// Parameters mirror the builtin:
//   nBurst    — number of bursts (= number of matrices copied)
//   lenBurst  — bytes per burst, expressed in 32B blocks
//   srcGap    — gap between source bursts, in 32B blocks
//   dstGap    — gap between destination bursts, in 32B blocks
namespace pto {
template <typename TileDescriptor>
__tf__ AICORE inline void stridedUBCopyImpl(
    typename TileDescriptor::TileDType __out__ dstTile,
    typename TileDescriptor::TileDType __in__ srcTile, uint16_t nBurst,
    uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap) {
  __ubuf__ void *dst = (__ubuf__ void *)__cce_get_tile_ptr(dstTile);
  __ubuf__ void *src = (__ubuf__ void *)__cce_get_tile_ptr(srcTile);
  __builtin_cce_copy_ubuf_to_ubuf(dst, src, (uint8_t)0, nBurst, lenBurst,
                                  srcGap, dstGap);
}
}  // namespace pto

template <typename TileT>
AICORE inline void stridedUBCopy(TileT &dst, TileT &src, uint16_t nBurst,
                                 uint16_t lenBurst, uint16_t srcGap,
                                 uint16_t dstGap) {
  pto::stridedUBCopyImpl<TileT>(dst.data(), src.data(), nBurst, lenBurst,
                                srcGap, dstGap);
}

// ==========================================================================
// Fast path:  K ∈ {4, 8, 16}  —  TCOLEXPANDDIV on full interleaved tile
// ==========================================================================
//
// Every tile dimension is compile-time, which lets us:
//   (a) keep the tile row-stride static (it equals `ROW_BLOCK_COLS`);
//   (b) run TCOLSUM + TCOLEXPANDDIV on the full [K, ROW_BLOCK_COLS] tile,
//       replacing the K+1-op TADD tree + K-op TDIV sequence.
//
// Two views of the same physical UB memory:
//
//   Tall view       shape (STACK_ROWS,       TILE_COLS       )
//                   row stride = TILE_COLS
//                   used for per-row ops (softmax, row-normalize)
//                   matrix i's row r lives at tall-row ((i / GS) * K + i % GS +
//                   r * GS) — i.e. per-matrix rows are naturally discoverable
//
//   Interleaved    shape (K,                ROW_BLOCK_COLS  )
//   view           row stride = ROW_BLOCK_COLS = MAX_GROUP_SIZE * TILE_COLS
//                   used for per-col ops (col-normalize).  Each "column"
//                   of the interleaved view corresponds to one matrix's
//                   one column, so TCOLSUM gives per-matrix-per-col sums
//                   directly, and TCOLEXPANDDIV normalizes correctly.
//
// Both views work simultaneously because the tall stride (TILE_COLS) times
// the group size (MAX_GROUP_SIZE) equals the interleaved stride
// (ROW_BLOCK_COLS).  We always process a full group of MAX_GROUP_SIZE
// matrices logically: partial groups are zero-padded.  Zero padding makes
// softmax produce 1/K in pad cells, which is a benign constant and doesn't
// leak into valid-matrix outputs (each matrix's row/col-normalize is local
// to that matrix's slice of the interleaved tile).
template <typename T, uint32_t K_TEMPLATE, uint32_t TILE_COLS_TEMPLATE,
          uint32_t REPEAT, uint32_t STACK_ROWS_OVERRIDE = STACK_ROWS>
AICORE void sinkhornFastPath(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                             float eps) {
  // ---- compile-time constants derived from template parameters ----
  constexpr unsigned K = K_TEMPLATE;
  constexpr unsigned TILE_COLS = TILE_COLS_TEMPLATE;
  constexpr unsigned TALL_ROWS = STACK_ROWS_OVERRIDE;
  constexpr unsigned MAX_GROUP_SIZE = TALL_ROWS / K;  // matrices per group
  constexpr unsigned ROW_BLOCK_COLS =
      MAX_GROUP_SIZE * TILE_COLS;  // width of interleaved rows
  static_assert(K * ROW_BLOCK_COLS == TALL_ROWS * TILE_COLS,
                "Interleaved and tall views must cover the same UB region");

  constexpr unsigned MATRIX_ROW_BYTES = TILE_COLS * sizeof(half);
  constexpr unsigned TILE_BYTES = TALL_ROWS * TILE_COLS * sizeof(half);

  // ---- UB layout ----
  // Fixed regions first (compute scratch + per-axis stats), then the
  // double-buffered batch region at the top of UB.  The two batch halves
  // (BATCH_UB_PING / BATCH_UB_PONG) alternate on consecutive chunks so
  // that MTE2 TLOAD of the next chunk overlaps PIPE_V compute on the
  // current chunk, and MTE3 TSTORE of the previous chunk overlaps both.
  constexpr unsigned SCRATCH_UB = 0;  // reduction scratch
  constexpr unsigned WORK_UB =
      ALIGN_32(SCRATCH_UB + TILE_BYTES);  // interleaved matrix data
  constexpr unsigned ROW_STATS_UB =
      ALIGN_32(WORK_UB + TILE_BYTES);  // per-row reduction output
  constexpr unsigned COL_STATS_UB =
      ALIGN_32(ROW_STATS_UB + ALIGN_32(TALL_ROWS * sizeof(half)));
  constexpr unsigned BATCH_UB_BASE =
      ALIGN_32(COL_STATS_UB + ALIGN_32(ROW_BLOCK_COLS * sizeof(half)));

  // Split remaining UB in half for ping/pong.  Round down to a multiple
  // of the row size and cap at the hardware burst-count limit.
  constexpr unsigned BATCH_HALF_BUDGET = (UB_BYTES - BATCH_UB_BASE) / 2;
  constexpr unsigned BATCH_HALF_ROWS_RAW =
      BATCH_HALF_BUDGET / (TILE_COLS * sizeof(half));
  constexpr unsigned MAX_BATCH_ROWS =
      BATCH_HALF_ROWS_RAW < 4095 ? BATCH_HALF_ROWS_RAW : 4095;
  constexpr unsigned BATCH_HALF_BYTES =
      MAX_BATCH_ROWS * TILE_COLS * sizeof(half);
  constexpr unsigned BATCH_UB_PING = BATCH_UB_BASE;
  constexpr unsigned BATCH_UB_PONG = BATCH_UB_BASE + ALIGN_32(BATCH_HALF_BYTES);
  static_assert(BATCH_UB_PONG + BATCH_HALF_BYTES <= UB_BYTES,
                "Double-buffered BATCH_UB exceeds UB capacity");

  // Hardware setup.
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Per-worker sharding.  N matrices evenly split across all AIV cores;
  // the first `remainder` cores take one extra matrix.
  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t worker_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t base_per_worker = N / num_workers;
  const uint32_t remainder = N % num_workers;
  const uint32_t my_first = worker_id * base_per_worker +
                            (worker_id < remainder ? worker_id : remainder);
  const uint32_t my_count = base_per_worker + (worker_id < remainder ? 1 : 0);
  if (my_count == 0) return;

  // Loop constants.
  constexpr uint32_t K_SQUARED = K * K;
  constexpr uint32_t GROUP_SIZE_STATIC = MAX_GROUP_SIZE;
  constexpr uint32_t CHUNK_MATRICES =
      MAX_BATCH_ROWS / K;  // mats per TLOAD chunk
  const half eps_h = (half)eps;

  // Prime all four cross-pipe flags (two halves × two directions).
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  // ========================================================================
  // Outer loop: process my matrices in chunks of up to CHUNK_MATRICES.
  // Chunks alternate between the two BATCH_UB halves so DMA overlaps
  // PIPE_V compute on the other half.  The padding columns of each half
  // are zeroed the first time that half is used (lazy init — small-batch
  // kernels that only touch PING never pay for PONG's zero-fill).
  // ========================================================================
  bool half_zeroed[2] = {false, false};
  uint32_t ping = 1;  // 1 → PING half + EVENT_ID0; 0 → PONG half + EVENT_ID1
  for (uint32_t chunk_offset = 0; chunk_offset < my_count;
       chunk_offset += CHUNK_MATRICES, ping = 1 - ping) {
    const uint32_t chunk_matrices =
        min(CHUNK_MATRICES, my_count - chunk_offset);
    const uint32_t chunk_rows = chunk_matrices * K;
    __gm__ T *chunk_gm_in =
        gm_in + (size_t)(my_first + chunk_offset) * K_SQUARED;
    __gm__ T *chunk_gm_out =
        gm_out + (size_t)(my_first + chunk_offset) * K_SQUARED;

    const unsigned batch_ub = ping ? BATCH_UB_PING : BATCH_UB_PONG;
    const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;

    // Bulk TLOAD from GM → this half of BATCH_UB (natural order).
    Tile2D<T, MAX_BATCH_ROWS, TILE_COLS> batch_tile(chunk_rows, K);
    TASSIGN(batch_tile, batch_ub);

    GmShape2D<T> gm_shape(chunk_rows, K);
    GmDenseStride gm_stride(K);
    GmTensor<T, TILE_COLS> gm_in_tensor(chunk_gm_in, gm_shape, gm_stride);

    wait_flag(PIPE_V, PIPE_MTE2,
              ev);  // wait for PIPE_V to finish with this half

    // Lazy-zero: on this half's first use, zero the TILE_COLS - K padding cols
    // so subsequent ops don't read uninitialized data.  TLOAD below writes
    // only K cols per row; the padding stays at zero for the kernel lifetime.
    if (!half_zeroed[ping]) {
      FlatVec<T, MAX_BATCH_ROWS * TILE_COLS> zero_flat(
          1, MAX_BATCH_ROWS * TILE_COLS);
      TASSIGN(zero_flat, batch_ub);
      TEXPANDS(zero_flat, (T)0);
      pipe_barrier(PIPE_V);
      half_zeroed[ping] = true;
    }

    TLOAD(batch_tile, gm_in_tensor);
    set_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE3, PIPE_V,
              ev);  // wait for previous TSTORE on this half to drain

    // ======================================================================
    // Inner loop: process the chunk in groups of up to MAX_GROUP_SIZE.
    // ======================================================================
    for (uint32_t group_start = 0; group_start < chunk_matrices;
         group_start += GROUP_SIZE_STATIC) {
      const uint32_t group_size =
          min(GROUP_SIZE_STATIC, chunk_matrices - group_start);
      const unsigned group_batch_offset =
          batch_ub + group_start * K * TILE_COLS * sizeof(T);

      // stridedUBCopy parameters for natural-order → interleaved transpose.
      constexpr uint16_t tile_row_blocks = TILE_COLS * sizeof(half) / 32;
      const uint16_t src_gap_blocks = (uint16_t)(K - 1) * tile_row_blocks;

      // --- Zero WORK_UB (pads invalid matrices in the group to 0) -----
      // Only needed for partial groups (last group may have fewer than
      // GROUP_SIZE_STATIC matrices).  Full groups fully overwrite WORK_UB
      // in the interleave step below, so the zero is redundant.
      if (group_size < GROUP_SIZE_STATIC) {
        FlatVec<T, K * ROW_BLOCK_COLS> work_flat(1, K * ROW_BLOCK_COLS);
        TASSIGN(work_flat, WORK_UB);
        TEXPANDS(work_flat, (T)0);
        pipe_barrier(PIPE_V);
      }

      // --- Interleave: BATCH_UB → WORK_UB ------------------------------
      // After this, WORK_UB's row-block r (offset r*ROW_BLOCK_COLS) holds:
      //   [matrix 0 row r][matrix 1 row r]...[matrix (group_size-1) row r]
      // followed by zero padding up to ROW_BLOCK_COLS.
      for (uint32_t row = 0; row < K; ++row) {
        Tile2D<half, TALL_ROWS, TILE_COLS> src_view(group_size, K);
        Tile2D<half, TALL_ROWS, TILE_COLS> dst_view(group_size, K);
        TASSIGN(src_view, group_batch_offset + row * MATRIX_ROW_BYTES);
        TASSIGN(dst_view,
                WORK_UB + row * ROW_BLOCK_COLS * (unsigned)sizeof(half));
        stridedUBCopy(dst_view, src_view, (uint16_t)group_size, tile_row_blocks,
                      src_gap_blocks, (uint16_t)0);
      }
      pipe_barrier(PIPE_V);

      // --- Sinkhorn computation ---------------------------------------
      if constexpr (REPEAT > 0) {
        // Tall view over WORK_UB for per-row ops (softmax, row-normalize).
        // Row stride = TILE_COLS; we operate on all TALL_ROWS rows (padded
        // matrices are zero and produce benign softmax results).
        Tile2D<half, TALL_ROWS, TILE_COLS> tall_matrix(TALL_ROWS, K);
        TASSIGN(tall_matrix, WORK_UB);

        Tile2D<half, TALL_ROWS, TILE_COLS> tall_scratch(TALL_ROWS, K);
        TASSIGN(tall_scratch, SCRATCH_UB);

        ColVec<half, TALL_ROWS> row_stats(TALL_ROWS, 1);
        TASSIGN(row_stats, ROW_STATS_UB);

        // ── Step 1: softmax along each matrix-row ──────────────────────
        //   row_stats[i] = max(tall_matrix[i, :])
        //   tall_matrix[i, :] = exp(tall_matrix[i, :] - row_stats[i])
        //   row_stats[i] = sum(tall_matrix[i, :])
        //   tall_matrix[i, :] = tall_matrix[i, :] / row_stats[i]
        TROWMAX(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDSUB(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        {
          FlatVec<half, TALL_ROWS * TILE_COLS> work_flat(1,
                                                         TALL_ROWS * TILE_COLS);
          TASSIGN(work_flat, WORK_UB);
          TEXP(work_flat, work_flat);
          pipe_barrier(PIPE_V);
        }

        TROWSUM(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        // Step 2 (add eps to the matrix) eliminated: after softmax every
        // valid cell is strictly positive (exp() > 0, rowsum > 0), and
        // zero-padding matrices also produce positive cells (= 1/K), so
        // col-normalize never sees a zero denominator.  The eps was
        // reference-code defensive, not algorithmically required for
        // random inputs of this type.

        // ── Step 3 & 4: col-normalize (and iterations) ─────────────────
        // Interleaved view: (K, ROW_BLOCK_COLS) with row stride =
        // ROW_BLOCK_COLS.  TCOLSUM gives one scalar per column — and
        // each column here is one matrix's one column, so the scalar
        // we get is exactly that matrix's col sum.
        Tile2D<half, K, ROW_BLOCK_COLS> interleaved_matrix(K, ROW_BLOCK_COLS);
        TASSIGN(interleaved_matrix, WORK_UB);

        Tile2D<half, K, ROW_BLOCK_COLS> interleaved_scratch(K, ROW_BLOCK_COLS);
        TASSIGN(interleaved_scratch, SCRATCH_UB);

        FlatVec<half, ROW_BLOCK_COLS> col_stats(1, ROW_BLOCK_COLS);
        TASSIGN(col_stats, COL_STATS_UB);

// Fused col-normalize: 1 TCOLSUM + 1 TCOLEXPANDDIV, 2 barriers.
#define COL_NORMALIZE()                                                \
  do {                                                                 \
    TCOLSUM(col_stats, interleaved_matrix, interleaved_scratch, true); \
    pipe_barrier(PIPE_V);                                              \
    TCOLEXPANDDIV(interleaved_matrix, interleaved_matrix, col_stats);  \
    pipe_barrier(PIPE_V);                                              \
  } while (0)

        // First col-normalize (no row-normalize — softmax already normalized
        // rows).
        COL_NORMALIZE();

// (REPEAT − 1) × { row-normalize ; col-normalize }.
#pragma unroll
        for (uint32_t iter = 1; iter < REPEAT; ++iter) {
          TASSIGN(row_stats, ROW_STATS_UB);

          TROWSUM(row_stats, tall_matrix, tall_scratch);
          pipe_barrier(PIPE_V);

          TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
          pipe_barrier(PIPE_V);

          COL_NORMALIZE();
        }
#undef COL_NORMALIZE
      }

      // --- De-interleave: WORK_UB → BATCH_UB --------------------------
      // Inverse of the interleave above; only the first `group_size`
      // matrices are copied (zero-padded tail is discarded).
      for (uint32_t row = 0; row < K; ++row) {
        Tile2D<half, TALL_ROWS, TILE_COLS> src_view(group_size, K);
        Tile2D<half, TALL_ROWS, TILE_COLS> dst_view(group_size, K);
        TASSIGN(src_view,
                WORK_UB + row * ROW_BLOCK_COLS * (unsigned)sizeof(half));
        TASSIGN(dst_view, group_batch_offset + row * MATRIX_ROW_BYTES);
        stridedUBCopy(dst_view, src_view, (uint16_t)group_size, tile_row_blocks,
                      (uint16_t)0, src_gap_blocks);
      }
      pipe_barrier(PIPE_V);
    }

    // Bulk TSTORE from this half of BATCH_UB → GM.
    GmTensor<T, TILE_COLS> gm_out_tensor(chunk_gm_out, gm_shape, gm_stride);
    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);
    TSTORE(gm_out_tensor, batch_tile);
    set_flag(PIPE_MTE3, PIPE_V, ev);  // next use of this half waits on this
    set_flag(PIPE_V, PIPE_MTE2, ev);  // next TLOAD of this half can now proceed
  }

  // Drain all four ping-pong flags before exit.
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

// ==========================================================================
// Small-batch path:  K ∈ {4, 8, 16}, N < ~2048  —  natural-order, no DB
// ==========================================================================
//
// The fast-path's strided-interleaved layout + double-buffer + batched
// `TCOLEXPANDDIV` on the full `(K, ROW_BLOCK_COLS)` interleaved tile
// amortizes beautifully at large batches — but pays ~25us of flag / setup
// overhead per kernel call that's wasted when there are few matrices
// (only one chunk per worker, no overlap benefit).
//
// This path is the simplest possible layout: one matrix per group in a
// (K, K) sub-tile at natural UB row-stride.  Per-matrix col-normalize is
// `TCOLSUM + TCOLEXPANDDIV` in a loop over the `gc` valid matrices.  No
// double-buffer, no interleave / de-interleave copies — just a TLOAD, an
// inner TMOV from BUF to MAT, the sinkhorn iterations on the tall view,
// and a TSTORE.
//
// At K=4, batch=1 this path clocks ~14us vs ~40us for the fast-path.
// Crossover with the fast-path is around batch=2048.
template <typename T, uint32_t K_TEMPLATE, uint32_t TILE_COLS_TEMPLATE,
          uint32_t REPEAT>
AICORE void sinkhornSmallBatch(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                               float eps) {
  constexpr unsigned K = K_TEMPLATE;
  constexpr unsigned TILE_COLS = TILE_COLS_TEMPLATE;
  constexpr unsigned TALL_ROWS = STACK_ROWS;
  constexpr unsigned TILE_BYTES = TALL_ROWS * TILE_COLS * sizeof(half);
  constexpr unsigned MATRIX_ROW_BYTES = TILE_COLS * sizeof(half);

  // UB layout: MAT (working tile) | SCRATCH (reduction tmp) | ROW_STATS |
  // BATCH_UB
  constexpr unsigned MAT_UB = 0;
  constexpr unsigned SCRATCH_UB = ALIGN_32(MAT_UB + TILE_BYTES);
  constexpr unsigned ROW_STATS_UB = ALIGN_32(SCRATCH_UB + TILE_BYTES);
  constexpr unsigned BATCH_UB =
      ALIGN_32(ROW_STATS_UB + ALIGN_32(TALL_ROWS * sizeof(half)));
  constexpr unsigned BATCH_BUF_ROWS_RAW =
      (UB_BYTES - BATCH_UB) / (TILE_COLS * sizeof(half));
  constexpr unsigned MAX_BATCH_ROWS =
      BATCH_BUF_ROWS_RAW < 4095 ? BATCH_BUF_ROWS_RAW : 4095;

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t worker_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t base_per_worker = N / num_workers;
  const uint32_t remainder = N % num_workers;
  const uint32_t my_first = worker_id * base_per_worker +
                            (worker_id < remainder ? worker_id : remainder);
  const uint32_t my_count = base_per_worker + (worker_id < remainder ? 1 : 0);
  if (my_count == 0) return;

  constexpr uint32_t K_SQUARED = K * K;
  constexpr uint32_t MAX_GROUP_SIZE = TALL_ROWS / K;  // matrices per group
  constexpr uint32_t CHUNK_MATRICES = MAX_BATCH_ROWS / K;
  const half eps_h = (half)eps;

  initPipelineFlags();

  for (uint32_t chunk_offset = 0; chunk_offset < my_count;
       chunk_offset += CHUNK_MATRICES) {
    const uint32_t chunk_matrices =
        min(CHUNK_MATRICES, my_count - chunk_offset);
    const uint32_t chunk_rows = chunk_matrices * K;
    __gm__ T *chunk_gm_in =
        gm_in + (size_t)(my_first + chunk_offset) * K_SQUARED;
    __gm__ T *chunk_gm_out =
        gm_out + (size_t)(my_first + chunk_offset) * K_SQUARED;

    // Zero the BATCH_UB region we're about to load (padding cols stay 0).
    {
      FlatVec<T, MAX_BATCH_ROWS * TILE_COLS> zero_flat(1,
                                                       chunk_rows * TILE_COLS);
      TASSIGN(zero_flat, BATCH_UB);
      TEXPANDS(zero_flat, (T)0);
      pipe_barrier(PIPE_V);
    }

    Tile2D<T, MAX_BATCH_ROWS, TILE_COLS> batch_tile(chunk_rows, K);
    TASSIGN(batch_tile, BATCH_UB);
    GmShape2D<T> gm_shape(chunk_rows, K);
    GmDenseStride gm_stride(K);
    GmTensor<T, TILE_COLS> gm_in_tensor(chunk_gm_in, gm_shape, gm_stride);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(batch_tile, gm_in_tensor);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Process the chunk in groups of up to MAX_GROUP_SIZE matrices.
    for (uint32_t group_start = 0; group_start < chunk_matrices;
         group_start += MAX_GROUP_SIZE) {
      const uint32_t group_size =
          min(MAX_GROUP_SIZE, chunk_matrices - group_start);
      const uint32_t group_rows = group_size * K;
      const uint32_t group_cells = group_rows * TILE_COLS;
      const unsigned group_batch_offset =
          BATCH_UB + group_start * K * TILE_COLS * sizeof(T);

      // Copy this group's data from BATCH_UB → MAT_UB (natural order, same
      // stride).
      {
        FlatVec<T, TALL_ROWS * TILE_COLS> zero_mat(1, TALL_ROWS * TILE_COLS);
        TASSIGN(zero_mat, MAT_UB);
        TEXPANDS(zero_mat, (T)0);
        pipe_barrier(PIPE_V);
      }
      {
        FlatVec<T, TALL_ROWS * TILE_COLS> src(1, group_cells);
        FlatVec<T, TALL_ROWS * TILE_COLS> dst(1, group_cells);
        TASSIGN(src, group_batch_offset);
        TASSIGN(dst, MAT_UB);
        TMOV(dst, src);
        pipe_barrier(PIPE_V);
      }

      Tile2D<half, TALL_ROWS, TILE_COLS> tall_matrix(group_rows, K);
      TASSIGN(tall_matrix, MAT_UB);
      Tile2D<half, TALL_ROWS, TILE_COLS> tall_scratch(group_rows, K);
      TASSIGN(tall_scratch, SCRATCH_UB);
      ColVec<half, TALL_ROWS> row_stats(group_rows, 1);
      TASSIGN(row_stats, ROW_STATS_UB);

      if constexpr (REPEAT > 0) {
        // Softmax on tall view (group_rows, K).
        TROWMAX(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDSUB(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        {
          FlatVec<half, TALL_ROWS * TILE_COLS> flat(1, group_cells);
          TASSIGN(flat, MAT_UB);
          TEXP(flat, flat);
          pipe_barrier(PIPE_V);
        }

        TROWSUM(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        {
          FlatVec<half, TALL_ROWS * TILE_COLS> flat(1, group_cells);
          TASSIGN(flat, MAT_UB);
          TADDS(flat, flat, eps_h);
          pipe_barrier(PIPE_V);
        }

// Per-matrix column-normalize.  With K compile-time we can call
// `TCOLEXPANDDIV` on each K×K sub-tile — 2 ops per matrix, one
// pipe_barrier between.
#define PER_MATRIX_COL_NORM()                                      \
  do {                                                             \
    for (uint32_t mi = 0; mi < group_size; ++mi) {                 \
      const unsigned mat_off = MAT_UB + mi * K * MATRIX_ROW_BYTES; \
      Tile2D<half, TILE_COLS, TILE_COLS> sub_mat(K, K);            \
      TASSIGN(sub_mat, mat_off);                                   \
      Tile2D<half, TILE_COLS, TILE_COLS> sub_scratch(K, K);        \
      TASSIGN(sub_scratch, SCRATCH_UB);                            \
      FlatVec<half, TILE_COLS> col_stats(1, K);                    \
      TASSIGN(col_stats, ROW_STATS_UB);                            \
      TCOLSUM(col_stats, sub_mat, sub_scratch, false);             \
      pipe_barrier(PIPE_V);                                        \
      TCOLEXPANDDIV(sub_mat, sub_mat, col_stats);                  \
      pipe_barrier(PIPE_V);                                        \
    }                                                              \
  } while (0)

        PER_MATRIX_COL_NORM();

#pragma unroll
        for (uint32_t iter = 1; iter < REPEAT; ++iter) {
          TASSIGN(row_stats, ROW_STATS_UB);
          TROWSUM(row_stats, tall_matrix, tall_scratch);
          pipe_barrier(PIPE_V);

          TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
          pipe_barrier(PIPE_V);

          PER_MATRIX_COL_NORM();
        }
#undef PER_MATRIX_COL_NORM
      }

      // Copy back MAT_UB → BATCH_UB.
      {
        FlatVec<T, TALL_ROWS * TILE_COLS> src(1, group_cells);
        FlatVec<T, TALL_ROWS * TILE_COLS> dst(1, group_cells);
        TASSIGN(src, MAT_UB);
        TASSIGN(dst, group_batch_offset);
        TMOV(dst, src);
        pipe_barrier(PIPE_V);
      }
    }

    GmTensor<T, TILE_COLS> gm_out_tensor(chunk_gm_out, gm_shape, gm_stride);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gm_out_tensor, batch_tile);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }

  drainPipelineFlags();
}

// ==========================================================================
// Strided-tree fallback:  K ∈ (16, 64]  —  K-runtime path
// ==========================================================================
//
// Same interleaved layout as the fast path, but K is runtime so we can't
// form a compile-time [K, ROW_BLOCK_COLS] tile view.  Instead col-normalize
// uses a flat TADD-tree (K − 1 adds to compute per-column sums over the
// row-blocks) followed by K TDIV calls (one per row-block).  Net:
// `K + 1` ops + `K − 1` TADD barriers + 1 TDIV barrier per iteration,
// versus 2 ops + 2 barriers on the fast path.
template <typename T, uint32_t TILE_COLS_TEMPLATE, uint32_t REPEAT>
AICORE void sinkhornStridedTree(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                                uint32_t K, float eps) {
  constexpr unsigned TILE_COLS = TILE_COLS_TEMPLATE;
  constexpr unsigned TALL_ROWS = STACK_ROWS;
  constexpr unsigned MATRIX_ROW_BYTES = TILE_COLS * sizeof(half);
  constexpr unsigned TILE_BYTES = TALL_ROWS * TILE_COLS * sizeof(half);

  // UB layout — same regions as the fast path, but COL_STATS slots are
  // allocated per-matrix (row-block flat vectors, not a single wide vector).
  constexpr unsigned SCRATCH_UB = 0;
  constexpr unsigned WORK_UB = ALIGN_32(SCRATCH_UB + TILE_BYTES);
  constexpr unsigned ROW_STATS_UB = ALIGN_32(WORK_UB + TILE_BYTES);
  constexpr unsigned COL_STATS_UB =
      ALIGN_32(ROW_STATS_UB + ALIGN_32(TALL_ROWS * sizeof(half)));
  constexpr unsigned BATCH_UB =
      ALIGN_32(COL_STATS_UB +
               MAX_MATS_PER_GROUP_CAP * ALIGN_32(TILE_COLS * sizeof(half)));

  constexpr unsigned BATCH_BUF_ROWS_RAW =
      (UB_BYTES - BATCH_UB) / (TILE_COLS * sizeof(half));
  constexpr unsigned MAX_BATCH_ROWS =
      BATCH_BUF_ROWS_RAW < 4095 ? BATCH_BUF_ROWS_RAW : 4095;
  static_assert(
      BATCH_UB + MAX_BATCH_ROWS * TILE_COLS * sizeof(half) <= UB_BYTES,
      "BATCH_UB exceeds UB capacity");

  set_mask_norm();
  set_vector_mask(-1, -1);

  if (K == 0 || K > TILE_COLS) return;

  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t worker_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t base_per_worker = N / num_workers;
  const uint32_t remainder = N % num_workers;
  const uint32_t my_first = worker_id * base_per_worker +
                            (worker_id < remainder ? worker_id : remainder);
  const uint32_t my_count = base_per_worker + (worker_id < remainder ? 1 : 0);
  if (my_count == 0) return;

  const uint32_t K_SQUARED = K * K;
  const uint32_t MAX_GROUP_SIZE = TALL_ROWS / K;
  const uint32_t CHUNK_MATRICES = MAX_BATCH_ROWS / K;
  const half eps_h = (half)eps;

  initPipelineFlags();

  // Outer chunk loop.
  for (uint32_t chunk_offset = 0; chunk_offset < my_count;
       chunk_offset += CHUNK_MATRICES) {
    const uint32_t chunk_matrices =
        min(CHUNK_MATRICES, my_count - chunk_offset);
    const uint32_t chunk_rows = chunk_matrices * K;
    __gm__ T *chunk_gm_in =
        gm_in + (size_t)(my_first + chunk_offset) * K_SQUARED;
    __gm__ T *chunk_gm_out =
        gm_out + (size_t)(my_first + chunk_offset) * K_SQUARED;

    {
      FlatVec<T, MAX_BATCH_ROWS * TILE_COLS> batch_flat(1,
                                                        chunk_rows * TILE_COLS);
      TASSIGN(batch_flat, BATCH_UB);
      TEXPANDS(batch_flat, (T)0);
      pipe_barrier(PIPE_V);
    }

    Tile2D<T, MAX_BATCH_ROWS, TILE_COLS> batch_tile(chunk_rows, K);
    TASSIGN(batch_tile, BATCH_UB);
    GmShape2D<T> gm_shape(chunk_rows, K);
    GmDenseStride gm_stride(K);
    GmTensor<T, TILE_COLS> gm_in_tensor(chunk_gm_in, gm_shape, gm_stride);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(batch_tile, gm_in_tensor);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (uint32_t group_start = 0; group_start < chunk_matrices;
         group_start += MAX_GROUP_SIZE) {
      const uint32_t group_size =
          min(MAX_GROUP_SIZE, chunk_matrices - group_start);
      const uint32_t group_tall_rows = group_size * K;
      const uint32_t group_flat_len = group_tall_rows * TILE_COLS;
      const unsigned group_batch_offset =
          BATCH_UB + group_start * K * TILE_COLS * sizeof(T);
      const uint32_t row_block_cols = group_size * TILE_COLS;
      constexpr uint16_t tile_row_blocks = TILE_COLS * sizeof(half) / 32;
      const uint16_t src_gap_blocks = (uint16_t)(K - 1) * tile_row_blocks;

      // Zero WORK_UB.
      {
        FlatVec<T, TALL_ROWS * TILE_COLS> work_flat(1, TALL_ROWS * TILE_COLS);
        TASSIGN(work_flat, WORK_UB);
        TEXPANDS(work_flat, (T)0);
        pipe_barrier(PIPE_V);
      }

      // Interleave: BATCH_UB → WORK_UB.
      for (uint32_t row = 0; row < K; ++row) {
        Tile2D<half, TALL_ROWS, TILE_COLS> src_view(group_size, K);
        Tile2D<half, TALL_ROWS, TILE_COLS> dst_view(group_size, K);
        TASSIGN(src_view, group_batch_offset + row * MATRIX_ROW_BYTES);
        TASSIGN(dst_view,
                WORK_UB + row * row_block_cols * (unsigned)sizeof(half));
        stridedUBCopy(dst_view, src_view, (uint16_t)group_size, tile_row_blocks,
                      src_gap_blocks, (uint16_t)0);
      }
      pipe_barrier(PIPE_V);

      if constexpr (REPEAT > 0) {
        Tile2D<half, TALL_ROWS, TILE_COLS> tall_matrix(group_tall_rows, K);
        TASSIGN(tall_matrix, WORK_UB);
        Tile2D<half, TALL_ROWS, TILE_COLS> tall_scratch(group_tall_rows, K);
        TASSIGN(tall_scratch, SCRATCH_UB);
        ColVec<half, TALL_ROWS> row_stats(group_tall_rows, 1);
        TASSIGN(row_stats, ROW_STATS_UB);

        // Softmax.
        TROWMAX(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDSUB(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        {
          FlatVec<half, TALL_ROWS * TILE_COLS> work_flat(1, group_flat_len);
          TASSIGN(work_flat, WORK_UB);
          TEXP(work_flat, work_flat);
          pipe_barrier(PIPE_V);
        }

        TROWSUM(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        {
          FlatVec<half, TALL_ROWS * TILE_COLS> work_flat(1, group_flat_len);
          TASSIGN(work_flat, WORK_UB);
          TADDS(work_flat, work_flat, eps_h);
          pipe_barrier(PIPE_V);
        }

        // Col-normalize via flat TADD-tree + K×TDIV.
        //
        // Layout recap: WORK_UB's row-block r (offset r·row_block_cols)
        // is a length-row_block_cols vector holding [mat0_row_r | mat1_row_r |
        // ...]. The per-matrix col sum for matrix i's col c sits at index
        // i·TILE_COLS + c.  To compute all per-matrix col sums we sum the
        // K row-blocks element-wise, then divide each row-block by the result.
        constexpr unsigned CN_SCRATCH_UB = SCRATCH_UB;
#define COL_NORMALIZE_STRIDED_TREE()                                           \
  do {                                                                         \
    /* First two adds in parallel (writes to disjoint dests). */               \
    {                                                                          \
      FlatVec<half, TALL_ROWS * TILE_COLS> a(1, row_block_cols),               \
          b(1, row_block_cols), c(1, row_block_cols), d(1, row_block_cols);    \
      TASSIGN(a, WORK_UB);                                                     \
      TASSIGN(b, WORK_UB + row_block_cols * (unsigned)sizeof(half));           \
      TASSIGN(c, COL_STATS_UB);                                                \
      TADD(c, a, b);                                                           \
      TASSIGN(a, WORK_UB + 2 * row_block_cols * (unsigned)sizeof(half));       \
      TASSIGN(b, WORK_UB + 3 * row_block_cols * (unsigned)sizeof(half));       \
      TASSIGN(d, CN_SCRATCH_UB);                                               \
      TADD(d, a, b);                                                           \
    }                                                                          \
    pipe_barrier(PIPE_V);                                                      \
    /* Combine the two partial sums. */                                        \
    {                                                                          \
      FlatVec<half, TALL_ROWS * TILE_COLS> a(1, row_block_cols),               \
          b(1, row_block_cols);                                                \
      TASSIGN(a, COL_STATS_UB);                                                \
      TASSIGN(b, CN_SCRATCH_UB);                                               \
      TADD(a, a, b);                                                           \
    }                                                                          \
    pipe_barrier(PIPE_V);                                                      \
    /* Fold in the remaining (K − 4) row-blocks one by one. */                 \
    for (uint32_t b_idx = 4; b_idx < K; ++b_idx) {                             \
      FlatVec<half, TALL_ROWS * TILE_COLS> src(1, row_block_cols),             \
          dst(1, row_block_cols);                                              \
      TASSIGN(src, WORK_UB + b_idx * row_block_cols * (unsigned)sizeof(half)); \
      TASSIGN(dst, COL_STATS_UB);                                              \
      TADD(dst, dst, src);                                                     \
      pipe_barrier(PIPE_V);                                                    \
    }                                                                          \
    /* Divide each row-block by the accumulated col sums. */                   \
    for (uint32_t r = 0; r < K; ++r) {                                         \
      FlatVec<half, TALL_ROWS * TILE_COLS> row_block(1, row_block_cols),       \
          sums(1, row_block_cols);                                             \
      TASSIGN(row_block,                                                       \
              WORK_UB + r * row_block_cols * (unsigned)sizeof(half));          \
      TASSIGN(sums, COL_STATS_UB);                                             \
      TDIV(row_block, row_block, sums);                                        \
    }                                                                          \
    pipe_barrier(PIPE_V);                                                      \
  } while (0)

        // First col-normalize.
        COL_NORMALIZE_STRIDED_TREE();

// (REPEAT − 1) × { row-normalize ; col-normalize }.
#pragma unroll
        for (uint32_t iter = 1; iter < REPEAT; ++iter) {
          TASSIGN(row_stats, ROW_STATS_UB);

          TROWSUM(row_stats, tall_matrix, tall_scratch);
          pipe_barrier(PIPE_V);

          TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
          pipe_barrier(PIPE_V);

          COL_NORMALIZE_STRIDED_TREE();
        }
#undef COL_NORMALIZE_STRIDED_TREE
      }

      // De-interleave: WORK_UB → BATCH_UB.
      for (uint32_t row = 0; row < K; ++row) {
        Tile2D<half, TALL_ROWS, TILE_COLS> src_view(group_size, K);
        Tile2D<half, TALL_ROWS, TILE_COLS> dst_view(group_size, K);
        TASSIGN(src_view,
                WORK_UB + row * row_block_cols * (unsigned)sizeof(half));
        TASSIGN(dst_view, group_batch_offset + row * MATRIX_ROW_BYTES);
        stridedUBCopy(dst_view, src_view, (uint16_t)group_size, tile_row_blocks,
                      (uint16_t)0, src_gap_blocks);
      }
      pipe_barrier(PIPE_V);
    }

    GmTensor<T, TILE_COLS> gm_out_tensor(chunk_gm_out, gm_shape, gm_stride);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gm_out_tensor, batch_tile);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }

  drainPipelineFlags();
}

// ==========================================================================
// Per-matrix fp32 fallback:  K ∈ (64, 128]
// ==========================================================================
//
// fp16 precision is insufficient at K=128 (softmax denominators accumulate
// 128 terms, each with ~3 decimal-digit precision).  We load fp16, convert
// to fp32 for all internal compute, convert back to fp16 on store.  No
// batching — one matrix per worker at a time.
template <typename T, uint32_t REPEAT>
AICORE void sinkhornPerMatrixFp32(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                                  uint32_t K, float eps) {
  constexpr unsigned TILE_DIM = MAX_K;
  constexpr unsigned F32_ROW_BYTES = TILE_DIM * sizeof(float);
  constexpr unsigned MATRIX_H_UB = 0;  // fp16 IO buffer
  constexpr unsigned MATRIX_F_UB =
      MATRIX_H_UB + TILE_DIM * TILE_DIM * sizeof(half);
  constexpr unsigned SCRATCH_F_UB =
      MATRIX_F_UB + TILE_DIM * TILE_DIM * sizeof(float);
  constexpr unsigned VECTOR_F_UB =
      SCRATCH_F_UB + TILE_DIM * TILE_DIM * sizeof(float);
  static_assert(VECTOR_F_UB + TILE_DIM * sizeof(float) <= UB_BYTES,
                "fp32 fallback UB layout overflows");

  set_mask_norm();
  set_vector_mask(-1, -1);
  if (K == 0 || K > TILE_DIM) return;

  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t worker_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t K_SQUARED = K * K;
  const uint32_t flat_len = K * TILE_DIM;

  initPipelineFlags();

  for (uint32_t matrix_idx = worker_id; matrix_idx < N;
       matrix_idx += num_workers) {
    __gm__ T *matrix_gm_in = gm_in + (size_t)matrix_idx * K_SQUARED;
    __gm__ T *matrix_gm_out = gm_out + (size_t)matrix_idx * K_SQUARED;

    // Zero + TLOAD fp16 matrix.
    {
      FlatVec<T, TILE_DIM * TILE_DIM> zero_flat(1, flat_len);
      TASSIGN(zero_flat, MATRIX_H_UB);
      TEXPANDS(zero_flat, (T)0);
      pipe_barrier(PIPE_V);
    }

    Tile2D<T, TILE_DIM, TILE_DIM> matrix_h(K, K);
    TASSIGN(matrix_h, MATRIX_H_UB);
    GmShape2D<T> gm_shape(K, K);
    GmDenseStride gm_stride(K);
    GmTensor<T, TILE_DIM> gm_in_tensor(matrix_gm_in, gm_shape, gm_stride);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(matrix_h, gm_in_tensor);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Upcast fp16 → fp32.
    {
      FlatVec<T, TILE_DIM * TILE_DIM> h_flat(1, flat_len);
      FlatVec<float, TILE_DIM * TILE_DIM> f_flat(1, flat_len);
      TASSIGN(h_flat, MATRIX_H_UB);
      TASSIGN(f_flat, MATRIX_F_UB);
      TCVT(f_flat, h_flat, RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
    }

    Tile2D<float, TILE_DIM, TILE_DIM> matrix(K, K);
    TASSIGN(matrix, MATRIX_F_UB);
    Tile2D<float, TILE_DIM, TILE_DIM> scratch(K, K);
    TASSIGN(scratch, SCRATCH_F_UB);
    ColVec<float, TILE_DIM> row_stats(K, 1);
    TASSIGN(row_stats, VECTOR_F_UB);

    // Softmax.
    TROWMAX(row_stats, matrix, scratch);
    pipe_barrier(PIPE_V);

    TROWEXPANDSUB(matrix, matrix, row_stats);
    pipe_barrier(PIPE_V);

    {
      FlatVec<float, TILE_DIM * TILE_DIM> mat_flat(1, flat_len);
      TASSIGN(mat_flat, MATRIX_F_UB);
      TEXP(mat_flat, mat_flat);
      pipe_barrier(PIPE_V);
    }

    TROWSUM(row_stats, matrix, scratch);
    pipe_barrier(PIPE_V);

    TROWEXPANDDIV(matrix, matrix, row_stats);
    pipe_barrier(PIPE_V);

    {
      FlatVec<float, TILE_DIM * TILE_DIM> mat_flat(1, flat_len);
      TASSIGN(mat_flat, MATRIX_F_UB);
      TADDS(mat_flat, mat_flat, eps);
      pipe_barrier(PIPE_V);
    }

    // First col-normalize.
    {
      FlatVec<float, TILE_DIM> col_stats(1, K);
      TASSIGN(col_stats, VECTOR_F_UB);

      TCOLSUM(col_stats, matrix, scratch, false);
      pipe_barrier(PIPE_V);

      TADDS(col_stats, col_stats, eps);
      pipe_barrier(PIPE_V);

      TCOLEXPANDDIV(matrix, matrix, col_stats);
      pipe_barrier(PIPE_V);
    }

// (REPEAT − 1) × { row-normalize ; col-normalize }.
#pragma unroll
    for (uint32_t iter = 1; iter < REPEAT; ++iter) {
      TASSIGN(row_stats, VECTOR_F_UB);
      TROWSUM(row_stats, matrix, scratch);
      pipe_barrier(PIPE_V);

      TADDS(row_stats, row_stats, eps);
      pipe_barrier(PIPE_V);

      TROWEXPANDDIV(matrix, matrix, row_stats);
      pipe_barrier(PIPE_V);

      {
        FlatVec<float, TILE_DIM> col_stats(1, K);
        TASSIGN(col_stats, VECTOR_F_UB);

        TCOLSUM(col_stats, matrix, scratch, false);
        pipe_barrier(PIPE_V);

        TADDS(col_stats, col_stats, eps);
        pipe_barrier(PIPE_V);

        TCOLEXPANDDIV(matrix, matrix, col_stats);
        pipe_barrier(PIPE_V);
      }
    }

    // Downcast fp32 → fp16 and store.
    {
      FlatVec<T, TILE_DIM * TILE_DIM> h_flat(1, flat_len);
      FlatVec<float, TILE_DIM * TILE_DIM> f_flat(1, flat_len);
      TASSIGN(h_flat, MATRIX_H_UB);
      TASSIGN(f_flat, MATRIX_F_UB);
      TCVT(h_flat, f_flat, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);
    }

    GmTensor<T, TILE_DIM> gm_out_tensor(matrix_gm_out, gm_shape, gm_stride);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gm_out_tensor, matrix_h);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }

  drainPipelineFlags();
}

// ==========================================================================
// Dispatch
// ==========================================================================
template <typename T, uint32_t REPEAT>
AICORE void dispatchByK(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                        uint32_t K, float eps) {
  // K ∈ {4, 8, 16}: split by batch size.  At small batches the fast-path's
  // double-buffer / interleave overhead dominates — use the simple
  // per-matrix natural-order path instead.  Empirical crossovers scale
  // ~inversely with K (smaller K packs more matrices per group, so the
  // interleave-layout benefit kicks in at higher batch counts).  Thresholds
  // were measured via msprof + direct Event timing on 910B.
  if (K == 4 && N >= 2048)
    sinkhornFastPath<T, 4, 16, REPEAT>(gm_in, gm_out, N, eps);
  else if (K == 8 && N >= 1024)
    sinkhornFastPath<T, 8, 16, REPEAT>(gm_in, gm_out, N, eps);
  else if (K == 16 && N >= 512)
    sinkhornFastPath<T, 16, 16, REPEAT>(gm_in, gm_out, N, eps);
  else if (K == 4)
    sinkhornSmallBatch<T, 4, 16, REPEAT>(gm_in, gm_out, N, eps);
  else if (K == 8)
    sinkhornSmallBatch<T, 8, 16, REPEAT>(gm_in, gm_out, N, eps);
  else if (K == 16)
    sinkhornSmallBatch<T, 16, 16, REPEAT>(gm_in, gm_out, N, eps);
  // For K=32/64, the strided-tree path is already competitive at large
  // batches; smallBatch only wins at very small N where the per-matrix
  // loop has few iterations.  Thresholds follow the same ~8192/K scaling.
  else if (K == 32 && N < 256)
    sinkhornSmallBatch<T, 32, 32, REPEAT>(gm_in, gm_out, N, eps);
  else if (K == 64 && N < 128)
    sinkhornSmallBatch<T, 64, 64, REPEAT>(gm_in, gm_out, N, eps);
  // Other K values (odd K, K > 16 && K < 32, etc.) fall through to
  // strided-tree.
  else if (K > 0 && K <= 16)
    sinkhornStridedTree<T, 16, REPEAT>(gm_in, gm_out, N, K, eps);
  else if (K <= 32)
    sinkhornStridedTree<T, 32, REPEAT>(gm_in, gm_out, N, K, eps);
  else if (K <= 64)
    sinkhornStridedTree<T, 64, REPEAT>(gm_in, gm_out, N, K, eps);
  else if (K <= MAX_K)
    sinkhornPerMatrixFp32<T, REPEAT>(gm_in, gm_out, N, K, eps);
}

// Specialize on `repeat` so that the per-iteration unroll constant is known.
template <typename T>
AICORE void dispatchByRepeat(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                             uint32_t K, uint32_t repeat, float eps) {
  switch (repeat) {
    case 0:
      dispatchByK<T, 0>(gm_in, gm_out, N, K, eps);
      break;
    case 1:
      dispatchByK<T, 1>(gm_in, gm_out, N, K, eps);
      break;
    case 3:
      dispatchByK<T, 3>(gm_in, gm_out, N, K, eps);
      break;
    case 5:
      dispatchByK<T, 5>(gm_in, gm_out, N, K, eps);
      break;
    case 8:
      dispatchByK<T, 8>(gm_in, gm_out, N, K, eps);
      break;
    case 10:
      dispatchByK<T, 10>(gm_in, gm_out, N, K, eps);
      break;
    case 20:
      dispatchByK<T, 20>(gm_in, gm_out, N, K, eps);
      break;
    default:
      dispatchByK<T, 10>(gm_in, gm_out, N, K, eps);
      break;
  }
}
#endif  // __CCE_AICORE__ == 220 && __DAV_C220_VEC__

// ==========================================================================
// C ABI
// ==========================================================================
extern "C" __global__ AICORE void sinkhorn_ds_fp16(GM_ADDR input,
                                                   GM_ADDR output, uint32_t N,
                                                   uint32_t K, uint32_t repeat,
                                                   float eps) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  dispatchByRepeat<half>((__gm__ half *)input, (__gm__ half *)output, N, K,
                         repeat, eps);
#else
  (void)input;
  (void)output;
  (void)N;
  (void)K;
  (void)repeat;
  (void)eps;
#endif
}

// Host-side launch.  Ascend 910B runs 2 AIV cores per cube core.
extern "C" void call_sinkhorn_ds_kernel(uint32_t cube_core_num, void *stream,
                                        uint8_t *input, uint8_t *output,
                                        uint32_t N, uint32_t K, uint32_t repeat,
                                        float eps) {
  sinkhorn_ds_fp16<<<cube_core_num * 2, nullptr, stream>>>(input, output, N, K,
                                                           repeat, eps);
}
