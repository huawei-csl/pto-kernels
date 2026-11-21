/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#include <type_traits>

#include "kernel_utils.h"

using namespace pto;

namespace {
// L0A and L0C are addressed in 16-row fractals whatever m the mad uses.
constexpr uint32_t kFractalRows = 16;

// Each block reduces its S x S tiles through the cube unit's GEMV mode. A mad
// issued with m == 1 takes its left operand as a single row rather than a
// 16-row fractal, so an all-ones row costs one column of MACs per input
// element. That puts the tile on the right:
//
//     ones[1, S] @ A[S, S] = the column sums of A, a 1 x S row.
//
// Each block therefore leaves the Vector stage S values to fold.
template <typename InputT, typename OutputT, uint32_t S>
AICORE void CubeReduce(__gm__ InputT* input, __gm__ OutputT* partials,
                       uint32_t num_tiles) {
  using AGlobal = GlobalTensor<InputT, TileShape2D<InputT, S, S, Layout::ND>,
                               BaseShape2D<InputT, S, S, Layout::ND>>;
  using CGlobal = GlobalTensor<OutputT, TileShape2D<OutputT, 1, S, Layout::ND>,
                               BaseShape2D<OutputT, 1, S, Layout::ND>>;
  using AMat = Tile<TileType::Mat, InputT, S, S, BLayout::ColMajor, S, S,
                    SLayout::RowMajor, 512>;
  using OnesMat =
      Tile<TileType::Mat, InputT, kFractalRows, S, BLayout::ColMajor,
           kFractalRows, S, SLayout::RowMajor, 512>;
  // With nothing but a ragged tail the Vector stage does all the work and
  // never reads `partials`, so this stage has no result to produce.
  if (num_tiles == 0) return;

  AMat a_l1;
  OnesMat ones_l1;
  // GEMV reads only row 0 of L0A, but L0A is still addressed in 16-row
  // fractals, so the operand is declared full height and filled with ones
  // throughout. L0C is addressed the same way: mad writes its block columns at
  // a pitch of ceil16(m) == 16 rows, and only row 0 of each carries a result.
  TileRight<InputT, S, S> a_l0;
  TileLeft<InputT, kFractalRows, S> ones_l0;
  TileAcc<OutputT, kFractalRows, S, 1, S> c_l0;
  TASSIGN(a_l1, 0);
  TASSIGN(ones_l1, S * S * sizeof(InputT));
  TASSIGN(a_l0, 0);
  TASSIGN(ones_l0, 0);
  TASSIGN(c_l0, 0);

  const uint32_t id = get_block_idx();
  const uint32_t blocks = get_block_num();
  const uint32_t count = num_tiles / blocks + (id < num_tiles % blocks);
  // Balanced contiguous ranges ensure every launched block writes a result.
  const uint32_t begin =
      id * (num_tiles / blocks) + min(id, num_tiles % blocks);
  // Build the all-ones operand on chip instead of copying it in from global
  // memory: on a Mat tile TEXPANDS lowers to create_cbuf_matrix, which fills
  // L1 directly. Every element is one, so the fractal layout of the tile is
  // irrelevant and the flat fill is already the matrix we want. Like TLOAD it
  // issues on MTE2, so the MTE2 -> MTE1 flag below still orders it before the
  // move into L0A.
  TEXPANDS(ones_l1, static_cast<InputT>(1));
  kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(ones_l0, ones_l1);
  kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  for (uint32_t i = 0; i < count; ++i) {
    AGlobal a_global(input + static_cast<uint64_t>(begin + i) * S * S);
    TLOAD(a_l1, a_global);
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TMOV(a_l0, a_l1);
    kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
    if (i == 0) {
      TGEMV(c_l0, ones_l0, a_l0);
    } else {
      TGEMV_ACC(c_l0, c_l0, ones_l0, a_l0);
    }
    // Complete the multiply before the next TMOV overwrites a_l0, and the
    // TMOV's read of a_l1 before the next TLOAD refills it.
    kernel_utils::SetWaitFlag<PIPE_M, PIPE_MTE1>(0);
    kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
  }
  kernel_utils::SetWaitFlag<PIPE_M, PIPE_FIX>(0);
  CGlobal c_global(partials + id * S);
  TSTORE(c_global, c_l0);
  pipe_barrier(PIPE_ALL);
}

// A separate launch on the same stream orders all Fixpipe stores before
// these reads. One AIV folds every block into one running lane, with no
// atomics or device-wide barriers. UB use is independent of the block count.
// Accumulator type the Cube stage writes for a given input type.
template <typename InputT>
using CubeOutT = typename std::conditional<std::is_same<InputT, int8_t>::value,
                                           int32_t, float>::type;

// Sums the `tail_len` elements at `input + tail_offset` and returns the total.
//
// The Cube stage consumes only whole S x S tiles, so the last `tail_len`
// elements (fewer than one tile) are reduced here instead of being padded up
// to a tile anywhere. A Vec tile carries a runtime valid-column count and
// TROWSUM honours it, so a short final chunk is loaded and summed as it is.
//
// TROWSUM on this part requires its accumulator and its source to have the
// same type, so every chunk is widened to fp32 first. That also keeps the
// tail's arithmetic as wide as the fp32 L0C accumulation the Cube stage
// does, rather than summing an fp16 tail in fp16. int8 has no direct
// conversion to fp32 and goes via half, which is exact for its range. The
// tail is shorter than one tile, so even an int8 tail of extreme values stays
// well inside the range fp32 represents exactly.
//
// `kUbBase` is the first UB byte this may use; it leaves everything below
// untouched. The V -> S flag it ends on also orders the caller's own vector
// accumulation before the caller reads it back, since that precedes the
// vector work here.
template <typename InputT, uint32_t kUbBase>
AICORE float AccumulateTail(__gm__ InputT* input, uint64_t tail_offset,
                            uint32_t tail_len) {
  constexpr bool kIsF32 = std::is_same<InputT, float>::value;
  constexpr bool kIsI8 = std::is_same<InputT, int8_t>::value;
  constexpr uint32_t kChunk = 256;
  constexpr uint32_t kInBytes = kChunk * sizeof(InputT);
  constexpr uint32_t kMidBytes = kIsI8 ? kChunk * sizeof(half) : 0;
  constexpr uint32_t kF32Off = kIsF32 ? 0 : kInBytes + kMidBytes;
  constexpr uint32_t kF32Bytes = kChunk * sizeof(float);
  using TailGlobal = GlobalTensor<InputT, Shape<1, 1, 1, 1, DYNAMIC>,
                                  pto::Stride<1, 1, 1, 1, 1>>;
  using TailIn =
      Tile<TileType::Vec, InputT, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;
  using TailMid =
      Tile<TileType::Vec, half, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;
  using TailF32 =
      Tile<TileType::Vec, float, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;
  Tile<TileType::Vec, float, 1, kChunk> tail_tmp;
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 1> tail_sum;
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 1> tail_acc;
  TASSIGN(tail_tmp, kUbBase + kF32Off + kF32Bytes);
  TASSIGN(tail_sum, kUbBase + kF32Off + 2 * kF32Bytes);
  TASSIGN(tail_acc, kUbBase + kF32Off + 2 * kF32Bytes + 32);

  TEXPANDS(tail_acc, 0.0f);
  kernel_utils::PipeBarrierVec();
  for (uint32_t off = 0; off < tail_len; off += kChunk) {
    const uint32_t n = min(kChunk, tail_len - off);
    TailGlobal tail_global(input + tail_offset + off,
                           {static_cast<int32_t>(n)});
    TailIn tail_in(n);
    TASSIGN(tail_in, kUbBase);
    TLOAD(tail_in, tail_global);
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    TailF32 tail_f32(n);
    TASSIGN(tail_f32, kUbBase + kF32Off);
    if constexpr (kIsI8) {
      TailMid tail_mid(n);
      TASSIGN(tail_mid, kUbBase + kInBytes);
      TCVT(tail_mid, tail_in, RoundMode::CAST_NONE);
      kernel_utils::PipeBarrierVec();
      TCVT(tail_f32, tail_mid, RoundMode::CAST_NONE);
      kernel_utils::PipeBarrierVec();
    } else if constexpr (!kIsF32) {
      TCVT(tail_f32, tail_in, RoundMode::CAST_NONE);
      kernel_utils::PipeBarrierVec();
    }
    TROWSUM(tail_sum, tail_f32, tail_tmp);
    kernel_utils::PipeBarrierVec();
    TADD(tail_acc, tail_acc, tail_sum);
    kernel_utils::PipeBarrierVec();
    kernel_utils::SetWaitFlag<PIPE_V, PIPE_MTE2>(0);
  }

  // GetValue uses the scalar pipe, so the vector accumulation must be visible
  // to it before the scalar read.
  kernel_utils::SetWaitFlag<PIPE_V, PIPE_S>(0);
  return tail_acc.GetValue(0);
}

// The partials are one contiguous run of `blocks * S` values, folded as a flat
// vector in chunks, the way the tail is. The accumulation stays in T: an int8
// reduction can exceed what fp32 holds exactly, and the test for a full tensor
// of -128 depends on it being exact.
template <typename InputT, typename T, uint32_t S>
AICORE void CompleteCubeReduce(__gm__ InputT* input, __gm__ T* partials,
                               __gm__ T* output, uint32_t blocks,
                               uint64_t tail_offset, uint32_t tail_len) {
  using OutputGlobal =
      GlobalTensor<T, Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>>;
  using PartGlobal =
      GlobalTensor<T, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;
  constexpr uint32_t kChunk = 256;
  constexpr uint32_t kChunkBytes = kChunk * sizeof(T);
  using PartIn =
      Tile<TileType::Vec, T, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;
  Tile<TileType::Vec, T, 1, kChunk> part_tmp;
  Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, 1, 1> part_sum;
  Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, 1, 1> totals;
  Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, 1, 1> result;
  constexpr uint32_t kTailBase = 2 * kChunkBytes + 3 * 32;

  set_mask_norm();
  set_vector_mask(-1, -1);

  TASSIGN(part_tmp, kChunkBytes);
  TASSIGN(part_sum, 2 * kChunkBytes);
  TASSIGN(totals, 2 * kChunkBytes + 32);
  TASSIGN(result, 2 * kChunkBytes + 64);
  TEXPANDS(totals, static_cast<T>(0));
  kernel_utils::PipeBarrierVec();
  const uint32_t partial_len = blocks * S;
  for (uint32_t off = 0; off < partial_len; off += kChunk) {
    const uint32_t n = min(kChunk, partial_len - off);
    PartGlobal source(partials + off, {static_cast<int32_t>(n)});
    PartIn part_in(n);
    TASSIGN(part_in, 0);
    TLOAD(part_in, source);
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    TROWSUM(part_sum, part_in, part_tmp);
    kernel_utils::PipeBarrierVec();
    TADD(totals, totals, part_sum);
    kernel_utils::PipeBarrierVec();
    kernel_utils::SetWaitFlag<PIPE_V, PIPE_MTE2>(0);
  }
  const float tail =
      AccumulateTail<InputT, kTailBase>(input, tail_offset, tail_len);
  result.SetValue(0, totals.GetValue(0) + static_cast<T>(tail));
  kernel_utils::SetWaitFlag<PIPE_S, PIPE_MTE3>(0);
  OutputGlobal destination(output);
  TSTORE(destination, result);
  pipe_barrier(PIPE_ALL);
}

// ---------------------------------------------------------------------------
// Reduction over the last dimension of an ND contiguous tensor.
//
// A contiguous tensor of shape (d0, ..., dk, N) is exactly a (rows x N) matrix
// with rows = d0 * ... * dk, so summing the last axis is one row sum per
// matrix row. The full-tensor reduction above puts the data on the right of
// the mad and an all-ones row on the left; a row sum is the mirror image:
//
//     A[M, K] @ ones[K, 1] = the row sums of A, an M x 1 column.
//
// A mad costs ceil(m/16) * ceil(k/16) * ceil(n/16) issues whatever n is, so
// n == 1 reads M * K inputs for the same price as a 16-wide mad: the Cube unit
// is left with far more throughput than HBM can feed it, which is what makes a
// one-column GEMM a sensible reducer.
// ---------------------------------------------------------------------------

// L0B holds the ones operand as full 16-column fractals whatever n the mad
// uses, mirroring the full-height L0A operand of the reduction above.
constexpr uint32_t kFractalCols = 16;

// Sums `cols` elements of each of `rows` contiguous rows into `dest`.
//
// `M` rows and `K` columns per tile, both template parameters so the tile
// shapes stay compile-time constants. The mad's m and the L0C row pitch both
// round up to 16, so a 16-row tile wastes no Cube cycles relative to a
// 128-row one and only pays more Fixpipe stores; `K` trades the length of an
// MTE2 burst against the padding the final column tile mads through. The host
// picks the pair.
//
// Work is split into `row_tiles * splits` items: one per (row tile, column
// range) pair. Splitting the column range matters when the input has fewer row
// tiles than there are cores -- a few very long rows -- and `splits` is 1
// otherwise, in which case `dest` is the final output rather than a partials
// buffer and no Vector stage runs at all.
template <typename InputT, typename OutputT, uint32_t M, uint32_t K>
AICORE void CubeRowReduce(__gm__ InputT* input, __gm__ OutputT* dest,
                          uint32_t rows, uint32_t cols, uint32_t splits) {
  using AGlobal = GlobalTensor<InputT, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                               pto::Stride<1, 1, 1, DYNAMIC, 1>, Layout::ND>;
  using CGlobal = GlobalTensor<OutputT, Shape<1, 1, 1, DYNAMIC, 1>,
                               pto::Stride<1, 1, 1, 1, 1>, Layout::ND>;
  // The staging tile for A is zN, the layout TLOAD converts ND global data
  // into, and the one TMOV forwards to L0A without transposing.
  using AMat = Tile<TileType::Mat, InputT, M, K, BLayout::ColMajor, DYNAMIC,
                    DYNAMIC, SLayout::RowMajor, 512>;
  using AMatWhole = Tile<TileType::Mat, InputT, M, K, BLayout::ColMajor, M, K,
                         SLayout::RowMajor, 512>;
  // The ones operand is staged nZ instead, which is the fractal layout L0B
  // wants, so its TMOV does not transpose either.
  using OnesMat =
      Tile<TileType::Mat, InputT, K, kFractalCols, BLayout::RowMajor, K,
           kFractalCols, SLayout::ColMajor, 512>;

  constexpr uint32_t kTileBytes = M * K * sizeof(InputT);
  const uint32_t row_tiles = kernel_utils::CeilDiv(rows, M);
  const uint32_t col_tiles = kernel_utils::CeilDiv(cols, K);
  const uint32_t items = row_tiles * splits;
  const uint32_t id = get_block_idx();
  const uint32_t blocks = get_block_num();
  // The host caps the launch at `items`, so every block owns at least one.
  if (id >= items) return;
  const uint32_t count = items / blocks + (id < items % blocks);
  const uint32_t begin = id * (items / blocks) + min(id, items % blocks);

  // Two staging buffers rather than one: the last column tile of a row is
  // short, and the columns past it must read as zero because the mad always
  // runs the full width K. Giving the short tile its own buffer means zeroing
  // it once here -- every later TLOAD into it rewrites the same leading
  // columns and leaves the zeroed remainder alone.
  AMatWhole tail_init;
  OnesMat ones_l1;
  TileLeft<InputT, M, K> a_l0;
  TileRight<InputT, K, kFractalCols, K, 1> ones_l0;
  TASSIGN(tail_init, kTileBytes);
  TASSIGN(ones_l1, 2 * kTileBytes);
  TASSIGN(a_l0, 0);
  TASSIGN(ones_l0, 0);

  // TEXPANDS on a Mat tile lowers to create_cbuf_matrix, which fills L1
  // directly. It writes through a different tile object than the TLOADs that
  // later refill the same buffer, so nothing but this barrier keeps the fill
  // from landing on top of loaded data -- which it does, intermittently and
  // one C0 block at a time, when a block's first column tile is the short
  // one. The barrier costs one drain per kernel, before any real work.
  TEXPANDS(ones_l1, static_cast<InputT>(1));
  if (cols % K != 0) TEXPANDS(tail_init, static_cast<InputT>(0));
  pipe_barrier(PIPE_ALL);
  kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(ones_l0, ones_l1);
  kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t item = begin + i;
    const uint32_t row_begin = (item / splits) * M;
    const uint32_t valid_rows = min(M, rows - row_begin);
    const uint32_t split = item % splits;
    // Balanced contiguous column-tile ranges, as for the item ranges above.
    // The host keeps `splits` at or below `col_tiles`, so no split is empty
    // and every (split, row tile) pair writes the partial it owns.
    const uint32_t tile_begin =
        split * (col_tiles / splits) + min(split, col_tiles % splits);
    const uint32_t tile_count =
        col_tiles / splits + (split < col_tiles % splits);

    TileAcc<OutputT, M, kFractalCols, DYNAMIC, 1> c_l0(valid_rows);
    TASSIGN(c_l0, 0);
    for (uint32_t t = 0; t < tile_count; ++t) {
      const uint32_t col_begin = (tile_begin + t) * K;
      const uint32_t valid_cols = min(K, cols - col_begin);
      AGlobal a_global(
          input + static_cast<uint64_t>(row_begin) * cols + col_begin,
          {static_cast<int64_t>(valid_rows), static_cast<int64_t>(valid_cols)},
          {static_cast<int64_t>(cols)});
      AMat a_l1(valid_rows, valid_cols);
      TASSIGN(a_l1, valid_cols == K ? 0 : kTileBytes);
      TLOAD(a_l1, a_global);
      kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
      // Rows past `valid_rows` carry whatever L1 held; they produce their own
      // junk row sums in L0C and are never stored. Only the column direction
      // has to be clean, which is what the zeroed tail buffer is for.
      TMOV(a_l0, a_l1);
      kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
      if (t == 0) {
        TMATMUL(c_l0, a_l0, ones_l0);
      } else {
        TMATMUL_ACC(c_l0, c_l0, a_l0, ones_l0);
      }
      // Complete the multiply before the next TMOV overwrites a_l0, and the
      // TMOV's read of a_l1 before the next TLOAD refills it.
      kernel_utils::SetWaitFlag<PIPE_M, PIPE_MTE1>(0);
      kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    }
    kernel_utils::SetWaitFlag<PIPE_M, PIPE_FIX>(0);
    CGlobal c_global(dest + static_cast<uint64_t>(split) * rows + row_begin,
                     {static_cast<int64_t>(valid_rows)});
    TSTORE(c_global, c_l0);
    // The next row tile's mad reuses this L0C tile.
    kernel_utils::SetWaitFlag<PIPE_FIX, PIPE_M>(0);
  }
  pipe_barrier(PIPE_ALL);
}

// Folds the `splits` partial rows the Cube stage left into one row of `rows`
// sums. A separate launch on the same stream orders all Fixpipe stores before
// these reads. Only runs when the Cube stage was split along the columns.
template <typename T>
AICORE void CompleteCubeRowReduce(__gm__ T* partials, __gm__ T* output,
                                  uint32_t rows, uint32_t splits) {
  constexpr uint32_t kChunk = 512;
  constexpr uint32_t kChunkBytes = kChunk * sizeof(T);
  using RowGlobal =
      GlobalTensor<T, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;
  using RowIn =
      Tile<TileType::Vec, T, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t chunks = kernel_utils::CeilDiv(rows, kChunk);
  const uint32_t id = get_block_idx();
  const uint32_t blocks = get_block_num();
  if (id >= chunks) return;
  const uint32_t count = chunks / blocks + (id < chunks % blocks);
  const uint32_t begin = id * (chunks / blocks) + min(id, chunks % blocks);

  for (uint32_t c = 0; c < count; ++c) {
    const uint32_t off = (begin + c) * kChunk;
    const uint32_t n = min(kChunk, rows - off);
    RowIn totals(n);
    RowIn addend(n);
    TASSIGN(totals, 0);
    TASSIGN(addend, kChunkBytes);
    RowGlobal first(partials + off, {static_cast<int32_t>(n)});
    TLOAD(totals, first);
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    for (uint32_t s = 1; s < splits; ++s) {
      RowGlobal source(partials + static_cast<uint64_t>(s) * rows + off,
                       {static_cast<int32_t>(n)});
      TLOAD(addend, source);
      kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
      TADD(totals, totals, addend);
      kernel_utils::SetWaitFlag<PIPE_V, PIPE_MTE2>(0);
    }
    kernel_utils::SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
    RowGlobal destination(output + off, {static_cast<int32_t>(n)});
    TSTORE(destination, totals);
    kernel_utils::SetWaitFlag<PIPE_MTE3, PIPE_MTE2>(0);
  }
  pipe_barrier(PIPE_ALL);
}
}  // namespace

template <typename InputT>
AICORE void CubeReduceDispatch(__gm__ InputT* input,
                               __gm__ CubeOutT<InputT>* partials,
                               uint32_t num_tiles, uint32_t size) {
  using OutputT = CubeOutT<InputT>;
  switch (size) {
    case 32:
      CubeReduce<InputT, OutputT, 32>(input, partials, num_tiles);
      break;
    case 64:
      CubeReduce<InputT, OutputT, 64>(input, partials, num_tiles);
      break;
    case 128:
      CubeReduce<InputT, OutputT, 128>(input, partials, num_tiles);
      break;
  }
}

template <typename InputT>
AICORE void CompleteCubeReduceDispatch(__gm__ InputT* input,
                                       __gm__ CubeOutT<InputT>* partials,
                                       __gm__ CubeOutT<InputT>* output,
                                       uint32_t blocks, uint64_t tail_offset,
                                       uint32_t tail_len, uint32_t size) {
  using OutputT = CubeOutT<InputT>;
  switch (size) {
    case 32:
      CompleteCubeReduce<InputT, OutputT, 32>(input, partials, output, blocks,
                                              tail_offset, tail_len);
      break;
    case 64:
      CompleteCubeReduce<InputT, OutputT, 64>(input, partials, output, blocks,
                                              tail_offset, tail_len);
      break;
    case 128:
      CompleteCubeReduce<InputT, OutputT, 128>(input, partials, output, blocks,
                                               tail_offset, tail_len);
      break;
  }
}

extern "C" __global__ AICORE void cube_reduce_fp16(__gm__ half* input,
                                                   __gm__ float* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeReduceDispatch<half>(input, partials, num_tiles, size);
#endif
}

extern "C" __global__ AICORE void complete_cube_reduce_fp16(
    __gm__ half* input, __gm__ float* partials, __gm__ float* output,
    uint32_t blocks, uint64_t tail_offset, uint32_t tail_len, uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeReduceDispatch<half>(input, partials, output, blocks, tail_offset,
                                   tail_len, size);
#endif
}

extern "C" void pto_launch_cube_reduce_fp16(uint32_t blockDim, void* stream,
                                            void* input, void* partials,
                                            void* output, uint32_t num_tiles,
                                            uint32_t blocks,
                                            uint64_t tail_offset,
                                            uint32_t tail_len, uint32_t size) {
  cube_reduce_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)partials, num_tiles, size);
  complete_cube_reduce_fp16<<<1, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)partials, (__gm__ float*)output,
      blocks, tail_offset, tail_len, size);
}

extern "C" __global__ AICORE void cube_reduce_fp32(__gm__ float* input,
                                                   __gm__ float* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeReduceDispatch<float>(input, partials, num_tiles, size);
#endif
}

extern "C" __global__ AICORE void complete_cube_reduce_fp32(
    __gm__ float* input, __gm__ float* partials, __gm__ float* output,
    uint32_t blocks, uint64_t tail_offset, uint32_t tail_len, uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeReduceDispatch<float>(input, partials, output, blocks,
                                    tail_offset, tail_len, size);
#endif
}

extern "C" void pto_launch_cube_reduce_fp32(uint32_t blockDim, void* stream,
                                            void* input, void* partials,
                                            void* output, uint32_t num_tiles,
                                            uint32_t blocks,
                                            uint64_t tail_offset,
                                            uint32_t tail_len, uint32_t size) {
  cube_reduce_fp32<<<blockDim, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)partials, num_tiles, size);
  complete_cube_reduce_fp32<<<1, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)partials, (__gm__ float*)output,
      blocks, tail_offset, tail_len, size);
}

extern "C" __global__ AICORE void cube_reduce_int8(__gm__ int8_t* input,
                                                   __gm__ int32_t* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeReduceDispatch<int8_t>(input, partials, num_tiles, size);
#endif
}

extern "C" __global__ AICORE void complete_cube_reduce_int8(
    __gm__ int8_t* input, __gm__ int32_t* partials, __gm__ int32_t* output,
    uint32_t blocks, uint64_t tail_offset, uint32_t tail_len, uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeReduceDispatch<int8_t>(input, partials, output, blocks,
                                     tail_offset, tail_len, size);
#endif
}

extern "C" void pto_launch_cube_reduce_int8(uint32_t blockDim, void* stream,
                                            void* input, void* partials,
                                            void* output, uint32_t num_tiles,
                                            uint32_t blocks,
                                            uint64_t tail_offset,
                                            uint32_t tail_len, uint32_t size) {
  cube_reduce_int8<<<blockDim, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int32_t*)partials, num_tiles, size);
  complete_cube_reduce_int8<<<1, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int32_t*)partials, (__gm__ int32_t*)output,
      blocks, tail_offset, tail_len, size);
}

// A (rows per tile, columns per tile) pair is only instantiated when its
// operands fit L0A and L0B, both 64 KB. The host never asks for a pair that
// does not fit, so the dropped cases are unreachable rather than silent
// no-ops.
template <typename InputT, uint32_t M, uint32_t K>
AICORE inline void CubeRowReduceIfFits(__gm__ InputT* input,
                                       __gm__ CubeOutT<InputT>* dest,
                                       uint32_t rows, uint32_t cols,
                                       uint32_t splits) {
  if constexpr (M * K * sizeof(InputT) <= 65536 &&
                K * kFractalCols * sizeof(InputT) <= 65536) {
    CubeRowReduce<InputT, CubeOutT<InputT>, M, K>(input, dest, rows, cols,
                                                  splits);
  }
}

template <typename InputT>
AICORE void CubeRowReduceDispatch(__gm__ InputT* input,
                                  __gm__ CubeOutT<InputT>* dest, uint32_t rows,
                                  uint32_t cols, uint32_t splits,
                                  uint32_t row_tile, uint32_t col_tile) {
  if (row_tile == 16) {
    switch (col_tile) {
      case 64:
        CubeRowReduceIfFits<InputT, 16, 64>(input, dest, rows, cols, splits);
        break;
      case 128:
        CubeRowReduceIfFits<InputT, 16, 128>(input, dest, rows, cols, splits);
        break;
      case 256:
        CubeRowReduceIfFits<InputT, 16, 256>(input, dest, rows, cols, splits);
        break;
      case 512:
        CubeRowReduceIfFits<InputT, 16, 512>(input, dest, rows, cols, splits);
        break;
      case 1024:
        CubeRowReduceIfFits<InputT, 16, 1024>(input, dest, rows, cols, splits);
        break;
    }
  } else {
    switch (col_tile) {
      case 64:
        CubeRowReduceIfFits<InputT, 128, 64>(input, dest, rows, cols, splits);
        break;
      case 128:
        CubeRowReduceIfFits<InputT, 128, 128>(input, dest, rows, cols, splits);
        break;
      case 256:
        CubeRowReduceIfFits<InputT, 128, 256>(input, dest, rows, cols, splits);
        break;
      case 512:
        CubeRowReduceIfFits<InputT, 128, 512>(input, dest, rows, cols, splits);
        break;
      case 1024:
        CubeRowReduceIfFits<InputT, 128, 1024>(input, dest, rows, cols, splits);
        break;
    }
  }
}

extern "C" __global__ AICORE void cube_row_reduce_fp16(
    __gm__ half* input, __gm__ float* dest, uint32_t rows, uint32_t cols,
    uint32_t splits, uint32_t row_tile, uint32_t col_tile) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeRowReduceDispatch<half>(input, dest, rows, cols, splits, row_tile,
                              col_tile);
#endif
}

extern "C" __global__ AICORE void complete_cube_row_reduce_fp16(
    __gm__ float* partials, __gm__ float* output, uint32_t rows,
    uint32_t splits) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeRowReduce<float>(partials, output, rows, splits);
#endif
}

extern "C" void pto_launch_cube_row_reduce_fp16(
    uint32_t blockDim, void* stream, void* input, void* dest, void* output,
    uint32_t rows, uint32_t cols, uint32_t splits, uint32_t row_tile,
    uint32_t col_tile, uint32_t vector_blocks) {
  cube_row_reduce_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)dest, rows, cols, splits, row_tile,
      col_tile);
  // With a single column range the Cube stage already wrote the final sums.
  if (splits > 1) {
    complete_cube_row_reduce_fp16<<<vector_blocks, nullptr, stream>>>(
        (__gm__ float*)dest, (__gm__ float*)output, rows, splits);
  }
}

extern "C" __global__ AICORE void cube_row_reduce_fp32(
    __gm__ float* input, __gm__ float* dest, uint32_t rows, uint32_t cols,
    uint32_t splits, uint32_t row_tile, uint32_t col_tile) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeRowReduceDispatch<float>(input, dest, rows, cols, splits, row_tile,
                               col_tile);
#endif
}

extern "C" __global__ AICORE void complete_cube_row_reduce_fp32(
    __gm__ float* partials, __gm__ float* output, uint32_t rows,
    uint32_t splits) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeRowReduce<float>(partials, output, rows, splits);
#endif
}

extern "C" void pto_launch_cube_row_reduce_fp32(
    uint32_t blockDim, void* stream, void* input, void* dest, void* output,
    uint32_t rows, uint32_t cols, uint32_t splits, uint32_t row_tile,
    uint32_t col_tile, uint32_t vector_blocks) {
  cube_row_reduce_fp32<<<blockDim, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)dest, rows, cols, splits, row_tile,
      col_tile);
  if (splits > 1) {
    complete_cube_row_reduce_fp32<<<vector_blocks, nullptr, stream>>>(
        (__gm__ float*)dest, (__gm__ float*)output, rows, splits);
  }
}

extern "C" __global__ AICORE void cube_row_reduce_int8(
    __gm__ int8_t* input, __gm__ int32_t* dest, uint32_t rows, uint32_t cols,
    uint32_t splits, uint32_t row_tile, uint32_t col_tile) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeRowReduceDispatch<int8_t>(input, dest, rows, cols, splits, row_tile,
                                col_tile);
#endif
}

extern "C" __global__ AICORE void complete_cube_row_reduce_int8(
    __gm__ int32_t* partials, __gm__ int32_t* output, uint32_t rows,
    uint32_t splits) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeRowReduce<int32_t>(partials, output, rows, splits);
#endif
}

extern "C" void pto_launch_cube_row_reduce_int8(
    uint32_t blockDim, void* stream, void* input, void* dest, void* output,
    uint32_t rows, uint32_t cols, uint32_t splits, uint32_t row_tile,
    uint32_t col_tile, uint32_t vector_blocks) {
  cube_row_reduce_int8<<<blockDim, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int32_t*)dest, rows, cols, splits,
      row_tile, col_tile);
  if (splits > 1) {
    complete_cube_row_reduce_int8<<<vector_blocks, nullptr, stream>>>(
        (__gm__ int32_t*)dest, (__gm__ int32_t*)output, rows, splits);
  }
}
