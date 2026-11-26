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
// L0A, L0B and L0C are addressed in fractals whatever m, k and n the mad uses,
// so an operand narrower than a fractal is still declared a whole one wide.
// Every fractal is 16 rows tall, but its width depends on the element size and
// on the buffer: a 512-byte L0A/L0B box is 32 columns of int8 or 16 of fp16,
// while a 1024-byte L0C box is 16 columns of fp32 or int32. A tile's declared
// column count has to divide by its own box width, so 32 is the one width
// legal for every supported dtype in both buffers.
constexpr uint32_t kFractalRows = 16;
constexpr uint32_t kFractalCols = 32;

// Elements in one input tile. Both modes stage a tile of `kTileElems` through
// L1 into L0A or L0B, which hold 64 KB, so this is what an fp32 tile fills
// exactly; the narrower dtypes leave it half or a quarter used. The host picks
// only the tile's width C and this fixes its height as kTileElems / C, which
// keeps the dispatch below a single switch and keeps every tile the same
// volume, so a wider tile is always a shorter one. Both are powers of two
// between 32 and 512, so a tile is always a whole number of fractals in each
// direction.
constexpr uint32_t kTileElems = 16384;

// Which of the three decompositions the host picked. Each is a different way
// of cutting the same (outer, mid, inner) view into tiles, and the host weighs
// them against each other before launching.
constexpr uint32_t kModeStrided = 0;
constexpr uint32_t kModeContiguous = 1;
constexpr uint32_t kModeFolded = 2;

// Accumulator type the Cube stage writes for a given input type.
template <typename InputT>
using ReduceOutT =
    typename std::conditional<std::is_same<InputT, int8_t>::value, int32_t,
                              float>::type;

// A single-axis reduction of a contiguous 3D tensor is handed over as the
// (outer, mid, inner) view that always reduces `mid`: the host folds the axes
// before the reduced one into `outer` and the axes after it into `inner`, so
// dim=0 arrives as (1, D0, D1*D2), dim=1 as (D0, D1, D2) and dim=2 as
// (D0*D1, D2, 1). Element (o, i) of the result is the sum over m of
// x[o, m, i], and the result is contiguous with row length `inner`.
//
// That leaves exactly two shapes of work, and the cube unit has a mode for
// each. Both tile the reduced axis and accumulate the tiles in L0C, so the
// reduction never leaves the core and no partial results reach global memory
// -- unlike a full reduction, every output element belongs to exactly one
// work item, so there is nothing to fold across blocks afterwards.

// `inner > 1`: the reduced axis is the strided one, so a tile of the input is
// an R x C block of rows to be summed column by column. A mad issued with
// m == 1 takes its left operand as a single row rather than a 16-row fractal,
// so an all-ones row costs one column of MACs per input element:
//
//     ones[1, R] @ A[R, C] = the column sums of A, a 1 x C row.
//
// The work items are the output tiles (o, column tile). Each loops over the
// reduced axis accumulating into one L0C row, then stores `cols` values.
template <typename InputT, typename OutputT, uint32_t C>
AICORE void ReduceStridedAxis(__gm__ InputT* input, __gm__ OutputT* output,
                              uint32_t outer, uint32_t mid, uint32_t inner) {
  // R rows of the reduced axis by C columns of the kept one. C is what sets
  // the length of each contiguous run the tile load fetches, so a wide tile
  // reads the input in long strides and a narrow one in short ones.
  constexpr uint32_t R = kTileElems / C;
  // The tile is cut from a matrix whose row pitch is `inner`, and its last
  // row and column tile are ragged. TFILLPAD zeroes whatever the load left
  // short so the mad can run at a fixed R x C; it takes a Mat tile only in NZ
  // layout, which is the ColMajor/RowMajor pair below.
  using AGlobal = GlobalTensor<InputT, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                               pto::Stride<1, 1, 1, DYNAMIC, 1>>;
  using CGlobal = GlobalTensor<OutputT, Shape<1, 1, 1, 1, DYNAMIC>,
                               pto::Stride<1, 1, 1, DYNAMIC, 1>>;
  using AMat = Tile<TileType::Mat, InputT, R, C, BLayout::ColMajor, DYNAMIC,
                    DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  // The ones row is as long as the mad's k, which is the tile's row count.
  using OnesMat =
      Tile<TileType::Mat, InputT, kFractalRows, R, BLayout::ColMajor,
           kFractalRows, R, SLayout::RowMajor, 512>;
  // GEMV reads only row 0 of L0A, but L0A is still addressed in 16-row
  // fractals, so the operand is declared full height and filled with ones
  // throughout. L0C is addressed the same way: mad writes its block columns at
  // a pitch of ceil16(m) == 16 rows, and only row 0 of each carries a result.
  using CAcc = TileAcc<OutputT, kFractalRows, C, DYNAMIC, DYNAMIC>;

  const uint32_t col_tiles = kernel_utils::CeilDiv(inner, C);
  const uint32_t work = outer * col_tiles;
  if (work == 0) return;

  OnesMat ones_l1;
  TileLeft<InputT, kFractalRows, R> ones_l0;
  TileRight<InputT, R, C> a_l0;
  TASSIGN(ones_l1, R * C * sizeof(InputT));
  TASSIGN(ones_l0, 0);
  TASSIGN(a_l0, 0);
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

  const uint32_t id = get_block_idx();
  const uint32_t blocks = get_block_num();
  const uint32_t count = work / blocks + (id < work % blocks);
  // Balanced contiguous ranges ensure every launched block writes a result.
  const uint32_t begin = id * (work / blocks) + min(id, work % blocks);
  // Walking (o, ci) forward keeps one division off the per-item path.
  uint32_t o = begin / col_tiles;
  uint32_t ci = begin - o * col_tiles;

  for (uint32_t item = 0; item < count; ++item) {
    const uint32_t i0 = ci * C;
    const uint32_t cols = min(C, inner - i0);
    CAcc c_l0(1, cols);
    TASSIGN(c_l0, 0);
    const uint64_t slice = static_cast<uint64_t>(o) * mid * inner + i0;
    for (uint32_t m0 = 0; m0 < mid; m0 += R) {
      const uint32_t rows = min(R, mid - m0);
      AMat a_l1(rows, cols);
      TASSIGN(a_l1, 0);
      AGlobal a_global(input + slice + static_cast<uint64_t>(m0) * inner,
                       {1, 1, 1, rows, cols}, {1, 1, 1, inner, 1});
      TLOAD(a_l1, a_global);
      // A ragged tile reaches neither past the end of the input nor into the
      // next row of it, so the elements the mad reads beyond it must be zeroed
      // here. TFILLPAD writes L1 through create_cbuf_matrix and so issues on
      // MTE2, in order behind the TLOAD it completes and ahead of the flag
      // that releases MTE1.
      if (rows < R || cols < C) TFILLPAD(a_l1, a_l1);
      kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
      TMOV(a_l0, a_l1);
      kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
      if (m0 == 0) {
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
    CGlobal c_global(output + static_cast<uint64_t>(o) * inner + i0,
                     {1, 1, 1, 1, cols}, {1, 1, 1, cols, 1});
    TSTORE(c_global, c_l0);
    // Let the store read L0C out before the next item's mad overwrites it.
    kernel_utils::SetWaitFlag<PIPE_FIX, PIPE_M>(0);
    if (++ci == col_tiles) {
      ci = 0;
      ++o;
    }
  }
  pipe_barrier(PIPE_ALL);
}

// `inner == 1`: the reduced axis is the contiguous one, so a tile of the input
// is an R x C block of rows to be summed row by row, which is the other mad
// orientation:
//
//     A[R, C] @ ones[C, 1] = the row sums of A, an R x 1 column.
//
// n comes from the right operand's valid column count and may be 1, so this
// costs the same R x C MACs per tile as the mode above -- the ones operand is
// declared 16 wide only because L0B is addressed in fractals, and just its
// first column is multiplied.
//
// The work items are the output tiles, one per R rows of the output, and each
// stores `rows` consecutive values.
template <typename InputT, typename OutputT, uint32_t C>
AICORE void ReduceContiguousAxis(__gm__ InputT* input, __gm__ OutputT* output,
                                 uint32_t outer, uint32_t mid) {
  // R rows of the kept axis by C columns of the reduced one. Here C is a run
  // of the contiguous axis, so once it covers the whole reduced extent the
  // tile is one unbroken block of the input and R only sets how many results
  // the tile produces at once.
  constexpr uint32_t R = kTileElems / C;
  using AGlobal = GlobalTensor<InputT, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                               pto::Stride<1, 1, 1, DYNAMIC, 1>>;
  using CGlobal = GlobalTensor<OutputT, Shape<1, 1, 1, DYNAMIC, 1>,
                               pto::Stride<1, 1, 1, 1, 1>>;
  using AMat = Tile<TileType::Mat, InputT, R, C, BLayout::ColMajor, DYNAMIC,
                    DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  // The ones column is as long as the mad's k, which is the tile's width.
  using OnesMat =
      Tile<TileType::Mat, InputT, C, kFractalCols, BLayout::ColMajor, C,
           kFractalCols, SLayout::RowMajor, 512>;
  // Valid n == 1 narrows the mad to the one column of ones that matters, and
  // narrows the store to the one column of L0C that holds the row sums.
  using OnesRight = TileRight<InputT, C, kFractalCols, C, 1>;
  using CAcc = TileAcc<OutputT, R, kFractalCols, DYNAMIC, 1>;

  const uint32_t work = kernel_utils::CeilDiv(outer, R);
  if (work == 0) return;

  OnesMat ones_l1;
  OnesRight ones_l0;
  TileLeft<InputT, R, C> a_l0;
  TASSIGN(ones_l1, R * C * sizeof(InputT));
  TASSIGN(ones_l0, 0);
  TASSIGN(a_l0, 0);
  // Filled over its whole declared width so that the single column the mad
  // reads is ones however the fractal layout places it.
  TEXPANDS(ones_l1, static_cast<InputT>(1));
  kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(ones_l0, ones_l1);
  kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  const uint32_t id = get_block_idx();
  const uint32_t blocks = get_block_num();
  const uint32_t count = work / blocks + (id < work % blocks);
  const uint32_t begin = id * (work / blocks) + min(id, work % blocks);

  for (uint32_t item = 0; item < count; ++item) {
    const uint32_t o0 = (begin + item) * R;
    const uint32_t rows = min(R, outer - o0);
    CAcc c_l0(rows);
    TASSIGN(c_l0, 0);
    const uint64_t slice = static_cast<uint64_t>(o0) * mid;
    for (uint32_t m0 = 0; m0 < mid; m0 += C) {
      const uint32_t cols = min(C, mid - m0);
      AMat a_l1(rows, cols);
      TASSIGN(a_l1, 0);
      AGlobal a_global(input + slice + m0, {1, 1, 1, rows, cols},
                       {1, 1, 1, mid, 1});
      TLOAD(a_l1, a_global);
      if (rows < R || cols < C) TFILLPAD(a_l1, a_l1);
      kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
      TMOV(a_l0, a_l1);
      kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
      if (m0 == 0) {
        TMATMUL(c_l0, a_l0, ones_l0);
      } else {
        TMATMUL_ACC(c_l0, c_l0, a_l0, ones_l0);
      }
      kernel_utils::SetWaitFlag<PIPE_M, PIPE_MTE1>(0);
      kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    }
    kernel_utils::SetWaitFlag<PIPE_M, PIPE_FIX>(0);
    CGlobal c_global(output + o0, {1, 1, 1, rows, 1}, {1, 1, 1, 1, 1});
    TSTORE(c_global, c_l0);
    kernel_utils::SetWaitFlag<PIPE_FIX, PIPE_M>(0);
  }
  pipe_barrier(PIPE_ALL);
}

// `inner` is small and larger than one: the reduced axis is strided, but its
// stride is short enough that a whole mid x inner slab is a brief run of
// memory, so R consecutive slabs are one contiguous block. Read as R rows of
// the (outer, mid * inner) matrix the input already is, the reduction becomes
// a plain matmul:
//
//     Y[R, mid * inner] @ P[mid * inner, inner] = R rows of `inner` sums,
//
// where P[j, i] is 1 when j % inner == i -- identity matrices stacked
// vertically, folding every column of Y onto the output slot its position
// implies. One tile therefore reduces R whole slabs at once where the strided
// mode reduces part of one, and reads them as a single contiguous run instead
// of R short strided ones. That is what makes a small kept extent cheap: the
// strided mode pays one work item per output row and pads the mad out to a
// fractal in both directions, so reducing dim 3 of (256, 32, 16, 8, 4) costs
// it 131072 tiles and 2.1G MACs against this mode's 256 and 16.8M.
//
// P cannot be built on chip -- TEXPANDS fills a constant and TTRI a triangle,
// and no instruction fills a pattern -- so the host passes it. It depends only
// on `inner` and the chunk width, never on the input, and is read once here.
template <typename InputT, typename OutputT, uint32_t C>
AICORE void ReduceFoldedAxis(__gm__ InputT* input, __gm__ InputT* pattern,
                             __gm__ OutputT* output, uint32_t outer,
                             uint32_t mid, uint32_t inner) {
  constexpr uint32_t R = kTileElems / C;
  // Columns per chunk, cut to a whole number of slabs' worth of residues so
  // that every chunk starts on a multiple of `inner` and P is the same for all
  // of them. The columns between this and C stay zero in both operands.
  const uint32_t chunk = (C / inner) * inner;
  // One row of Y, which is also the pitch the tile load carries.
  const uint32_t width = mid * inner;
  // The mad runs at the pattern's full width; the columns past `inner` are
  // zero there, so they produce zero sums and the store simply stops at
  // `inner`. Paying that is what keeps the pattern a full, statically shaped
  // tile, and the mad is not this mode's cost.

  using AGlobal = GlobalTensor<InputT, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                               pto::Stride<1, 1, 1, DYNAMIC, 1>>;
  using CGlobal = GlobalTensor<OutputT, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                               pto::Stride<1, 1, 1, DYNAMIC, 1>>;
  using AMat = Tile<TileType::Mat, InputT, R, C, BLayout::ColMajor, DYNAMIC,
                    DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  // The pattern arrives already padded to the whole tile -- its rows past the
  // chunk and its columns past `inner` are zero -- so this is a full-tile load
  // like the all-ones operands above. A partly valid ND2NZ load into a tile
  // taller than one fractal places its rows wrong, which cost exactly the
  // second half of every reduction wider than 16.
  using PMat = Tile<TileType::Mat, InputT, C, kFractalCols, BLayout::ColMajor,
                    C, kFractalCols, SLayout::RowMajor, 512>;
  using PGlobal = GlobalTensor<InputT, Shape<1, 1, 1, C, kFractalCols>,
                               pto::Stride<1, 1, 1, kFractalCols, 1>>;
  using CAcc = TileAcc<OutputT, R, kFractalCols, DYNAMIC, DYNAMIC>;

  const uint32_t work = kernel_utils::CeilDiv(outer, R);
  if (work == 0) return;

  PMat p_l1;
  TileRight<InputT, C, kFractalCols> p_l0;
  TileLeft<InputT, R, C> a_l0;
  TASSIGN(p_l1, R * C * sizeof(InputT));
  TASSIGN(p_l0, 0);
  TASSIGN(a_l0, 0);
  PGlobal p_global(pattern);
  TLOAD(p_l1, p_global);
  kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TMOV(p_l0, p_l1);
  kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);

  const uint32_t id = get_block_idx();
  const uint32_t blocks = get_block_num();
  const uint32_t count = work / blocks + (id < work % blocks);
  const uint32_t begin = id * (work / blocks) + min(id, work % blocks);

  for (uint32_t item = 0; item < count; ++item) {
    const uint32_t o0 = (begin + item) * R;
    const uint32_t rows = min(R, outer - o0);
    CAcc c_l0(rows, inner);
    TASSIGN(c_l0, 0);
    const uint64_t slab = static_cast<uint64_t>(o0) * width;
    for (uint32_t j0 = 0; j0 < width; j0 += chunk) {
      const uint32_t cols = min(chunk, width - j0);
      AMat a_l1(rows, cols);
      TASSIGN(a_l1, 0);
      AGlobal a_global(input + slab + j0, {1, 1, 1, rows, cols},
                       {1, 1, 1, width, 1});
      TLOAD(a_l1, a_global);
      if (rows < R || cols < C) TFILLPAD(a_l1, a_l1);
      kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
      TMOV(a_l0, a_l1);
      kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(0);
      if (j0 == 0) {
        TMATMUL(c_l0, a_l0, p_l0);
      } else {
        TMATMUL_ACC(c_l0, c_l0, a_l0, p_l0);
      }
      kernel_utils::SetWaitFlag<PIPE_M, PIPE_MTE1>(0);
      kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
    }
    kernel_utils::SetWaitFlag<PIPE_M, PIPE_FIX>(0);
    CGlobal c_global(output + static_cast<uint64_t>(o0) * inner,
                     {1, 1, 1, rows, inner}, {1, 1, 1, inner, 1});
    TSTORE(c_global, c_l0);
    kernel_utils::SetWaitFlag<PIPE_FIX, PIPE_M>(0);
  }
  pipe_barrier(PIPE_ALL);
}

// The tile width is one of 32 through 512, chosen by the host: the mad runs at
// a fixed R x C whatever the tile holds, so each direction pays for the rows or
// columns the padding adds, and the host weighs that against how many tiles the
// shape then splits into.
template <typename InputT, typename OutputT, uint32_t C>
AICORE void ReduceDim(__gm__ InputT* input, __gm__ InputT* pattern,
                      __gm__ OutputT* output, uint32_t outer, uint32_t mid,
                      uint32_t inner, uint32_t mode) {
  if (mode == kModeContiguous) {
    ReduceContiguousAxis<InputT, OutputT, C>(input, output, outer, mid);
  } else if (mode == kModeFolded) {
    ReduceFoldedAxis<InputT, OutputT, C>(input, pattern, output, outer, mid,
                                         inner);
  } else {
    ReduceStridedAxis<InputT, OutputT, C>(input, output, outer, mid, inner);
  }
}
}  // namespace

template <typename InputT>
AICORE void ReduceDimDispatch(__gm__ InputT* input, __gm__ InputT* pattern,
                              __gm__ ReduceOutT<InputT>* output, uint32_t outer,
                              uint32_t mid, uint32_t inner, uint32_t tile_cols,
                              uint32_t mode) {
  using OutputT = ReduceOutT<InputT>;
  switch (tile_cols) {
    case 32:
      ReduceDim<InputT, OutputT, 32>(input, pattern, output, outer, mid, inner,
                                     mode);
      break;
    case 64:
      ReduceDim<InputT, OutputT, 64>(input, pattern, output, outer, mid, inner,
                                     mode);
      break;
    case 128:
      ReduceDim<InputT, OutputT, 128>(input, pattern, output, outer, mid, inner,
                                      mode);
      break;
    case 256:
      ReduceDim<InputT, OutputT, 256>(input, pattern, output, outer, mid, inner,
                                      mode);
      break;
    case 512:
      ReduceDim<InputT, OutputT, 512>(input, pattern, output, outer, mid, inner,
                                      mode);
      break;
  }
}

extern "C" __global__ AICORE void cube_reduce_dim_fp16(
    __gm__ half* input, __gm__ half* pattern, __gm__ float* output,
    uint32_t outer, uint32_t mid, uint32_t inner, uint32_t tile_cols,
    uint32_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  ReduceDimDispatch<half>(input, pattern, output, outer, mid, inner, tile_cols,
                          mode);
#endif
}

extern "C" void pto_launch_cube_reduce_dim_fp16(uint32_t blockDim, void* stream,
                                                void* input, void* pattern,
                                                void* output, uint32_t outer,
                                                uint32_t mid, uint32_t inner,
                                                uint32_t tile_cols,
                                                uint32_t mode) {
  cube_reduce_dim_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ half*)pattern, (__gm__ float*)output, outer,
      mid, inner, tile_cols, mode);
}

extern "C" __global__ AICORE void cube_reduce_dim_fp32(
    __gm__ float* input, __gm__ float* pattern, __gm__ float* output,
    uint32_t outer, uint32_t mid, uint32_t inner, uint32_t tile_cols,
    uint32_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  ReduceDimDispatch<float>(input, pattern, output, outer, mid, inner, tile_cols,
                           mode);
#endif
}

extern "C" void pto_launch_cube_reduce_dim_fp32(uint32_t blockDim, void* stream,
                                                void* input, void* pattern,
                                                void* output, uint32_t outer,
                                                uint32_t mid, uint32_t inner,
                                                uint32_t tile_cols,
                                                uint32_t mode) {
  cube_reduce_dim_fp32<<<blockDim, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)pattern, (__gm__ float*)output,
      outer, mid, inner, tile_cols, mode);
}

extern "C" __global__ AICORE void cube_reduce_dim_int8(
    __gm__ int8_t* input, __gm__ int8_t* pattern, __gm__ int32_t* output,
    uint32_t outer, uint32_t mid, uint32_t inner, uint32_t tile_cols,
    uint32_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  ReduceDimDispatch<int8_t>(input, pattern, output, outer, mid, inner,
                            tile_cols, mode);
#endif
}

extern "C" void pto_launch_cube_reduce_dim_int8(uint32_t blockDim, void* stream,
                                                void* input, void* pattern,
                                                void* output, uint32_t outer,
                                                uint32_t mid, uint32_t inner,
                                                uint32_t tile_cols,
                                                uint32_t mode) {
  cube_reduce_dim_int8<<<blockDim, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int8_t*)pattern, (__gm__ int32_t*)output,
      outer, mid, inner, tile_cols, mode);
}
