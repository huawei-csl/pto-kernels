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
//
// `tail_rows` are the whole S-element rows of the ragged tail. They ride here
// as one extra work item rather than going to the Vector stage: a mad over a
// zero-padded tile costs this stage one more iteration on one block, while the
// same elements cost the single Vector core a serialized load-and-reduce pass.
// The Vector stage keeps only what is left, which is less than one row.
//
// Measured on one 910B2 with the tile count pinned at 64 and the tail length
// the only variable, 6 repeats of 100 iterations: the tail costs 39 ns per row
// on the Vector stage (R^2 = 0.998 across 16 tail lengths) and 0.8 ns per row
// here, so a full 127-row tail goes from 5.03 us to 0.09 us. A length one
// element short of a whole tile was ~5 us slower than its tile-aligned
// neighbour at every size and dtype; it is now 0.2-0.45 us slower.
template <typename InputT, typename OutputT, uint32_t S>
AICORE void CubeReduce(__gm__ InputT* input, __gm__ OutputT* partials,
                       uint32_t num_tiles, uint32_t tail_rows) {
  using AGlobal = GlobalTensor<InputT, TileShape2D<InputT, S, S, Layout::ND>,
                               BaseShape2D<InputT, S, S, Layout::ND>>;
  using CGlobal = GlobalTensor<OutputT, TileShape2D<OutputT, 1, S, Layout::ND>,
                               BaseShape2D<OutputT, 1, S, Layout::ND>>;
  using AMat = Tile<TileType::Mat, InputT, S, S, BLayout::ColMajor, S, S,
                    SLayout::RowMajor, 512>;
  // The same L1 tile addressed with a runtime row count, so fewer than S rows
  // can be loaded and the remainder zero-filled in place. TFILLPAD takes a Mat
  // tile only in NZ layout, which is the ColMajor/RowMajor pair above.
  using ATail = Tile<TileType::Mat, InputT, S, S, BLayout::ColMajor, DYNAMIC,
                     DYNAMIC, SLayout::RowMajor, 512, PadValue::Zero>;
  using TailGlobal = GlobalTensor<InputT, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                                  pto::Stride<1, 1, 1, DYNAMIC, 1>>;
  using OnesMat =
      Tile<TileType::Mat, InputT, kFractalRows, S, BLayout::ColMajor,
           kFractalRows, S, SLayout::RowMajor, 512>;
  // Whole tiles plus, when the tail has at least one whole row, one more item
  // for it. Counting it as work spreads it with everything else, so it lands
  // on a block that is one tile short rather than on top of a full share.
  const uint32_t work = num_tiles + (tail_rows != 0 ? 1u : 0u);
  // With nothing but a short tail the Vector stage does all the work and never
  // reads `partials`, so this stage has no result to produce.
  if (work == 0) return;

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
  const uint32_t count = work / blocks + (id < work % blocks);
  // Balanced contiguous ranges ensure every launched block writes a result.
  const uint32_t begin = id * (work / blocks) + min(id, work % blocks);
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
    const uint32_t item = begin + i;
    if (item < num_tiles) {
      AGlobal a_global(input + static_cast<uint64_t>(item) * S * S);
      TLOAD(a_l1, a_global);
    } else {
      // The tail item. Load its whole rows and zero the rest of the tile, so
      // the full-width mad below sums exactly the tail without the load ever
      // reaching past the end of the input. TFILLPAD writes L1 through
      // create_cbuf_matrix and so issues on MTE2, in order behind the TLOAD
      // it completes and ahead of the flag that releases MTE1.
      ATail a_tail(tail_rows, S);
      TASSIGN(a_tail, 0);
      TailGlobal tail_global(input + static_cast<uint64_t>(num_tiles) * S * S,
                             {1, 1, 1, tail_rows, S}, {1, 1, 1, S, 1});
      TLOAD(a_tail, tail_global);
      TFILLPAD(a_tail, a_tail);
    }
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

// How many elements the single Vector core folds per loop iteration, in both
// the partials fold and the ragged tail.
//
// Each iteration is a fully serialized TLOAD -> TROWSUM -> TADD chain with a
// flag round trip at either end, so this stage costs iteration count, not
// element count. At 256 a worst-case tail of one element short of a full
// 128x128 tile took 64 serialized iterations on one core while 24 Cube cores
// stood idle, and even a tail-free reduction spent 12 on the `blocks * S`
// partials. At 4096 those become 4 and 1.
//
// Measured on one 910B2, four values interleaved in a single process (A,B,C,
// D,A,B,...), 3 repeats of 100 iterations: 4096 is 2.0-2.9x faster than 256
// at the ragged sizes, 1.16-1.27x at the tail-free ones, and never slower at
// any size or dtype. 8192 adds a further 3-5%, but only at the ragged sizes,
// and doubles the footprint below; 4096 is the knee.
//
// UB cost is ~64 KB for an FP32 input and ~76 KB for int8, which widens via
// half before reducing. Raising this is bounded by that budget, not by the
// loop.
constexpr uint32_t kFoldChunk = 4096;

// Sums the `tail_len` elements at `input + tail_offset` and returns the total.
//
// The Cube stage consumes whole S x S tiles plus the tail's whole S-element
// rows, so what is left here is the final partial row: fewer than S elements,
// always one chunk. A Vec tile carries a runtime valid-column count and
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
  constexpr uint32_t kChunk = kFoldChunk;
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

// Sums a row of W values into its first 8 lanes with log2(W/8) vector adds,
// each one adding the upper half onto the lower half in place. The halves are
// disjoint, so nothing has to be copied, and every add is a full-width vector
// operation rather than an 8-element one.
//
// This stands in for TROWSUM on the integer path. TRowSum's integer path would
// take W/8 adds with a pipe barrier between each, plus a PIPE_ALL drain: 16
// adds and 17 barriers at S = 128, against 4 and 4 here. The 8 lanes left over
// are folded by the scalar read at the end of CompleteCubeReduce, which is the
// same scalar work TRowSum does internally, just done once.
template <typename T, uint32_t W>
AICORE void HalveRowTo8Lanes(uint32_t base) {
  static_assert(W >= 8 && (W & (W - 1)) == 0,
                "the row width must be a power of two of at least 8");
  if constexpr (W > 8) {
    constexpr uint32_t kHalf = W / 2;
    Tile<TileType::Vec, T, 1, kHalf> lo;
    Tile<TileType::Vec, T, 1, kHalf> hi;
    TASSIGN(lo, base);
    TASSIGN(hi, base + kHalf * sizeof(T));
    TADD(lo, lo, hi);
    kernel_utils::PipeBarrierVec();
    HalveRowTo8Lanes<T, kHalf>(base);
  }
}

// The partials are `blocks` rows of S values, one row per Cube block, and they
// are laid out exactly that way in global memory. The accumulation stays in T:
// an int8 reduction can exceed what fp32 holds exactly, and the test for a
// full tensor of -128 depends on it being exact.
//
// Which way they are folded matters, because TROWSUM is two different
// algorithms. For float it is a vcadd halving tree over whole 64-element
// repeats. For int32 it takes a separate integer path that accumulates one
// 8-element block per vadd with a pipe barrier inside the loop
// (`pto/npu/a2a3/TRowSum.hpp`, unchanged in CANN 9.1.0 and 9.2.1), so a flat
// fold of the maximum 3072 partials costs 384 serialized vector operations.
// Measured on one 910B2, that fold is 4.25 us for int8 against 0.38 us for
// the same count in float, and it grows with `blocks * S` rather than with
// the input. The same split exists in TRowMax, TRowMin and TRowProd; no other
// kernel here feeds an integer tile to a row reduction, so it is handled in
// place rather than behind a shared helper.
//
// TCOLSUM has no such split: it masks to the full row width and, with the
// binary tree enabled, halves the row count each step whatever the type. So
// the int32 fold goes down the columns first, which leaves the row reduction
// only S values -- 16 of those 8-element steps instead of 384. Integer
// addition wraps and wrapping addition is associative, so regrouping the sum
// this way is exact, not merely close.
//
// Float deliberately keeps the flat fold, for two reasons, neither of which
// is correctness: its row reduction is already a tree so there is nothing to
// win, and leaving the path alone keeps its generated code identical, which
// bounds what this change can affect. Note for anyone tempted to unify them:
// the suite would not stop you. Its exact float cases are built so that every
// intermediate sum is representable, and the rest carry a 1e-2 tolerance, so
// no test here pins the summation order -- but real inputs would see
// different rounding, for no measured gain.
template <typename InputT, typename T, uint32_t S>
AICORE void CompleteCubeReduce(__gm__ InputT* input, __gm__ T* partials,
                               __gm__ T* output, uint32_t blocks,
                               uint64_t tail_offset, uint32_t tail_len) {
  using OutputGlobal =
      GlobalTensor<T, Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>>;
  constexpr bool kIsInt = std::is_same<T, int32_t>::value;
  constexpr uint32_t kChunk = kFoldChunk;
  constexpr uint32_t kChunkBytes = kChunk * sizeof(T);
  // Rows of S that fit one chunk buffer, so both folds use the same UB.
  constexpr uint32_t kChunkRows = kChunk / S;
  static_assert(kChunk % S == 0 && kChunkRows > 0,
                "the chunk buffer must hold a whole number of partial rows");

  // UB map. Both folds take the chunk buffer at 0 and their scratch at
  // kChunkBytes. Only the column fold needs the two S-wide rows above that, so
  // the float layout, and with it the float object code, is what it was before
  // this path existed.
  constexpr uint32_t kRowBytes = S * sizeof(T);
  constexpr uint32_t kColSumBase = 2 * kChunkBytes;
  constexpr uint32_t kLaneBytes = 8 * sizeof(T);  // one 8-lane scalar tile
  constexpr uint32_t kScalarBase =
      kIsInt ? kColSumBase + kRowBytes : kColSumBase;
  constexpr uint32_t kTailBase = kScalarBase + 3 * kLaneBytes;

  // The integer fold stops at 8 lanes and sums them in the scalar read below;
  // the float fold's row reduction already produces a single value.
  constexpr uint32_t kTotalLanes = kIsInt ? 8 : 1;
  Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, 1, kTotalLanes> totals;
  Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, 1, 1> result;

  set_mask_norm();
  set_vector_mask(-1, -1);

  TASSIGN(totals, kScalarBase + kLaneBytes);
  TASSIGN(result, kScalarBase + 2 * kLaneBytes);
  TEXPANDS(totals, static_cast<T>(0));
  kernel_utils::PipeBarrierVec();
  if constexpr (kIsInt) {
    using Part2DGlobal = GlobalTensor<T, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                                      pto::Stride<1, 1, 1, DYNAMIC, 1>>;
    // Only the row count varies; S stays static so TLOAD and TCOLSUM read the
    // column count as a constant rather than from the tile's runtime field.
    using Part2D =
        Tile<TileType::Vec, T, kChunkRows, S, BLayout::RowMajor, DYNAMIC, S>;
    Tile<TileType::Vec, T, kChunkRows, S> col_tmp;
    Tile<TileType::Vec, T, 1, S> col_sum;
    Tile<TileType::Vec, T, 1, 8> col_lanes;  // the first 8 lanes of col_sum
    TASSIGN(col_tmp, kChunkBytes);
    TASSIGN(col_sum, kColSumBase);
    TASSIGN(col_lanes, kColSumBase);
    for (uint32_t first = 0; first < blocks; first += kChunkRows) {
      const uint32_t row_count = min(kChunkRows, blocks - first);
      Part2DGlobal source(partials + static_cast<uint64_t>(first) * S,
                          {1, 1, 1, row_count, S}, {1, 1, 1, S, 1});
      Part2D part_in(row_count);
      TASSIGN(part_in, 0);
      TLOAD(part_in, source);
      kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
      TCOLSUM(col_sum, part_in, col_tmp, true);
      kernel_utils::PipeBarrierVec();
      HalveRowTo8Lanes<T, S>(kColSumBase);
      TADD(totals, totals, col_lanes);
      kernel_utils::PipeBarrierVec();
      kernel_utils::SetWaitFlag<PIPE_V, PIPE_MTE2>(0);
    }
  } else {
    using PartGlobal =
        GlobalTensor<T, Shape<1, 1, 1, 1, DYNAMIC>, pto::Stride<1, 1, 1, 1, 1>>;
    using PartIn =
        Tile<TileType::Vec, T, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;
    Tile<TileType::Vec, T, 1, 8, BLayout::RowMajor, 1, 1> part_sum;
    Tile<TileType::Vec, T, 1, kChunk> part_tmp;
    TASSIGN(part_sum, kScalarBase);
    TASSIGN(part_tmp, kChunkBytes);
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
  }
  // AccumulateTail ends on a V -> S flag, so the vector accumulation above is
  // visible to these scalar reads.
  const float tail =
      AccumulateTail<InputT, kTailBase>(input, tail_offset, tail_len);
  T total = totals.GetValue(0);
  if constexpr (kIsInt) {
    for (uint32_t lane = 1; lane < kTotalLanes; ++lane) {
      total += totals.GetValue(lane);
    }
  }
  result.SetValue(0, total + static_cast<T>(tail));
  kernel_utils::SetWaitFlag<PIPE_S, PIPE_MTE3>(0);
  OutputGlobal destination(output);
  TSTORE(destination, result);
  pipe_barrier(PIPE_ALL);
}
}  // namespace

template <typename InputT>
AICORE void CubeReduceDispatch(__gm__ InputT* input,
                               __gm__ CubeOutT<InputT>* partials,
                               uint32_t num_tiles, uint32_t tail_rows,
                               uint32_t size) {
  using OutputT = CubeOutT<InputT>;
  switch (size) {
    case 32:
      CubeReduce<InputT, OutputT, 32>(input, partials, num_tiles, tail_rows);
      break;
    case 64:
      CubeReduce<InputT, OutputT, 64>(input, partials, num_tiles, tail_rows);
      break;
    case 128:
      CubeReduce<InputT, OutputT, 128>(input, partials, num_tiles, tail_rows);
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
                                                   uint32_t tail_rows,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeReduceDispatch<half>(input, partials, num_tiles, tail_rows, size);
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
                                            uint32_t tail_rows, uint32_t blocks,
                                            uint64_t tail_offset,
                                            uint32_t tail_len, uint32_t size) {
  cube_reduce_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)partials, num_tiles, tail_rows, size);
  complete_cube_reduce_fp16<<<1, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)partials, (__gm__ float*)output,
      blocks, tail_offset, tail_len, size);
}

extern "C" __global__ AICORE void cube_reduce_fp32(__gm__ float* input,
                                                   __gm__ float* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t tail_rows,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeReduceDispatch<float>(input, partials, num_tiles, tail_rows, size);
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
                                            uint32_t tail_rows, uint32_t blocks,
                                            uint64_t tail_offset,
                                            uint32_t tail_len, uint32_t size) {
  cube_reduce_fp32<<<blockDim, nullptr, stream>>>((__gm__ float*)input,
                                                  (__gm__ float*)partials,
                                                  num_tiles, tail_rows, size);
  complete_cube_reduce_fp32<<<1, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)partials, (__gm__ float*)output,
      blocks, tail_offset, tail_len, size);
}

extern "C" __global__ AICORE void cube_reduce_int8(__gm__ int8_t* input,
                                                   __gm__ int32_t* partials,
                                                   uint32_t num_tiles,
                                                   uint32_t tail_rows,
                                                   uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeReduceDispatch<int8_t>(input, partials, num_tiles, tail_rows, size);
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
                                            uint32_t tail_rows, uint32_t blocks,
                                            uint64_t tail_offset,
                                            uint32_t tail_len, uint32_t size) {
  cube_reduce_int8<<<blockDim, nullptr, stream>>>((__gm__ int8_t*)input,
                                                  (__gm__ int32_t*)partials,
                                                  num_tiles, tail_rows, size);
  complete_cube_reduce_int8<<<1, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int32_t*)partials, (__gm__ int32_t*)output,
      blocks, tail_offset, tail_len, size);
}
