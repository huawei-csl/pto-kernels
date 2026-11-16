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

// Accumulator the Cube unit writes for a given input type: int8 operands
// multiply into an int32 L0C, every other supported input into an fp32 one.
template <typename InputT>
using CubeAccT = typename std::conditional<std::is_same<InputT, int8_t>::value,
                                           int32_t, float>::type;

// The L2 norm of x is the Frobenius norm of any matrix holding its elements,
// and ||A||_F^2 = trace(A^T A). So each block reads the flat input as S x S
// tiles A_0, A_1, ..., accumulates F = sum_i A_i^T @ A_i in L0C, and adds
// that into one S x S tile shared by the whole grid; the Vector stage adds up
// the diagonal of that single tile.
template <typename InputT, typename AccT, uint32_t S>
AICORE void CubeNorm(__gm__ InputT* input, __gm__ AccT* partials,
                     uint32_t num_tiles) {
  using AGlobal = GlobalTensor<InputT, TileShape2D<InputT, S, S, Layout::ND>,
                               BaseShape2D<InputT, S, S, Layout::ND>>;
  using CGlobal = GlobalTensor<AccT, TileShape2D<AccT, S, S, Layout::ND>,
                               BaseShape2D<AccT, S, S, Layout::ND>>;
  using AMat = Tile<TileType::Mat, InputT, S, S, BLayout::ColMajor, S, S,
                    SLayout::RowMajor, 512>;
  // The transposed operand costs nothing: an NZ tile of A_i and a ZN tile of
  // A_i^T are the same bytes, so reading the one L1 copy under a ZN type is
  // all it takes for the L1 -> L0A path to deliver A_i^T.
  using AMatTransposed = Tile<TileType::Mat, InputT, S, S, BLayout::RowMajor, S,
                              S, SLayout::ColMajor, 512>;
  // Without a whole tile to work on, the Vector stage reduces the input by
  // itself and never reads `partials`, so this stage has no result to produce.
  if (num_tiles == 0) return;

  // Two L1 tiles in flight: the load of the next tile is issued before the
  // current one is consumed, so a block's DMA runs under its multiplies
  // instead of in front of them. Reading every element once is the whole job
  // here, which makes the GM -> L1 copy the part worth hiding.
  AMat a_l1_ping;
  AMat a_l1_pong;
  AMatTransposed at_l1_ping;
  AMatTransposed at_l1_pong;
  TileLeft<InputT, S, S> at_l0;
  TileRight<InputT, S, S> a_l0;
  TileAcc<AccT, S, S> f_l0;
  constexpr uint32_t kL1TileBytes = S * S * sizeof(InputT);
  TASSIGN(a_l1_ping, 0);
  TASSIGN(a_l1_pong, kL1TileBytes);
  TASSIGN(at_l0, 0);
  TASSIGN(a_l0, 0);
  TASSIGN(f_l0, 0);
  // Binds the transposed views to the same L1 addresses; the tiles never
  // move, so this is done once rather than per iteration.
  TRESHAPE(at_l1_ping, a_l1_ping);
  TRESHAPE(at_l1_pong, a_l1_pong);

  // Balanced contiguous ranges ensure every launched block writes a result.
  const uint32_t id = get_block_idx();
  const uint32_t blocks = get_block_num();
  const uint32_t count = num_tiles / blocks + (id < num_tiles % blocks);
  const uint32_t begin =
      id * (num_tiles / blocks) + min(id, num_tiles % blocks);

  AGlobal first(input + static_cast<uint64_t>(begin) * S * S);
  TLOAD(a_l1_ping, first);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  for (uint32_t i = 0; i < count; ++i) {
    const bool ping = (i % 2 == 0);
    const event_t filled = ping ? EVENT_ID0 : EVENT_ID1;
    const event_t consumed = ping ? EVENT_ID2 : EVENT_ID3;
    const event_t next_filled = ping ? EVENT_ID1 : EVENT_ID0;
    const event_t next_consumed = ping ? EVENT_ID3 : EVENT_ID2;

    if (i + 1 < count) {
      // The buffer being refilled last held tile i - 1, whose move into L0
      // signalled `next_consumed` on its way out.
      if (i >= 1) wait_flag(PIPE_MTE1, PIPE_MTE2, next_consumed);
      AGlobal next(input + static_cast<uint64_t>(begin + i + 1) * S * S);
      if (ping) {
        TLOAD(a_l1_pong, next);
      } else {
        TLOAD(a_l1_ping, next);
      }
      set_flag(PIPE_MTE2, PIPE_MTE1, next_filled);
    }

    wait_flag(PIPE_MTE2, PIPE_MTE1, filled);
    if (ping) {
      TEXTRACT(at_l0, at_l1_ping, 0, 0);
      TEXTRACT(a_l0, a_l1_ping, 0, 0);
    } else {
      TEXTRACT(at_l0, at_l1_pong, 0, 0);
      TEXTRACT(a_l0, a_l1_pong, 0, 0);
    }
    // Only tile i + 2 refills this buffer, so the flag is raised only when
    // there is such a tile to wait on it.
    if (i + 2 < count) set_flag(PIPE_MTE1, PIPE_MTE2, consumed);
    kernel_utils::SetWaitFlag<PIPE_MTE1, PIPE_M>(4);
    if (i == 0) {
      TMATMUL(f_l0, at_l0, a_l0);
    } else {
      TMATMUL_ACC(f_l0, f_l0, at_l0, a_l0);
    }
    // Complete the multiply before the next TEXTRACT overwrites the single
    // pair of L0 operands.
    kernel_utils::SetWaitFlag<PIPE_M, PIPE_MTE1>(5);
  }
  kernel_utils::SetWaitFlag<PIPE_M, PIPE_FIX>(0);
  // One workspace tile for the whole grid: Fixpipe adds F into it atomically,
  // so the blocks need no slots of their own and the Vector stage is left with
  // a single tile to trace. The host hands over a zeroed tile, which is what
  // makes the first of these adds land on zeros.
  CGlobal c_global(partials);
  TSTORE<TileAcc<AccT, S, S>, CGlobal, AtomicType::AtomicAdd>(c_global, f_l0);
  pipe_barrier(PIPE_ALL);
}

// A separate launch on the same stream orders all Fixpipe stores before these
// reads. The grid has already summed itself into the one workspace tile, so a
// single AIV traces it and the work here does not depend on the block count.
template <typename InputT, typename AccT, uint32_t S>
AICORE void CompleteCubeNorm(__gm__ InputT* input, __gm__ AccT* partials,
                             __gm__ float* output, uint64_t tail_offset,
                             uint32_t tail_len) {
  // int8 reaches the Vector unit in two places that fp16 and fp32 do not: the
  // Cube stage's accumulator is int32 rather than fp32, and the tail has no
  // direct conversion to fp32 and goes via half (exact over int8's range).
  constexpr bool kIsF32 = std::is_same<InputT, float>::value;
  constexpr bool kIsI8 = std::is_same<InputT, int8_t>::value;
  using PartialGlobal = GlobalTensor<AccT, TileShape2D<AccT, S, S, Layout::ND>,
                                     BaseShape2D<AccT, S, S, Layout::ND>>;
  using OutputGlobal =
      GlobalTensor<float, Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>>;
  using SquareTile = Tile<TileType::Vec, AccT, S, S>;
  using AccRowTile = Tile<TileType::Vec, AccT, 1, S>;
  using RowTile = Tile<TileType::Vec, float, 1, S>;
  using IndexTile = Tile<TileType::Vec, int32_t, 1, S>;
  using ScalarTile = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;

  // AccT is four bytes wide either way, so one layout serves both.
  constexpr uint32_t kTileBytes = S * S * sizeof(AccT);
  constexpr uint32_t kRowBytes = S * sizeof(float);
  constexpr uint32_t kScalarBytes = 8 * sizeof(float);
  constexpr uint32_t kRowsBase = kTileBytes;
  SquareTile partial;
  AccRowTile diagonal;
  RowTile diagonal_f32;
  RowTile row_tmp;
  IndexTile indices;
  IndexTile gather_tmp;
  ScalarTile total;
  ScalarTile result;
  TASSIGN(partial, 0);
  TASSIGN(diagonal, kRowsBase);
  TASSIGN(row_tmp, kRowsBase + kRowBytes);
  TASSIGN(indices, kRowsBase + 2 * kRowBytes);
  TASSIGN(gather_tmp, kRowsBase + 3 * kRowBytes);
  TASSIGN(diagonal_f32, kRowsBase + 4 * kRowBytes);
  TASSIGN(total, kRowsBase + 5 * kRowBytes);
  TASSIGN(result, kRowsBase + 5 * kRowBytes + kScalarBytes);

  // The Cube stage consumes only whole S x S tiles, so the last `tail_len`
  // elements (fewer than one tile) are squared and summed here instead of
  // being padded up to a tile anywhere. A Vec tile carries a runtime
  // valid-column count and TROWSUM honours it, so a short final chunk is
  // loaded and reduced as it is.
  //
  // The tail is widened to fp32 before squaring: that keeps its arithmetic as
  // wide as the L0C accumulation the Cube stage does, and keeps squares of
  // large fp16 inputs from overflowing.
  constexpr uint32_t kChunk = 256;
  constexpr uint32_t kTailBase = kRowsBase + 5 * kRowBytes + 2 * kScalarBytes;
  constexpr uint32_t kInBytes = kChunk * sizeof(InputT);
  // int8 is widened in two steps, so its intermediate half tile needs room.
  constexpr uint32_t kMidBytes = kIsI8 ? kChunk * sizeof(half) : 0;
  constexpr uint32_t kF32Offset = kIsF32 ? 0 : kInBytes + kMidBytes;
  constexpr uint32_t kF32Bytes = kChunk * sizeof(float);
  using TailGlobal = GlobalTensor<InputT, Shape<1, 1, 1, 1, DYNAMIC>,
                                  pto::Stride<1, 1, 1, 1, 1>>;
  using TailIn =
      Tile<TileType::Vec, InputT, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;
  using TailMid =
      Tile<TileType::Vec, half, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;
  using TailF32 =
      Tile<TileType::Vec, float, 1, kChunk, BLayout::RowMajor, 1, DYNAMIC>;
  Tile<TileType::Vec, float, 1, kChunk> tail_squares;
  Tile<TileType::Vec, float, 1, kChunk> tail_tmp;
  ScalarTile tail_sum;
  ScalarTile tail_total;
  TASSIGN(tail_squares, kTailBase + kF32Offset + kF32Bytes);
  TASSIGN(tail_tmp, kTailBase + kF32Offset + 2 * kF32Bytes);
  TASSIGN(tail_sum, kTailBase + kF32Offset + 3 * kF32Bytes);
  TASSIGN(tail_total, kTailBase + kF32Offset + 3 * kF32Bytes + kScalarBytes);

  // Dynamic-extent tiles are addressed per chunk, so put the vector unit in
  // the all-lanes mask mode they assume.
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Element i of a row-major S x S tile's diagonal sits at i * S + i, one row
  // and one column on from element i - 1, so gathering at that stride reads
  // the diagonal. TCI lays down the progression {0, 1, ..., S - 1} and one
  // scale turns it into {i * S + i}.
  TCI<IndexTile, int32_t, /*descending=*/0>(indices, 0);
  kernel_utils::SetWaitFlag<PIPE_S, PIPE_V>(0);
  TMULS(indices, indices, static_cast<int32_t>(S + 1));
  kernel_utils::PipeBarrierVec();

  // Every block's F has already been added into this one tile, so a single
  // gather of its diagonal and one row reduction give the squared norm of
  // everything the Cube stage saw. A grid that was given no whole tile stores
  // nothing, and the zeroed workspace then contributes zero here.
  PartialGlobal source(partials);
  TLOAD(partial, source);
  kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
  TGATHER(diagonal, partial, indices, gather_tmp);
  kernel_utils::PipeBarrierVec();
  if constexpr (kIsI8) {
    // The diagonal of an int8 F is a sum of squares, so it is exact in int32
    // as long as the host keeps the input short enough not to wrap it; the
    // sum across the diagonal is taken in fp32, where it cannot.
    TCVT(diagonal_f32, diagonal, RoundMode::CAST_NONE);
    kernel_utils::PipeBarrierVec();
    TROWSUM(total, diagonal_f32, row_tmp);
  } else {
    TROWSUM(total, diagonal, row_tmp);
  }
  kernel_utils::PipeBarrierVec();

  TEXPANDS(tail_total, 0.0f);
  kernel_utils::PipeBarrierVec();
  for (uint32_t offset = 0; offset < tail_len; offset += kChunk) {
    const uint32_t n = min(kChunk, tail_len - offset);
    TailGlobal tail_global(input + tail_offset + offset,
                           {static_cast<int32_t>(n)});
    TailIn tail_in(n);
    TASSIGN(tail_in, kTailBase);
    TLOAD(tail_in, tail_global);
    kernel_utils::SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    TailF32 tail_f32(n);
    TASSIGN(tail_f32, kTailBase + kF32Offset);
    if constexpr (kIsI8) {
      TailMid tail_mid(n);
      TASSIGN(tail_mid, kTailBase + kInBytes);
      TCVT(tail_mid, tail_in, RoundMode::CAST_NONE);
      kernel_utils::PipeBarrierVec();
      TCVT(tail_f32, tail_mid, RoundMode::CAST_NONE);
      kernel_utils::PipeBarrierVec();
    } else if constexpr (!kIsF32) {
      TCVT(tail_f32, tail_in, RoundMode::CAST_NONE);
      kernel_utils::PipeBarrierVec();
    }
    TailF32 squares(n);
    TASSIGN(squares, kTailBase + kF32Offset + kF32Bytes);
    TMUL(squares, tail_f32, tail_f32);
    kernel_utils::PipeBarrierVec();
    TROWSUM(tail_sum, squares, tail_tmp);
    kernel_utils::PipeBarrierVec();
    TADD(tail_total, tail_total, tail_sum);
    kernel_utils::PipeBarrierVec();
    kernel_utils::SetWaitFlag<PIPE_V, PIPE_MTE2>(0);
  }

  TADD(total, total, tail_total);
  kernel_utils::PipeBarrierVec();
  TSQRT(result, total);
  kernel_utils::SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
  OutputGlobal destination(output);
  TSTORE(destination, result);
  pipe_barrier(PIPE_ALL);
}

// An int8 L1 tile is laid out in 16 x 32 boxes, so a 16 x 16 tile of it does
// not exist and S = 16 must not even be instantiated for that input type.
// The host never asks for it: an int8 input too short for a 32 x 32 tile is
// left entirely to the Vector stage's tail path.
template <typename InputT>
constexpr bool kSupportsSize16 = !std::is_same<InputT, int8_t>::value;

template <typename InputT>
AICORE void CubeNormDispatch(__gm__ InputT* input,
                             __gm__ CubeAccT<InputT>* partials,
                             uint32_t num_tiles, uint32_t size) {
  using AccT = CubeAccT<InputT>;
  switch (size) {
    case 16:
      if constexpr (kSupportsSize16<InputT>) {
        CubeNorm<InputT, AccT, 16>(input, partials, num_tiles);
      }
      break;
    case 32:
      CubeNorm<InputT, AccT, 32>(input, partials, num_tiles);
      break;
    case 64:
      CubeNorm<InputT, AccT, 64>(input, partials, num_tiles);
      break;
    case 128:
      CubeNorm<InputT, AccT, 128>(input, partials, num_tiles);
      break;
  }
}

template <typename InputT>
AICORE void CompleteCubeNormDispatch(__gm__ InputT* input,
                                     __gm__ CubeAccT<InputT>* partials,
                                     __gm__ float* output, uint64_t tail_offset,
                                     uint32_t tail_len, uint32_t size) {
  using AccT = CubeAccT<InputT>;
  switch (size) {
    case 16:
      if constexpr (kSupportsSize16<InputT>) {
        CompleteCubeNorm<InputT, AccT, 16>(input, partials, output, tail_offset,
                                           tail_len);
      }
      break;
    case 32:
      CompleteCubeNorm<InputT, AccT, 32>(input, partials, output, tail_offset,
                                         tail_len);
      break;
    case 64:
      CompleteCubeNorm<InputT, AccT, 64>(input, partials, output, tail_offset,
                                         tail_len);
      break;
    case 128:
      CompleteCubeNorm<InputT, AccT, 128>(input, partials, output, tail_offset,
                                          tail_len);
      break;
  }
}
}  // namespace

extern "C" __global__ AICORE void cube_norm_fp16(__gm__ half* input,
                                                 __gm__ float* partials,
                                                 uint32_t num_tiles,
                                                 uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeNormDispatch<half>(input, partials, num_tiles, size);
#endif
}

extern "C" __global__ AICORE void complete_cube_norm_fp16(
    __gm__ half* input, __gm__ float* partials, __gm__ float* output,
    uint64_t tail_offset, uint32_t tail_len, uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeNormDispatch<half>(input, partials, output, tail_offset, tail_len,
                                 size);
#endif
}

extern "C" void pto_launch_cube_norm_fp16(uint32_t blockDim, void* stream,
                                          void* input, void* partials,
                                          void* output, uint32_t num_tiles,
                                          uint64_t tail_offset,
                                          uint32_t tail_len, uint32_t size) {
  cube_norm_fp16<<<blockDim, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)partials, num_tiles, size);
  complete_cube_norm_fp16<<<1, nullptr, stream>>>(
      (__gm__ half*)input, (__gm__ float*)partials, (__gm__ float*)output,
      tail_offset, tail_len, size);
}

extern "C" __global__ AICORE void cube_norm_fp32(__gm__ float* input,
                                                 __gm__ float* partials,
                                                 uint32_t num_tiles,
                                                 uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeNormDispatch<float>(input, partials, num_tiles, size);
#endif
}

extern "C" __global__ AICORE void complete_cube_norm_fp32(
    __gm__ float* input, __gm__ float* partials, __gm__ float* output,
    uint64_t tail_offset, uint32_t tail_len, uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeNormDispatch<float>(input, partials, output, tail_offset,
                                  tail_len, size);
#endif
}

extern "C" void pto_launch_cube_norm_fp32(uint32_t blockDim, void* stream,
                                          void* input, void* partials,
                                          void* output, uint32_t num_tiles,
                                          uint64_t tail_offset,
                                          uint32_t tail_len, uint32_t size) {
  cube_norm_fp32<<<blockDim, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)partials, num_tiles, size);
  complete_cube_norm_fp32<<<1, nullptr, stream>>>(
      (__gm__ float*)input, (__gm__ float*)partials, (__gm__ float*)output,
      tail_offset, tail_len, size);
}

// int8 operands multiply into an int32 L0C, so this path carries an int32
// workspace all the way to the Vector stage, which widens the diagonal to
// fp32 before summing it. The result is still an fp32 norm.
extern "C" __global__ AICORE void cube_norm_int8(__gm__ int8_t* input,
                                                 __gm__ int32_t* partials,
                                                 uint32_t num_tiles,
                                                 uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#if defined(__DAV_CUBE__)
  CubeNormDispatch<int8_t>(input, partials, num_tiles, size);
#endif
}

extern "C" __global__ AICORE void complete_cube_norm_int8(
    __gm__ int8_t* input, __gm__ int32_t* partials, __gm__ float* output,
    uint64_t tail_offset, uint32_t tail_len, uint32_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if defined(__DAV_VEC__)
  CompleteCubeNormDispatch<int8_t>(input, partials, output, tail_offset,
                                   tail_len, size);
#endif
}

extern "C" void pto_launch_cube_norm_int8(uint32_t blockDim, void* stream,
                                          void* input, void* partials,
                                          void* output, uint32_t num_tiles,
                                          uint64_t tail_offset,
                                          uint32_t tail_len, uint32_t size) {
  cube_norm_int8<<<blockDim, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int32_t*)partials, num_tiles, size);
  complete_cube_norm_int8<<<1, nullptr, stream>>>(
      (__gm__ int8_t*)input, (__gm__ int32_t*)partials, (__gm__ float*)output,
      tail_offset, tail_len, size);
}
