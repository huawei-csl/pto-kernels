/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

/**
 * Doubly-stochastic Sinkhorn normalization kernel (fp16 I/O, fp32 internal).
 *
 * Implements the DeepSeek MHC pre-processing sinkhorn:
 *   1. softmax per row + eps
 *   2. column-normalize (+ eps)
 *   3. repeat (repeat-1) times: row-normalize (+ eps), column-normalize (+ eps)
 *
 * Design:
 *   - One vector core per (K, K) matrix.
 *   - The entire matrix lives in UB as fp32 via a 2D tile with static dims
 *     MAX_DIM × MAX_DIM but dynamic dims (K, K). All reductions (TROWSUM,
 *     TROWMAX, TCOLSUM) respect the dynamic K, ignoring padding.
 *   - K must be <= MAX_DIM (128).
 */

#include <pto/pto-inst.hpp>

// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format on

using namespace pto;

constexpr uint32_t UB_USABLE_BYTES = 192 * 1024;
constexpr uint32_t MAX_DIM = 128;

namespace UbOfs {
constexpr unsigned MAT_HALF = 0;
constexpr unsigned MAT_FP32 = MAT_HALF + MAX_DIM * MAX_DIM * sizeof(half);
constexpr unsigned TMP      = MAT_FP32 + MAX_DIM * MAX_DIM * sizeof(float);
constexpr unsigned VEC_BUF  = TMP + MAX_DIM * MAX_DIM * sizeof(float);
constexpr unsigned TOTAL    = VEC_BUF + MAX_DIM * sizeof(float);
}  // namespace UbOfs

static_assert(UbOfs::TOTAL <= UB_USABLE_BYTES, "Sinkhorn DS UB exceeds 192 KB.");

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

using StrideDim5 = pto::Stride<1, 1, 1, 1, 1>;

template <typename T, uint32_t N>
using Vec1D = Tile<TileType::Vec, T, 1, N, BLayout::RowMajor, -1, -1>;

template <typename T, uint32_t N>
using Global1D = GlobalTensor<T, Shape<1, 1, 1, 1, N>, StrideDim5>;

template <typename T, uint32_t R, uint32_t C>
using Tile2D = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

using DynStride = Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T>
using Shape2D = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
template <typename T, uint32_t C>
using Global2D = GlobalTensor<T, Shape2D<T>, DynStride, Layout::ND>;

template <typename T, uint32_t R>
using ColVec = Tile<TileType::Vec, T, R, 1, BLayout::ColMajor, DYNAMIC, DYNAMIC>;

AICORE inline void initPipeFlags() {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

AICORE inline void drainPipeFlags() {
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

// Column-normalize: divide mat[r,:] by vec[:] for each row.
// (Replaces unavailable TCOLEXPANDDIV.)
AICORE void colNormDiv(uint32_t K) {
  constexpr unsigned rowBytes = MAX_DIM * sizeof(float);
  for (uint32_t r = 0; r < K; ++r) {
    Vec1D<float, MAX_DIM> row(1, K);
    Vec1D<float, MAX_DIM> vec(1, K);
    TASSIGN(row, UbOfs::MAT_FP32 + r * rowBytes);
    TASSIGN(vec, UbOfs::VEC_BUF);
    TDIV(row, row, vec);
    pipe_barrier(PIPE_V);
  }
}

template <typename T>
AICORE void runSinkhornDS(__gm__ T *input, __gm__ T *output,
                          uint32_t N, uint32_t K,
                          uint32_t repeat, float eps) {
  set_mask_norm();
  set_vector_mask(-1, -1);
  if (K == 0 || K > MAX_DIM) return;

  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t wid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t KK = K * K;
  // Flat count covering the 2D buffer (row stride = MAX_DIM).
  const uint32_t flat = K * MAX_DIM;

  initPipeFlags();

  for (uint32_t bi = wid; bi < N; bi += num_workers) {
    __gm__ T *gm_in  = input  + static_cast<size_t>(bi) * KK;
    __gm__ T *gm_out = output + static_cast<size_t>(bi) * KK;

    // ---- Zero fp16 buffer, then load (K, K) ----
    {
      Vec1D<T, MAX_DIM * MAX_DIM> zHalf(1, flat);
      TASSIGN(zHalf, UbOfs::MAT_HALF);
      TEXPANDS(zHalf, (T)0);
      pipe_barrier(PIPE_V);
    }

    Tile2D<T, MAX_DIM, MAX_DIM> matHalf(K, K);
    TASSIGN(matHalf, UbOfs::MAT_HALF);
    Shape2D<T> inShape(K, K);
    DynStride inStride(K);
    Global2D<T, MAX_DIM> gIn(gm_in, inShape, inStride);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(matHalf, gIn);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // fp16 → fp32
    Vec1D<T, MAX_DIM * MAX_DIM>     hFlat(1, flat);
    Vec1D<float, MAX_DIM * MAX_DIM> fFlat(1, flat);
    TASSIGN(hFlat, UbOfs::MAT_HALF);
    TASSIGN(fFlat, UbOfs::MAT_FP32);
    TCVT(fFlat, hFlat, RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);

    // 2D view with dynamic (K, K) — reductions respect this.
    Tile2D<float, MAX_DIM, MAX_DIM> mat(K, K);
    TASSIGN(mat, UbOfs::MAT_FP32);
    Tile2D<float, MAX_DIM, MAX_DIM> tmp(K, K);
    TASSIGN(tmp, UbOfs::TMP);

    // ============================================================
    // Softmax per row: max-subtract, exp, sum, divide
    // ============================================================
    ColVec<float, MAX_DIM> vecCol(K, 1);
    TASSIGN(vecCol, UbOfs::VEC_BUF);

    // Row max → ColVec(K, 1)
    TROWMAX(vecCol, mat, tmp);
    pipe_barrier(PIPE_V);

    // Subtract max per row
    TROWEXPANDSUB(mat, mat, vecCol);
    pipe_barrier(PIPE_V);

    // Exp (flat — includes padding, but padding was 0, exp(0-max)=exp(-max)≈0)
    TEXP(fFlat, fFlat);
    pipe_barrier(PIPE_V);

    // Row sum → ColVec(K, 1)
    TROWSUM(vecCol, mat, tmp);
    pipe_barrier(PIPE_V);

    // Add eps to row sums (as 1D view of VEC_BUF)
    {
      Vec1D<float, MAX_DIM> vecFlat(1, K);
      TASSIGN(vecFlat, UbOfs::VEC_BUF);
      TADDS(vecFlat, vecFlat, eps);
      pipe_barrier(PIPE_V);
    }

    // Divide by (row_sum + eps)
    TROWEXPANDDIV(mat, mat, vecCol);
    pipe_barrier(PIPE_V);

    // Add eps to all elements
    TADDS(fFlat, fFlat, eps);
    pipe_barrier(PIPE_V);

    // ============================================================
    // Column normalize
    // ============================================================
    {
      Vec1D<float, MAX_DIM> colSums(1, K);
      TASSIGN(colSums, UbOfs::VEC_BUF);
      TCOLSUM(colSums, mat, tmp, false);
      pipe_barrier(PIPE_V);
      TADDS(colSums, colSums, eps);
      pipe_barrier(PIPE_V);
    }
    colNormDiv(K);

    // ============================================================
    // Iterate (repeat-1) times: row-norm + col-norm
    // ============================================================
    for (uint32_t it = 1; it < repeat; ++it) {
      // Row normalize
      TASSIGN(vecCol, UbOfs::VEC_BUF);
      TROWSUM(vecCol, mat, tmp);
      pipe_barrier(PIPE_V);
      {
        Vec1D<float, MAX_DIM> vecFlat(1, K);
        TASSIGN(vecFlat, UbOfs::VEC_BUF);
        TADDS(vecFlat, vecFlat, eps);
        pipe_barrier(PIPE_V);
      }
      TROWEXPANDDIV(mat, mat, vecCol);
      pipe_barrier(PIPE_V);

      // Column normalize
      {
        Vec1D<float, MAX_DIM> colSums(1, K);
        TASSIGN(colSums, UbOfs::VEC_BUF);
        TCOLSUM(colSums, mat, tmp, false);
        pipe_barrier(PIPE_V);
        TADDS(colSums, colSums, eps);
        pipe_barrier(PIPE_V);
      }
      colNormDiv(K);
    }

    // ============================================================
    // Store: fp32 → fp16 → HBM
    // ============================================================
    TCVT(hFlat, fFlat, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);

    Tile2D<T, MAX_DIM, MAX_DIM> outHalf(K, K);
    TASSIGN(outHalf, UbOfs::MAT_HALF);
    Shape2D<T> outShape(K, K);
    DynStride outStride(K);
    Global2D<T, MAX_DIM> gOut(gm_out, outShape, outStride);

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gOut, outHalf);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }

  drainPipeFlags();
}

#endif

extern "C" __global__ AICORE void sinkhorn_ds_fp16(GM_ADDR input,
                                                    GM_ADDR output,
                                                    uint32_t N, uint32_t K,
                                                    uint32_t repeat,
                                                    float eps) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  runSinkhornDS<half>((__gm__ half *)input, (__gm__ half *)output,
                      N, K, repeat, eps);
#else
  (void)input; (void)output; (void)N; (void)K; (void)repeat; (void)eps;
#endif
}

extern "C" void call_sinkhorn_ds_kernel(uint32_t blockDim, void *stream,
                                        uint8_t *input, uint8_t *output,
                                        uint32_t N, uint32_t K,
                                        uint32_t repeat, float eps) {
  sinkhorn_ds_fp16<<<blockDim * 2, nullptr, stream>>>(
      input, output, N, K, repeat, eps);
}
