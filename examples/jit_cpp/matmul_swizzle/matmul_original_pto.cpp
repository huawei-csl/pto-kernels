/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

// Modified from
// https://gitcode.com/cann/pto-isa/blob/master/demos/torch_jit/gemm/gemm_kernel.cpp
// to make it work for the benchmark. Original license above.

#include <cstdint>
#include <pto/pto-inst.hpp>
using namespace pto;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t L0_PINGPONG_BYTES =
    32 * 1024;  // L0A/L0B ping-pong split (32 KiB per buffer)

#ifndef ORIG_PTO_M
#define ORIG_PTO_M 6144
#endif
#ifndef ORIG_PTO_N
#define ORIG_PTO_N 6144
#endif
#ifndef ORIG_PTO_K
#define ORIG_PTO_K 6144
#endif
#ifndef ORIG_PTO_BASE_M
#define ORIG_PTO_BASE_M 128
#endif
#ifndef ORIG_PTO_BASE_N
#define ORIG_PTO_BASE_N 256
#endif
#ifndef ORIG_PTO_BASE_K
#define ORIG_PTO_BASE_K 64
#endif
#ifndef ORIG_PTO_M_ITER
#define ORIG_PTO_M_ITER 4
#endif
#ifndef ORIG_PTO_N_ITER
#define ORIG_PTO_N_ITER 6
#endif
#ifndef ORIG_PTO_BLOCK_DIM
#define ORIG_PTO_BLOCK_DIM (ORIG_PTO_M_ITER * ORIG_PTO_N_ITER)
#endif

template <typename OutTile, typename LeftTile, typename RightTile>
AICORE inline void MatmulAcc(OutTile cTile, LeftTile aTile, RightTile bTile,
                             uint32_t k) {
  if (k == 0) {
    TMATMUL(cTile, aTile, bTile);
  } else {
    TMATMUL_ACC(cTile, cTile, aTile, bTile);
  }
}

template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void SetFlag(uint32_t id) {
  set_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}
template <pipe_t srcPipe, pipe_t dstPipe>
AICORE inline void WaitFlag(uint32_t id) {
  wait_flag(srcPipe, dstPipe, static_cast<event_t>(id));
}

template <typename T, typename U, typename S, int m, int k, int n,
          uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN>
AICORE inline void InitGMOffsets(__gm__ U *&currentSrc0, __gm__ S *&currentSrc1,
                                 __gm__ T *&currentDst, __gm__ T *out,
                                 __gm__ U *src0, __gm__ S *src1) {
  // Work partition (SPMD-style):
  // - Each core owns a contiguous C tile of shape [singleCoreM, singleCoreN].
  // - It reads the corresponding A panel [singleCoreM, K] and B panel [K,
  // singleCoreN].
  constexpr uint32_t mIter = m / singleCoreM;
  uint32_t mIterIdx = get_block_idx() % mIter;  // get current launch core idx
  uint32_t nIterIdx = get_block_idx() / mIter;
  uint64_t gmOffsetA = mIterIdx * singleCoreM * k;
  uint64_t gmOffsetB = nIterIdx * k * singleCoreN;
  uint64_t gmOffsetC = mIterIdx * singleCoreM * n + nIterIdx * singleCoreN;
  currentSrc0 = src0 + gmOffsetA;
  currentSrc1 = src1 + gmOffsetB;
  currentDst = out + gmOffsetC;
}

template <typename T, typename U, typename S, int m, int k, int n,
          uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepKa,
          uint32_t stepKb, uint32_t singleCoreK>
AICORE inline void ProcessKIteration(
    uint32_t kIter, uint32_t i, uint32_t j, __gm__ U *currentSrc0,
    __gm__ S *currentSrc1,
    Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM,
         baseK * stepKa, SLayout::RowMajor>
        aMatTile[BUFFER_NUM],
    Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor,
         baseK * stepKb, baseN, SLayout::ColMajor>
        bMatTile[BUFFER_NUM],
    TileLeft<U, baseM, baseK, baseM, baseK> aTile[BUFFER_NUM],
    TileRight<S, baseK, baseN, baseK, baseN> bTile[BUFFER_NUM],
    TileAcc<T, baseM, baseN, baseM, baseN> &cTile, uint8_t &mte2DBFlag,
    uint8_t &mte1DBFlag) {
  // A panel staged by each TLOAD (GM->L1) when kModstepKa == 0: [baseM, baseK *
  // stepKa]
  using NDValidShapeA = TileShape2D<U, baseM, baseK * stepKa, Layout::ND>;
  using NDsingleCoreShapeA = BaseShape2D<U, m, k, Layout::ND>;
  using GlobalDataSrcA =
      GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA, Layout::ND>;

  // B panel staged by each TLOAD (GM->L1) when kModstepKa == 0: [baseK *
  // stepKb, baseN]
  using NDValidShapeB = TileShape2D<U, baseK * stepKb, baseN, Layout::DN>;
  using NDsingleCoreShapeB = BaseShape2D<U, k, n, Layout::DN>;
  using GlobalDataSrcB =
      GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

  const uint32_t kModstepKa = kIter % stepKa;

  // TLOAD stage:
  // - Every stepKa iterations, load a larger [baseM, baseK * stepKa] panel into
  // L1 and then slice it with TEXTRACT.
  // - Double buffering is driven by mte2DBFlag.
  if (kModstepKa == 0) {
    GlobalDataSrcA gmA(currentSrc0 + i * singleCoreK * baseM + kIter * baseK);
    GlobalDataSrcB gmB(currentSrc1 + j * singleCoreK * baseN + kIter * baseK);

    // Wait until TEXTRACT is done with this L1 buffer before reusing it.
    WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
    TLOAD(aMatTile[mte2DBFlag], gmA);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TLOAD(bMatTile[mte2DBFlag], gmB);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
    mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
  }

  const uint32_t currMte2Idx =
      (mte2DBFlag == 0) ? 1 : 0;  // mte2DBFlag reversed
  // Wait until TMATMUL is done with the current L0A/L0B buffer before
  // overwriting it via TEXTRACT.
  WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

  // TEXTRACT stage: slice the loaded L1 panel into the baseK chunk we need this
  // iteration.
  if (kModstepKa == 0) WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
  TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);

  if (kModstepKa == 0) WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
  TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK,
           0);

  if ((kIter + 1) % stepKa == 0) {
    // Allow the next TLOAD to reuse this L1 slot.
    SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
  }

  // TMATMUL stage: compute (or accumulate) into cTile.
  SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
  WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
  MatmulAcc(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag], kIter);
  // Signal that TMATMUL is done, so the next iteration may TEXTRACT into the
  // other ping-pong slot.
  SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
  mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

template <typename T, typename U, typename S, int m, int n, uint32_t baseM,
          uint32_t baseN, uint32_t singleCoreK>
AICORE inline void StoreResult(TileAcc<T, baseM, baseN, baseM, baseN> &cTile,
                               __gm__ T *currentDst, uint32_t i, uint32_t j) {
  // TSTORE stage: write the finished C tile [baseM, baseN] back to GM.
  SetFlag<PIPE_M, PIPE_FIX>(0);
  WaitFlag<PIPE_M, PIPE_FIX>(0);

  // the data size read from L0C after single k loop is [baseM, baseN]
  using NDValidShapeC = TileShape2D<T, baseM, baseN, Layout::ND>;
  using NDWholeShapeC =
      BaseShape2D<T, m, n, Layout::ND>;  // stride use global C m n
  using GlobalDataOut =
      GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;

  GlobalDataOut dstGlobal(currentDst + i * baseM * n + j * baseN);
  TSTORE(dstGlobal, cTile);

  SetFlag<PIPE_FIX, PIPE_M>(0);
  WaitFlag<PIPE_FIX, PIPE_M>(0);
}

AICORE inline void WaitSyncFlags() {
  // supplement last sync instr for reverse sync in ProcessKIteration
  WaitFlag<PIPE_M, PIPE_MTE1>(0);
  WaitFlag<PIPE_M, PIPE_MTE1>(1);
  WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
  WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
}

AICORE inline void InitSyncFlags() {
  // supplement first sync instr for reverse sync in ProcessKIteration
  SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
  SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
  SetFlag<PIPE_M, PIPE_MTE1>(0);
  SetFlag<PIPE_M, PIPE_MTE1>(1);
}

template <typename T, typename U, typename S, typename B, uint32_t blockDim,
          int m, int k, int n, int validM, int validK, int validN,
          uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN,
          uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM,
          uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
AICORE inline void RunGemmE2E(__gm__ T *out, __gm__ U *src0, __gm__ S *src1) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  __gm__ U *currentSrc0 = nullptr;
  __gm__ S *currentSrc1 = nullptr;
  __gm__ T *currentDst = nullptr;
  InitGMOffsets<T, U, S, m, k, n, singleCoreM, singleCoreK, singleCoreN>(
      currentSrc0, currentSrc1, currentDst, out, src0, src1);

  using TileMatA =
      Tile<TileType::Mat, U, baseM, baseK * stepKa, BLayout::ColMajor, baseM,
           baseK * stepKa, SLayout::RowMajor>;
  using TileMatB =
      Tile<TileType::Mat, S, baseK * stepKb, baseN, BLayout::RowMajor,
           baseK * stepKb, baseN, SLayout::ColMajor>;

  TileMatA aMatTile[BUFFER_NUM];
  TileMatB bMatTile[BUFFER_NUM];

  using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
  using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
  using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

  LeftTile aTile[BUFFER_NUM];
  RightTile bTile[BUFFER_NUM];
  ResTile cTile;

  // L1 staging buffers (aMatTile/bMatTile) are double-buffered for TLOAD
  // overlap.
  TASSIGN(aMatTile[0], 0x0);
  TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U));
  TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U));
  TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa * BUFFER_NUM * sizeof(U) +
                           baseK * baseN * stepKb * sizeof(U));

  // L0A/L0B ping-pong buffers (TEXTRACT destination).
  // Keep each per-buffer footprint <= 32 KiB to fit in a ping/pang slot.
  TASSIGN(aTile[0], 0x0);                      // L0A ping
  TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);  // L0A pang
  TASSIGN(bTile[0], 0x0);                      // L0B ping
  TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES);  // L0B pang
  TASSIGN(cTile, 0x0);

  constexpr uint32_t mLoop = singleCoreM / baseM;
  constexpr uint32_t nLoop = singleCoreN / baseN;
  constexpr uint32_t kLoop = singleCoreK / baseK;
  uint8_t mte2DBFlag = 0, mte1DBFlag = 0;

  InitSyncFlags();

  for (uint32_t i = 0; i < mLoop; i++) {
    for (uint32_t j = 0; j < nLoop; j++) {
      for (uint32_t kIter = 0; kIter < kLoop; kIter++) {
        ProcessKIteration<T, U, S, m, k, n, baseM, baseK, baseN, stepKa, stepKb,
                          singleCoreK>(kIter, i, j, currentSrc0, currentSrc1,
                                       aMatTile, bMatTile, aTile, bTile, cTile,
                                       mte2DBFlag, mte1DBFlag);
      }
      StoreResult<T, U, S, m, n, baseM, baseN, singleCoreK>(cTile, currentDst,
                                                            i, j);
    }
  }

  WaitSyncFlags();

#else

  // do nothing for Vector branch

#endif
}

__global__ AICORE void gemm_kernel_entry(__gm__ void *out, __gm__ void *src0,
                                         __gm__ void *src1) {
  constexpr uint32_t blockDim = ORIG_PTO_BLOCK_DIM;
  constexpr int m = ORIG_PTO_M, n = ORIG_PTO_N, k = ORIG_PTO_K;
  constexpr uint32_t baseM = ORIG_PTO_BASE_M;
  constexpr uint32_t baseN = ORIG_PTO_BASE_N;
  constexpr uint32_t baseK = ORIG_PTO_BASE_K;
  constexpr uint32_t singleCoreM = m / ORIG_PTO_M_ITER;
  constexpr uint32_t singleCoreN = n / ORIG_PTO_N_ITER;
  constexpr uint32_t singleCoreK = k;
  constexpr uint32_t stepM = 1;
  constexpr uint32_t stepKa = 1;
  constexpr uint32_t stepKb = 1;
  constexpr uint32_t stepN = 1;

  static_assert(m % baseM == 0, "M must be multiple of baseM");
  static_assert(n % baseN == 0, "N must be multiple of baseN");
  static_assert(k % baseK == 0, "K must be multiple of baseK");
  static_assert(m % ORIG_PTO_M_ITER == 0,
                "M must be divisible by ORIG_PTO_M_ITER");
  static_assert(n % ORIG_PTO_N_ITER == 0,
                "N must be divisible by ORIG_PTO_N_ITER");
  static_assert(singleCoreM % baseM == 0,
                "singleCoreM must be multiple of baseM");
  static_assert(singleCoreN % baseN == 0,
                "singleCoreN must be multiple of baseN");
  static_assert(singleCoreK % baseK == 0,
                "singleCoreK must be multiple of baseK");

  RunGemmE2E<float, half, half, float, blockDim, m, k, n, m, k, n, singleCoreM,
             singleCoreK, singleCoreN, baseM, baseK, baseN, stepM, stepKa,
             stepKb, stepN>(reinterpret_cast<__gm__ float *>(out),
                            reinterpret_cast<__gm__ half *>(src0),
                            reinterpret_cast<__gm__ half *>(src1));
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out,
                            uint8_t *src0, uint8_t *src1) {
  gemm_kernel_entry<<<blockDim, nullptr, stream>>>(out, src0, src1);
}
