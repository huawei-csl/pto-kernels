/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

/**
 * @file int4_cvt.hpp
 * @brief Type Conversion (TCVT) implementation for packed FP16 -> INT4
 * conversion
 *
 * FILE ORGANIZATION (for easy navigation):
 * =======================================
 *
 * SUPPORTED CONVERSIONS (quick lookup):
 * ====================================
 * FP16:  -> packed S4
 *
 * 1. GenCastCallFp16ToInt4 helper
 *    - fp16 -> packed int4
 *
 * 2. TCvtHeadFp16ToInt4Packed
 *    - Processes aligned repeat blocks for the main data region
 *
 * 3. TCvtFp16ToInt4Packed
 *    - Handles aligned region and remainder with vector masking
 *
 * 4. TCVT_FP16_TO_INT4_PACKED_IMPL / TCVT_FP16_TO_INT4_PACKED
 *    - High-level entry points computing repeat configuration
 *
 * QUICK FIND: Search for "GenCastCallFp16ToInt4" or "TCVT_FP16_TO_INT4_PACKED".
 */

/*
CANN 8.5.0's TCVT does not support int4, thus this custom wrapper as workaround
*/
#ifndef FAST_HADAMARD_INTCVT4_HPP
#define FAST_HADAMARD_INTCVT4_HPP

#include <pto/pto-inst.hpp>

using namespace pto;

namespace fast_hadamard_int4 {
inline namespace TCvtInternel {
// CTRL[59] controls saturation mode for FP to INT conversions:
// - 0 (ON):  Clamp to datatype range
// - 1 (OFF): Truncate via bit masking
constexpr const int SAT_MODE_BIT = 59;
}  // namespace TCvtInternel

// FP16 -> INT4 conversion (packed byte storage)
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt4(__ubuf__ void *dst,
                                        __ubuf__ typename TileDataS::DType *src,
                                        uint8_t repeatNum, RoundMode mode,
                                        uint16_t dstBlockStride,
                                        uint16_t srcBlockStride,
                                        uint16_t dstRepeatStride,
                                        uint16_t srcRepeatStride) {
  switch (static_cast<RoundMode>(mode)) {
    case RoundMode::CAST_RINT:
      vconv_f162s4r(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_ROUND:
      vconv_f162s4a(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_FLOOR:
      vconv_f162s4f(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_CEIL:
      vconv_f162s4c(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_TRUNC:
      vconv_f162s4z(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
    case RoundMode::CAST_NONE:
      vconv_f162s4(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                   dstRepeatStride, srcRepeatStride);
      break;
    default:
      vconv_f162s4z(dst, src, repeatNum, dstBlockStride, srcBlockStride,
                    dstRepeatStride, srcRepeatStride);
      break;
  }
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void GenCastCallFp16ToInt4None(
    __ubuf__ void *dst, __ubuf__ typename TileDataS::DType *src,
    uint8_t repeatNum, uint16_t dstBlockStride, uint16_t srcBlockStride,
    uint16_t dstRepeatStride, uint16_t srcRepeatStride) {
  vconv_f162s4(dst, src, repeatNum, dstBlockStride, srcBlockStride,
               dstRepeatStride, srcRepeatStride);
}

// ============================================================================
// Tile Conversion Helper: Process Main Data Block
// ============================================================================
// Packed int4 uses byte storage where each destination byte stores two signed
// int4 values. The source is advanced in fp16 elements while the destination is
// advanced in packed bytes.
template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
PTO_INST void TCvtHeadFp16ToInt4Packed(
    __ubuf__ typename TileDataD::DType *dstPtr,
    __ubuf__ typename TileDataS::DType *srcPtr, RoundMode mode,
    unsigned numRepeatPerLine, unsigned validRow, unsigned srcElementsPerRepeat,
    unsigned dstBytesPerRepeat, unsigned dstRepeatStride,
    unsigned srcRepeatStride) {
  unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
  unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
  if (mode == RoundMode::CAST_NONE) {
    for (uint32_t i = 0; i < validRow; i++) {
      if (numLoop > 0) {
        for (uint32_t j = 0; j < numLoop; j++) {
          GenCastCallFp16ToInt4None<TileDataD, TileDataS>(
              (__ubuf__ void *)(dstPtr + i * DS +
                                j * dstBytesPerRepeat * REPEAT_MAX),
              srcPtr + i * SS + j * srcElementsPerRepeat * REPEAT_MAX,
              (uint8_t)REPEAT_MAX, 1, 1, (uint16_t)dstRepeatStride,
              (uint16_t)srcRepeatStride);
        }
      }
      if (remainAfterLoop > 0) {
        GenCastCallFp16ToInt4None<TileDataD, TileDataS>(
            (__ubuf__ void *)(dstPtr + i * DS +
                              numLoop * dstBytesPerRepeat * REPEAT_MAX),
            srcPtr + i * SS + numLoop * srcElementsPerRepeat * REPEAT_MAX,
            (uint8_t)remainAfterLoop, 1, 1, (uint16_t)dstRepeatStride,
            (uint16_t)srcRepeatStride);
      }
    }
    return;
  }

  for (uint32_t i = 0; i < validRow; i++) {
    if (numLoop > 0) {
      for (uint32_t j = 0; j < numLoop; j++) {
        GenCastCallFp16ToInt4<TileDataD, TileDataS>(
            (__ubuf__ void *)(dstPtr + i * DS +
                              j * dstBytesPerRepeat * REPEAT_MAX),
            srcPtr + i * SS + j * srcElementsPerRepeat * REPEAT_MAX,
            (uint8_t)REPEAT_MAX, mode, 1, 1, (uint16_t)dstRepeatStride,
            (uint16_t)srcRepeatStride);
      }
    }
    if (remainAfterLoop > 0) {
      GenCastCallFp16ToInt4<TileDataD, TileDataS>(
          (__ubuf__ void *)(dstPtr + i * DS +
                            numLoop * dstBytesPerRepeat * REPEAT_MAX),
          srcPtr + i * SS + numLoop * srcElementsPerRepeat * REPEAT_MAX,
          (uint8_t)remainAfterLoop, mode, 1, 1, (uint16_t)dstRepeatStride,
          (uint16_t)srcRepeatStride);
    }
  }
}

// ============================================================================
// Core Tile Conversion Kernel
// ============================================================================
// TCvtFp16ToInt4Packed orchestrates the complete packed int4 conversion by
// handling both:
//   1. Aligned region: Complete repeat units processed via
//   TCvtHeadFp16ToInt4Packed
//   2. Remainder region: Partial repeats processed with vector masking
//
// Template parameters:
//   SS: Source row stride
//   DS: Destination row stride
//
// @param dst: Destination tile (packed int4 byte storage)
// @param src: Source tile (fp16 values)
// @param mode: Rounding mode (RINT/ROUND/FLOOR/CEIL/TRUNC/NONE)
// @param numRepeatPerLine: Number of complete repeats per line
// @param numRemainPerLine: Remaining fp16 elements per line (must be even)
// @param validRow: Number of rows containing valid data
// @param srcElementsPerRepeat: Number of fp16 source elements per repeat
// @param dstBytesPerRepeat: Number of packed destination bytes per repeat
// @param dstRepeatStride: Stride between repeats in destination buffer
// @param srcRepeatStride: Stride between repeats in source buffer
template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
__tf__ AICORE void TCvtFp16ToInt4Packed(
    typename TileDataD::TileDType __out__ dst,
    typename TileDataS::TileDType __in__ src, RoundMode mode,
    unsigned numRepeatPerLine, unsigned numRemainPerLine, unsigned validRow,
    unsigned srcElementsPerRepeat, unsigned dstBytesPerRepeat,
    unsigned dstRepeatStride, unsigned srcRepeatStride) {
  // Save the original saturation mode state and force saturation ON.
  uint64_t originalCtrl = get_ctrl();
  bool originalSatMode = (originalCtrl & (1ULL << SAT_MODE_BIT)) == 0;
  set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT));

  __ubuf__ typename TileDataD::DType *dstPtr =
      (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
  __ubuf__ typename TileDataS::DType *srcPtr =
      (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src);
  constexpr unsigned dstNElemPerBlock =
      BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType);
  constexpr unsigned srcNElemPerBlock =
      BLOCK_BYTE_SIZE / sizeof(typename TileDataS::DType);

  if (numRepeatPerLine > 0) {
    TCvtHeadFp16ToInt4Packed<TileDataD, TileDataS, SS, DS>(
        dstPtr, srcPtr, mode, numRepeatPerLine, validRow, srcElementsPerRepeat,
        dstBytesPerRepeat, dstRepeatStride, srcRepeatStride);
  }

  dstPtr += numRepeatPerLine * dstBytesPerRepeat;
  srcPtr += numRepeatPerLine * srcElementsPerRepeat;

  if (numRemainPerLine > 0) {
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    SetContinuousMask(numRemainPerLine);
    if (mode == RoundMode::CAST_NONE) {
      if (numLoop > 0) {
        for (uint32_t j = 0; j < numLoop; j++) {
          GenCastCallFp16ToInt4None<TileDataD, TileDataS>(
              (__ubuf__ void *)(dstPtr + j * DS * REPEAT_MAX),
              srcPtr + j * SS * REPEAT_MAX, (uint8_t)REPEAT_MAX, 1, 1,
              (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
        }
      }
      if (remainAfterLoop > 0) {
        GenCastCallFp16ToInt4None<TileDataD, TileDataS>(
            (__ubuf__ void *)(dstPtr + numLoop * DS * REPEAT_MAX),
            srcPtr + numLoop * SS * REPEAT_MAX, (uint8_t)remainAfterLoop, 1, 1,
            (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
      }
    } else {
      if (numLoop > 0) {
        for (uint32_t j = 0; j < numLoop; j++) {
          GenCastCallFp16ToInt4<TileDataD, TileDataS>(
              (__ubuf__ void *)(dstPtr + j * DS * REPEAT_MAX),
              srcPtr + j * SS * REPEAT_MAX, (uint8_t)REPEAT_MAX, mode, 1, 1,
              (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
        }
      }
      if (remainAfterLoop > 0) {
        GenCastCallFp16ToInt4<TileDataD, TileDataS>(
            (__ubuf__ void *)(dstPtr + numLoop * DS * REPEAT_MAX),
            srcPtr + numLoop * SS * REPEAT_MAX, (uint8_t)remainAfterLoop, mode,
            1, 1, (uint16_t)DS / dstNElemPerBlock,
            (uint16_t)SS / srcNElemPerBlock);
      }
    }
    set_vector_mask(-1, -1);
  }

  if (originalSatMode) {
    set_ctrl(sbitset0(get_ctrl(), SAT_MODE_BIT));
  } else {
    set_ctrl(sbitset1(get_ctrl(), SAT_MODE_BIT));
  }
}

template <typename TileDataD, typename TileDataS, unsigned SS, unsigned DS>
__tf__ AICORE void TCvtFp16ToInt4PackedNoCtrl(
    typename TileDataD::TileDType __out__ dst,
    typename TileDataS::TileDType __in__ src, RoundMode mode,
    unsigned numRepeatPerLine, unsigned numRemainPerLine, unsigned validRow,
    unsigned srcElementsPerRepeat, unsigned dstBytesPerRepeat,
    unsigned dstRepeatStride, unsigned srcRepeatStride) {
  __ubuf__ typename TileDataD::DType *dstPtr =
      (__ubuf__ typename TileDataD::DType *)__cce_get_tile_ptr(dst);
  __ubuf__ typename TileDataS::DType *srcPtr =
      (__ubuf__ typename TileDataS::DType *)__cce_get_tile_ptr(src);
  constexpr unsigned dstNElemPerBlock =
      BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType);
  constexpr unsigned srcNElemPerBlock =
      BLOCK_BYTE_SIZE / sizeof(typename TileDataS::DType);

  if (numRepeatPerLine > 0) {
    unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
    unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
    if (mode == RoundMode::CAST_NONE) {
      for (uint32_t i = 0; i < validRow; i++) {
        if (numLoop > 0) {
          for (uint32_t j = 0; j < numLoop; j++) {
            GenCastCallFp16ToInt4None<TileDataD, TileDataS>(
                (__ubuf__ void *)(dstPtr + i * DS +
                                  j * dstBytesPerRepeat * REPEAT_MAX),
                srcPtr + i * SS + j * srcElementsPerRepeat * REPEAT_MAX,
                (uint8_t)REPEAT_MAX, 1, 1, (uint16_t)dstRepeatStride,
                (uint16_t)srcRepeatStride);
          }
        }
        if (remainAfterLoop > 0) {
          GenCastCallFp16ToInt4None<TileDataD, TileDataS>(
              (__ubuf__ void *)(dstPtr + i * DS +
                                numLoop * dstBytesPerRepeat * REPEAT_MAX),
              srcPtr + i * SS + numLoop * srcElementsPerRepeat * REPEAT_MAX,
              (uint8_t)remainAfterLoop, 1, 1, (uint16_t)dstRepeatStride,
              (uint16_t)srcRepeatStride);
        }
      }
    } else {
      TCvtHeadFp16ToInt4Packed<TileDataD, TileDataS, SS, DS>(
          dstPtr, srcPtr, mode, numRepeatPerLine, validRow,
          srcElementsPerRepeat, dstBytesPerRepeat, dstRepeatStride,
          srcRepeatStride);
    }
  }

  dstPtr += numRepeatPerLine * dstBytesPerRepeat;
  srcPtr += numRepeatPerLine * srcElementsPerRepeat;

  if (numRemainPerLine > 0) {
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    SetContinuousMask(numRemainPerLine);
    if (mode == RoundMode::CAST_NONE) {
      if (numLoop > 0) {
        for (uint32_t j = 0; j < numLoop; j++) {
          GenCastCallFp16ToInt4None<TileDataD, TileDataS>(
              (__ubuf__ void *)(dstPtr + j * DS * REPEAT_MAX),
              srcPtr + j * SS * REPEAT_MAX, (uint8_t)REPEAT_MAX, 1, 1,
              (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
        }
      }
      if (remainAfterLoop > 0) {
        GenCastCallFp16ToInt4None<TileDataD, TileDataS>(
            (__ubuf__ void *)(dstPtr + numLoop * DS * REPEAT_MAX),
            srcPtr + numLoop * SS * REPEAT_MAX, (uint8_t)remainAfterLoop, 1, 1,
            (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
      }
    } else {
      if (numLoop > 0) {
        for (uint32_t j = 0; j < numLoop; j++) {
          GenCastCallFp16ToInt4<TileDataD, TileDataS>(
              (__ubuf__ void *)(dstPtr + j * DS * REPEAT_MAX),
              srcPtr + j * SS * REPEAT_MAX, (uint8_t)REPEAT_MAX, mode, 1, 1,
              (uint16_t)DS / dstNElemPerBlock, (uint16_t)SS / srcNElemPerBlock);
        }
      }
      if (remainAfterLoop > 0) {
        GenCastCallFp16ToInt4<TileDataD, TileDataS>(
            (__ubuf__ void *)(dstPtr + numLoop * DS * REPEAT_MAX),
            srcPtr + numLoop * SS * REPEAT_MAX, (uint8_t)remainAfterLoop, mode,
            1, 1, (uint16_t)DS / dstNElemPerBlock,
            (uint16_t)SS / srcNElemPerBlock);
      }
    }
    set_vector_mask(-1, -1);
  }
}

// ============================================================================
// High-Level Tile Conversion Interface
// ============================================================================
// TCVT_FP16_TO_INT4_PACKED_IMPL is the main entry point for packed fp16 ->
// int4. Calculates repeat configuration and delegates to the packed kernel.
template <bool ManageSatMode, typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_FP16_TO_INT4_PACKED_IMPL(TileDataD &dst, TileDataS &src,
                                                RoundMode mode) {
  static_assert(std::is_same<typename TileDataD::DType, int8_t>::value,
                "Packed int4 destination must use int8_t.");
  static_assert(std::is_same<typename TileDataS::DType, half>::value,
                "Packed int4 conversion expects fp16 source.");

  if (dst.GetValidRow() != src.GetValidRow()) {
    return;
  }

  unsigned logicalSrcCol = src.GetValidCol();
  if ((logicalSrcCol & 1U) != 0 || dst.GetValidCol() * 2U != logicalSrcCol) {
    return;
  }

  constexpr unsigned SS = TileDataS::RowStride;
  constexpr unsigned DS = TileDataD::RowStride;
  unsigned srcElementsPerRepeat =
      REPEAT_BYTE / sizeof(typename TileDataS::DType);
  unsigned dstBytesPerRepeat = srcElementsPerRepeat / 2;
  unsigned dstRepeatStride =
      dstBytesPerRepeat / (BLOCK_BYTE_SIZE / sizeof(typename TileDataD::DType));
  unsigned srcRepeatStride =
      srcElementsPerRepeat /
      (BLOCK_BYTE_SIZE / sizeof(typename TileDataS::DType));
  unsigned numRepeatPerLine = logicalSrcCol / srcElementsPerRepeat;
  unsigned numRemainPerLine = logicalSrcCol % srcElementsPerRepeat;
  unsigned validRow = dst.GetValidRow();

  if constexpr (ManageSatMode) {
    TCvtFp16ToInt4Packed<TileDataD, TileDataS, SS, DS>(
        dst.data(), src.data(), mode, numRepeatPerLine, numRemainPerLine,
        validRow, srcElementsPerRepeat, dstBytesPerRepeat, dstRepeatStride,
        srcRepeatStride);
  } else {
    TCvtFp16ToInt4PackedNoCtrl<TileDataD, TileDataS, SS, DS>(
        dst.data(), src.data(), mode, numRepeatPerLine, numRemainPerLine,
        validRow, srcElementsPerRepeat, dstBytesPerRepeat, dstRepeatStride,
        srcRepeatStride);
  }
}

template <typename DstTile, typename SrcTile>
AICORE void TCVT_FP16_TO_INT4_PACKED(DstTile &dst, SrcTile &src,
                                     RoundMode mode) {
  TCVT_FP16_TO_INT4_PACKED_IMPL<true>(dst, src, mode);
}

template <typename DstTile, typename SrcTile>
AICORE void TCVT_FP16_TO_INT4_PACKED_NOCTRL(DstTile &dst, SrcTile &src,
                                            RoundMode mode) {
  TCVT_FP16_TO_INT4_PACKED_IMPL<false>(dst, src, mode);
}

}  // namespace fast_hadamard_int4

#endif
