/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#pragma once

#include "kernel_utils.h"

// Synchronization and packed tile I/O for the fused pipeline.
#ifdef __CCE_AICORE__

template <pipe_t Pipe>
AICORE inline void mega_signal(uint16_t flag) {
  kernel_utils::SetCubeVecFlag<Pipe>(flag);
}

AICORE inline void mega_wait(uint16_t flag) {
  kernel_utils::WaitCubeVecFlag(flag);
}

#endif

// Static full-tile descriptors avoid dynamic DMA setup on the common path.
template <int R, int C, int Stride, typename Full, typename Tail>
AICORE inline void mega_load_packed(Full& full, Tail& tail,
                                    __gm__ half* pointer, int rows, int cols) {
  if (rows == R && cols == C) {
    GlobalTensor<half, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, Stride, 1>>
        gm(pointer);
    TLOAD(full, gm);
  } else {
    GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                 pto::Stride<1, 1, 1, Stride, 1>>
        gm(pointer, {1, 1, 1, rows, cols});
    TLOAD(tail, gm);
  }
}

template <int R, int C, int Stride, typename Full, typename Tail>
AICORE inline void mega_store_packed(Full& full, Tail& tail,
                                     __gm__ half* pointer, int rows, int cols) {
  if (rows == R && cols == C) {
    GlobalTensor<half, Shape<1, 1, 1, R, C>, pto::Stride<1, 1, 1, Stride, 1>>
        gm(pointer);
    TSTORE(gm, full);
  } else {
    GlobalTensor<half, Shape<1, 1, 1, DYNAMIC, DYNAMIC>,
                 pto::Stride<1, 1, 1, Stride, 1>>
        gm(pointer, {1, 1, 1, rows, cols});
    TSTORE(gm, tail);
  }
}
