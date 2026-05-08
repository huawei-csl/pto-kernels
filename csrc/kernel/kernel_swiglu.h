/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef CSRC_KERNEL_KERNEL_SWIGLU_H
#define CSRC_KERNEL_KERNEL_SWIGLU_H

#include <stdint.h>

#include "kernel_utils.h"

#ifdef __CPU_SIM
extern "C" __global__ AICORE void swiglu_fp16(GM_ADDR x, GM_ADDR y,
                                              uint32_t batch, uint32_t input_n);
#endif

#endif  // CSRC_KERNEL_KERNEL_SWIGLU_H
