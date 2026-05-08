/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef CSRC_KERNEL_KERNEL_SCAN_UL1_H
#define CSRC_KERNEL_KERNEL_SCAN_UL1_H

#include <stdint.h>

#include "kernel_utils.h"

#ifdef __CPU_SIM
extern "C" __global__ AICORE void scan_ul1_fp16(__gm__ void* x, __gm__ void* o,
                                                __gm__ void* u, __gm__ void* l,
                                                __gm__ void* s,
                                                uint32_t matrix_size);
extern "C" __global__ AICORE void scan_ul1_fp32(__gm__ void* x, __gm__ void* o,
                                                __gm__ void* u, __gm__ void* l,
                                                __gm__ void* s,
                                                uint32_t matrix_size);
#endif

#endif  // CSRC_KERNEL_KERNEL_SCAN_UL1_H
