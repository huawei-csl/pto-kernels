/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef CSRC_KERNEL_KERNEL_SIMPLE_MATMUL_H
#define CSRC_KERNEL_KERNEL_SIMPLE_MATMUL_H

#include <stdint.h>

#include "kernel_utils.h"

#ifdef __CPU_SIM
extern "C" __global__ AICORE void simple_matmul_fp16(__gm__ void* a,
                                                     __gm__ void* b,
                                                     __gm__ void* c,
                                                     uint32_t matrix_size);
extern "C" __global__ AICORE void simple_matmul_bf16(__gm__ void* a,
                                                     __gm__ void* b,
                                                     __gm__ void* c,
                                                     uint32_t matrix_size);
extern "C" __global__ AICORE void simple_matmul_fp32(__gm__ void* a,
                                                     __gm__ void* b,
                                                     __gm__ void* c,
                                                     uint32_t matrix_size);
#endif

#endif  // CSRC_KERNEL_KERNEL_SIMPLE_MATMUL_H
