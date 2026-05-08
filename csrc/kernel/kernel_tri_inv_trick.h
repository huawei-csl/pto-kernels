/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef CSRC_KERNEL_KERNEL_TRI_INV_TRICK_H
#define CSRC_KERNEL_KERNEL_TRI_INV_TRICK_H

#include <stdint.h>

#include "kernel_utils.h"

#ifdef __CPU_SIM
extern "C" __global__ AICORE void tri_inv_trick_fp16(__gm__ void* tensor_out,
                                                     __gm__ void* tensor_in,
                                                     __gm__ void* identity_in,
                                                     uint32_t matrix_size,
                                                     uint32_t max_block_size);
#endif

#endif  // CSRC_KERNEL_KERNEL_TRI_INV_TRICK_H
