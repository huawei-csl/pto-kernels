/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef CSRC_KERNEL_KERNEL_TRI_INV_H
#define CSRC_KERNEL_KERNEL_TRI_INV_H

#include <stdint.h>

#include "kernel_utils.h"

#ifdef __CPU_SIM
extern "C" __global__ AICORE void triv_inv_col_sweep_fp16(GM_ADDR x, GM_ADDR z,
                                                          uint32_t in_length,
                                                          uint32_t matrix_size);
extern "C" __global__ AICORE void triv_inv_col_sweep_fp32(GM_ADDR x, GM_ADDR z,
                                                          uint32_t in_length,
                                                          uint32_t matrix_size);
#endif

#endif  // CSRC_KERNEL_KERNEL_TRI_INV_H
