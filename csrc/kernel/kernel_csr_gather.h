/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef CSRC_KERNEL_KERNEL_CSR_GATHER_H
#define CSRC_KERNEL_KERNEL_CSR_GATHER_H

#include <stdint.h>

#include "kernel_utils.h"

#ifdef __CPU_SIM
extern "C" __global__ AICORE void csr_gather_fp16(GM_ADDR values,
                                                  GM_ADDR indices, GM_ADDR x,
                                                  GM_ADDR z, uint32_t x_size,
                                                  uint32_t indices_size);
extern "C" __global__ AICORE void csr_gather_fp32(GM_ADDR values,
                                                  GM_ADDR indices, GM_ADDR x,
                                                  GM_ADDR z, uint32_t x_size,
                                                  uint32_t indices_size);
#endif

#endif  // CSRC_KERNEL_KERNEL_CSR_GATHER_H
