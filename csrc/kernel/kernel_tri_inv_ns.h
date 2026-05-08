/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#ifndef CSRC_KERNEL_KERNEL_TRI_INV_NS_H
#define CSRC_KERNEL_KERNEL_TRI_INV_NS_H

#include <stdint.h>

#include "kernel_utils.h"

#ifdef __CPU_SIM
extern "C" __global__ AICORE void tri_inv_ns_fp16(
    __gm__ void* tensor_out, __gm__ void* tensor_in,
    __gm__ void* identity_neg_in, __gm__ void* identity_over_n_in,
    uint32_t matrix_size, uint32_t num_iters, uint32_t num_matrices);
#endif

#endif  // CSRC_KERNEL_KERNEL_TRI_INV_NS_H
