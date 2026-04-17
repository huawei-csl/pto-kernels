/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#if defined __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// Placeholder for VEC compilation (the real kernel is CUBE-only).
#define MEMORY_BASE
#include <pto/common/type.hpp>

extern "C" __global__ AICORE void scan_ul1_fp16(__gm__ void* x,
                                                     __gm__ void* s,
                                                     uint32_t matrix_size) {}

extern "C" __global__ AICORE void scan_ul1_fp32(__gm__ void* x,
                                                     __gm__ void* s,
                                                     uint32_t matrix_size) {}

#elif (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))

#define MEMORY_BASE

#include <pto/pto-inst.hpp>

#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"

using namespace pto;

constexpr unsigned UB_SIZE = 0x30000;  // 192KB UB of A2A3


template <typename InputT, typename OutputT, uint32_t matrix_size>
AICORE void runKernelSimpleMatMul(__gm__ InputT* a,
                                  __gm__ OutputT* c) {

}

template <typename T>
AICORE void run_scan_ul1(__gm__ T* a, __gm__ float* c,
                              uint32_t matrix_size) {
  static_assert(std::is_same_v<T, half> or std::is_same_v<T, float>,
                "scan_ul1 supports only fp16/fp32.");

  switch (matrix_size) {
    case 16:
      runKernelSimpleMatMul<T, float, 16>(a, b, c);
      break;
    case 32:
      runKernelSimpleMatMul<T, float, 32>(a, b, c);
      break;

    case 64:
      runKernelSimpleMatMul<T, float, 64>(a, b, c);
      break;

    case 96:
      runKernelSimpleMatMul<T, float, 96>(a, b, c);
      break;

    case 128:
      runKernelSimpleMatMul<T, float, 128>(a, b, c);
      break;
  }
}

extern "C" __global__ AICORE void scan_ul1_fp16(__gm__ void* x,
                                                     __gm__ void* s,
                                                     uint32_t matrix_size) {
  run_scan_ul1<half>((__gm__ half*)a, (__gm__ float*)c,
                          matrix_size);
}

extern "C" __global__ AICORE void scan_ul1_fp32(__gm__ void* x,
                                                     __gm__ void* s,
                                                     uint32_t matrix_size) {
  run_scan_ul1<float>((__gm__ float*)a, (__gm__ float*)c,
                           matrix_size);
}

#endif
