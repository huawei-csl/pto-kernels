/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

#define GM_ADDR __gm__ uint8_t*  // To avoid #include "kernel_operator.h"
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr unsigned UB_SIZE = 0x30000;  // 192KB UB of A2A3


template <typename InputT, typename OutputT, uint32_t matrix_size>
AICORE void runKernelScanUl1(__gm__ InputT* x,
                                  __gm__ OutputT* s) {

#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
#else
// Nothing to do on VEC
#endif
}

template <typename T>
AICORE void run_scan_ul1(__gm__ T* x, __gm__ float* s,
                              uint32_t matrix_size) {
  static_assert(std::is_same_v<T, half> or std::is_same_v<T, float>,
                "scan_ul1 supports only fp16/fp32.");

  switch (matrix_size) {
    case 16:
      runKernelScanUl1<T, float, 16>(x, s);
      break;
    case 32:
      runKernelScanUl1<T, float, 32>(x, s);
      break;

    case 64:
      runKernelScanUl1<T, float, 64>(x, s);
      break;

    case 96:
      runKernelScanUl1<T, float, 96>(x, s);
      break;

    case 128:
      runKernelScanUl1<T, float, 128>(x, s);
      break;
  }
}

extern "C" __global__ AICORE void scan_ul1_fp16(__gm__ void* x,
                                                     __gm__ void* s,
                                                     uint32_t matrix_size) {
  run_scan_ul1<half>((__gm__ half*)x, (__gm__ float*)s,
                          matrix_size);
}

extern "C" __global__ AICORE void scan_ul1_fp32(__gm__ void* x,
                                                     __gm__ void* s,
                                                     uint32_t matrix_size) {
  run_scan_ul1<float>((__gm__ float*)x, (__gm__ float*)s,
                           matrix_size);
}
