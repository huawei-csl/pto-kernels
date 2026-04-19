/**
 * Lower triangular all-ones matrices statically allocated in the binary of the
 * kernel.
 *
 * To generate these matrices:
 *
 *
```python
import numpy as np
for s in [16, 32, 64, 128]:
    I_minus_s = -np.eye(s).astype("int")
    np.savetxt(f"matrix_{s}.csv", I_minus_s, fmt='%5.0f', delimiter=",")
    !sed '$!s/$/,/' matrix_{s}_with_trailing_comma.csv > matrix_{s}_final.csv
```
 */
#pragma once

#define MEMORY_BASE
#include <pto/pto-inst.hpp>

#define CONST_HALF_TO_GM(x) \
  reinterpret_cast<GM_ADDR>(const_cast<__gm__ half*>((x)))

// clang format off
const static __gm__ half minus_eye_fp16_16[256] = {};
const static __gm__ half minus_eye_fp16_32[1024] = {};
const static __gm__ half minus_eye_fp16_64[4096] = {};
const static __gm__ half minus_eye_fp16_128[16384] = {};
// clang format on

__aicore__ inline GM_ADDR load_minus_eye_fp16_matrix(uint32_t matmul_size) {
  if (matmul_size == 16) {
    return CONST_HALF_TO_GM(minus_eye_fp16_16);
  } else if (matmul_size == 32) {
    return CONST_HALF_TO_GM(minus_eye_fp16_32);
  } else if (matmul_size == 64) {
    return CONST_HALF_TO_GM(minus_eye_fp16_64);
  } else if (matmul_size == 128) {
    return CONST_HALF_TO_GM(minus_eye_fp16_128);
  }
}
