/**
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 This program is free software, you can redistribute it and/or modify it
 under the terms and conditions of CANN Open Software License Agreement
 Version 2.0 (the "License"). Please refer to the License for details. You may
 not use this file except in compliance with the License. THIS SOFTWARE IS
 PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
 repository for the full text of the License.
*/
#pragma once

#define MEMORY_BASE
#include <pto/pto-inst.hpp>

namespace kernel_utils {
/**
 * @brief Do a sync step (set-wait flag) between two pipes.
 *
 * @tparam SrcPipe The pipe that sets the flag.
 * @tparam DstPipe The pipe that waits for the flag.
 * @param [in] id The event id to sync for.
 */
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetWaitFlag(uint32_t id) {
  set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
  wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

/**
 * @brief Performs a division on two integral numbers and rounds the result up
 * to the nearest integer.
 *
 * @tparam T1 Data type of dividend.
 * @tparam T2 Data type of divisor.
 * @param [in] value Dividend.
 * @param [in] divisor Divisor.
 * @return Result of division.
 */
template <typename T1, typename T2,
          typename std::enable_if<std::is_integral<T1>::value &&
                                      std::is_integral<T2>::value,
                                  int>::type = 0>
AICORE inline T1 CeilDiv(T1 value, T2 divisor) {
  return (value + divisor - 1) / divisor;
}

}  // namespace kernel_utils