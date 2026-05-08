/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef EXTENSION_CSRC_UTILS_CPU_H
#define EXTENSION_CSRC_UTILS_CPU_H

#include <ATen/ATen.h>

#include <cstdint>
#include <tuple>
#include <type_traits>

// Dummy defintions to be used by SOC_VERSION
#define Ascend910B1 1
#define Ascend910B2 2
#define Ascend910B3 3
#define Ascend910B4 4

#if !defined(SOC_VERSION)
#define SOC_VERSION Ascend910B4
#endif

namespace pto_isa_ops {

/**
 * @brief Thread-local variables to store block information for CPU simulation.
 */
inline thread_local uint32_t g_block_num = 1;

/**
 * @brief Returns the number of Cube cores on the specified device.
 *
 * @param [in] device_id Device ID, default is 0.
 * @return uint32_t Number of Cube cores on the specified device.
 */
inline uint32_t GetNumCubeCores(int32_t device_id = 0) {
#if (SOC_VERSION == Ascend910B1) || (SOC_VERSION == Ascend910B2)
  return 24;
#elif (SOC_VERSION == Ascend910B3) || (SOC_VERSION == Ascend910B4)
  return 20;
#else
#error "Unsupported SOC_VERSION value provided."
#endif
  return 1;
}

/**
 * @brief Get the number of vector Cores.
 *
 * @param [in] device_id Device ID, default is 0.
 * @return uint32_t Number of vector cores on the specified device.
 */
inline uint32_t GetNumVectorCores(int32_t device_id = 0) {
  return 2 * GetNumCubeCores();
}

/**
 * @brief Converts the type of input tensor to (uint8_t*)
 *
 * @param [in] tensor Input tensor
 * @return uint8_t* Pointer of input tensor.
 */
inline uint8_t* ConvertType(const at::Tensor& tensor) {
  return reinterpret_cast<uint8_t*>(const_cast<void*>(tensor.storage().data()));
}

/**
 * @brief Converts any pointer type to (uint8_t*)
 *
 * @tparam T Input pointer type.
 * @param [in] value Input pointer
 * @return uint8_t* Converted pointer.
 */
template <typename T>
inline uint8_t* ConvertType(T* value) {
  return reinterpret_cast<uint8_t*>(const_cast<std::remove_const_t<T>*>(value));
}

/**
 * @brief Identity conversion for non-pointer types.
 *
 * @tparam T Input type.
 * @param [in] value Input value
 * @return T Returns same value
 */
template <typename T>
inline std::enable_if_t<!std::is_pointer_v<T>, T> ConvertType(T value) {
  return value;
}

/**
 * @brief Converts types given a variadic list.
 *
 * @tparam Ts Variadic list of types
 *
 * @param [in] args Variadic list of input arguments
 * @return Tuple of converted types.
 */
template <typename... Ts>
constexpr auto ConvertTypes(Ts&... args) {
  return std::make_tuple(ConvertType(args)...);
}

#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                  \
  do {                                                               \
    auto converted_params = pto_isa_ops::ConvertTypes(__VA_ARGS__);  \
    pto_isa_ops::g_block_num = blockdim;                             \
    for (uint32_t i = 0; i < static_cast<uint32_t>(blockdim); ++i) { \
      pto::cpu_sim::ScopedExecutionContext ctx(i, 0, 1);             \
      std::apply([&](auto&&... params) { kernel_name(params...); },  \
                 converted_params);                                  \
    }                                                                \
  } while (false)

}  // namespace pto_isa_ops

/**
 * @brief Global accessor for block number in CPU simulation.
 *
 * We need this function because pto/common/cpu_stub.hpp doesn't define it.
 */
extern "C" uint32_t get_block_num() { return pto_isa_ops::g_block_num; }

#endif  // EXTENSION_CSRC_UTILS_CPU_H
