/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/
#pragma once

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch/library.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace pto_isa_ops {

#define DEVICE_TYPE c10::DeviceType::PrivateUse1

/**
 * @brief Queries an integer device attribute, caching the result.
 *
 * The core counts are fixed properties of the device, so the ACL query is done
 * once per (device, attribute) pair and served from a cache afterwards. The
 * cache is shared by all threads and guarded by a mutex.
 *
 * @param [in] device_id Device ID.
 * @param [in] attr Device attribute to query.
 * @param [in] what Name of the caller, used in error messages.
 * @return uint32_t Value of the attribute on the specified device.
 */
inline uint32_t GetCachedDeviceInfo(int32_t device_id, aclrtDevAttr attr,
                                    const char* what) {
  // aclrtGetDevice overwrites device_id with the device bound to this thread;
  // the cache is keyed on that resolved id, not on the argument.
  auto ret = aclrtGetDevice(&device_id);
  TORCH_CHECK(ret == ACL_SUCCESS, what, ": aclrtGetDevice failed, error ",
              static_cast<int>(ret));

  static std::mutex cache_mutex;
  static std::map<std::pair<int32_t, int32_t>, uint32_t> cache;
  const auto key = std::make_pair(device_id, static_cast<int32_t>(attr));

  std::lock_guard<std::mutex> lock(cache_mutex);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  int64_t value = 0;
  ret = aclrtGetDeviceInfo(device_id, attr, &value);
  TORCH_CHECK(ret == ACL_SUCCESS, what, ": aclrtGetDeviceInfo failed, error ",
              static_cast<int>(ret));
  const auto result = static_cast<uint32_t>(value);
  cache.emplace(key, result);
  return result;
}

/**
 * @brief Returns the number of Cube cores on the specified device.
 *
 * Important: aclInit() must be called before this function to initialize the
 * ACL runtime. The result is cached per device after the first call.
 *
 * @param [in] device_id Device ID, default is 0.
 * @return uint32_t Number of Cube cores on the specified device.
 */
inline uint32_t GetNumCubeCores(int32_t device_id = 0) {
  return GetCachedDeviceInfo(device_id, ACL_DEV_ATTR_AICORE_CORE_NUM,
                             "GetNumCubeCores");
}

/**
 * @brief Get the number of vector Cores.
 *
 * Important: aclInit() must be called before this function to initialize the
 * ACL runtime. The result is cached per device after the first call.
 *
 * @param [in] device_id Device ID, default is 0.
 * @return uint32_t Number of vector cores on the specified device.
 */
inline uint32_t GetNumVectorCores(int32_t device_id = 0) {
  return GetCachedDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                             "GetNumVectorCores");
}

/**
 * @brief Copies a tensor from host to device.
 *
 * @param [in] cpu_tensor Input CPU tesnor
 * @return at::Tensor Tensor copied from input on current NPU device.
 */
inline at::Tensor CopyTensorHostToDevice(const at::Tensor& cpu_tensor) {
  at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
  int deviceIndex = 0;
  c10_npu::GetDevice(&deviceIndex);
  return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex),
                            cpuPinMemTensor.scalar_type(), true, true);
}

/**
 * @brief Copies a scalar into a NPU device tensor.
 *
 * @param [in] cpu_scalar Scalar on host/CPU.
 * @param [in] scalar_data_type Data type of scalar
 * @return at::Tensor Tensor on NPU containing the `cpu_scalar`.
 */
inline at::Tensor CopyScalarToDevice(const c10::Scalar& cpu_scalar,
                                     at::ScalarType scalar_data_type) {
  return CopyTensorHostToDevice(
      scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

/**
 * @brief Converts the type of input tensor to (void*)
 *
 * @param [in] tensor Input tensor
 * @return void* Pointer of input tensor.
 */
inline void* ConvertType(const at::Tensor& tensor) {
  // Contiguous views may start inside their storage. The kernel must receive
  // the first element of the tensor, not the base of its backing allocation.
  return tensor.data_ptr();
}

/**
 * @brief Identity type conversion
 *
 * @tparam T Input type.
 *
 * @param [in] value Input value
 * @return T Returns same value
 */
template <typename T>
T ConvertType(T value) {
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

#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                     \
  do {                                                                  \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);     \
    auto converted_params = pto_isa_ops::ConvertTypes(__VA_ARGS__);     \
    auto acl_call = [acl_stream, blockdim, converted_params]() -> int { \
      std::apply(                                                       \
          [&](auto&&... params) {                                       \
            pto_launch_##kernel_name(blockdim, acl_stream, params...);  \
          },                                                            \
          converted_params);                                            \
      return 0;                                                         \
    };                                                                  \
    at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);        \
  } while (false)
}  // namespace pto_isa_ops
