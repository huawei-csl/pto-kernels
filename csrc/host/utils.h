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

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace pto_isa_ops {

#define DEVICE_TYPE c10::DeviceType::PrivateUse1

// Copied from tools/build/asc_rt/ascendc_runtime.h to avoid dependency on the
// header file. See
// https://gitcode.com/cann/asc-devkit/blob/v8.5.0/tools/build/asc_rt/ascendc_runtime.h
#define ASSERT_RETVAL(exp, ret)         \
  do {                                  \
    if (!(exp)) {                       \
      printf("Assert %s failed", #exp); \
      return (ret);                     \
    }                                   \
  } while (0)

#define ASSERT_RTOK_RETVAL(v) ASSERT_RETVAL(((v) == 0), (1))

/**
 * @brief Returns the number of Cube cores on the specified device.
 *
 * @param [in] device_id Device ID, default is 0.
 * @return uint32_t Number of Cube cores on the specified device.
 */
uint32_t GetNumCubeCores(int32_t device_id = 0) {
  int64_t aicoreNum64 = 0;
  ASSERT_RTOK_RETVAL(aclrtGetDevice(&device_id));
  ASSERT_RTOK_RETVAL(aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_AICORE_CORE_NUM,
                                        &aicoreNum64));
  return static_cast<uint32_t>(aicoreNum64);
}

/**
 * @brief Get the number of vector Cores.
 *
 * @param [in] device_id Device ID, default is 0.
 * @return uint32_t Number of vector cores on the specified device.
 */
uint32_t GetNumVectorCores(int32_t device_id = 0) {
  int64_t numVectorCores = 0;
  ASSERT_RTOK_RETVAL(aclrtGetDevice(&device_id));
  ASSERT_RTOK_RETVAL(aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                        &numVectorCores));
  return static_cast<uint32_t>(numVectorCores);
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
  return const_cast<void*>(tensor.storage().data());
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

#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                            \
  do {                                                                         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);            \
    auto converted_params = pto_isa_ops::ConvertTypes(__VA_ARGS__);            \
    auto acl_call = [acl_stream, blockdim, converted_params]() -> int {        \
      std::apply(                                                              \
          [&](auto&&... params) {                                              \
            ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, acl_stream, params...); \
          },                                                                   \
          converted_params);                                                   \
      return 0;                                                                \
    };                                                                         \
    at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);               \
  } while (false)
}  // namespace pto_isa_ops
