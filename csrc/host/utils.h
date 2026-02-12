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
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace pto_isa_ops {

#define DEVICE_TYPE c10::DeviceType::PrivateUse1

inline at::Tensor CopyTensorHostToDevice(const at::Tensor& cpu_tensor) {
  at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
  int deviceIndex = 0;
  c10_npu::GetDevice(&deviceIndex);
  return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex),
                            cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar& cpu_scalar,
                                     at::ScalarType scalar_data_type) {
  return CopyTensorHostToDevice(
      scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline void* ConvertType(const at::Tensor& at_tensor) {
  return const_cast<void*>(at_tensor.storage().data());
}

template <typename T>
T ConvertType(T value) {
  return value;
}

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
