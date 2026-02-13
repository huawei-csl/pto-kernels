# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. This program is free
# software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the
# "License"). Please refer to the License for details. You may not use this file
# except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS"
# BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
# full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(pto-kernels)
set(LINUX TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PIP_INSTALL "True if building via pip" OFF)

find_package(PythonLibs REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

# user-defined configuration
set(SOC_VERSION
    "Ascend910b4"
    CACHE STRING "system on chip type")
set(CMAKE_BUILD_TYPE
    "Debug"
    CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
set(CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_LIST_DIR}/out"
    CACHE STRING "path for install()" FORCE)
if(DEFINED ASCEND_CANN_PACKAGE_PATH)

elseif(DEFINED ENV{ASCEND_HOME_PATH})
  set(ASCEND_CANN_PACKAGE_PATH
      "$ENV{ASCEND_HOME_PATH}"
      CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(-D_FORTIFY_SOURCE=2 -O2 -std=c++17 -Wno-macro-redefined
                    -Wno-ignored-attributes -fstack-protector-strong)

add_link_options(-s -Wl,-z,relro -Wl,-z,now)

set(CMAKE_CPP_COMPILE_OPTIONS -xc++ "SHELL:-include stdint.h"
                              "SHELL:-include stddef.h")

include_directories(${ASCEND_HOME_PATH}/include
                    ${ASCEND_DRIVER_PATH}/kernel/inc)

# set PTO_LIB_PATH before running cmake, for example: export
# PTO_LIB_PATH=[YOUR_PATH]/pto-isa
set(PTO_LIB_PATH
    "$ENV{PTO_LIB_PATH}"
    CACHE STRING "PTO library path")

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR
      ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
  set(ASCENDC_CMAKE_DIR
      ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
  set(ASCENDC_CMAKE_DIR
      ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
  message(
    FATAL_ERROR
      "ascendc_kernel_cmake does not exist, please check whether the cann package is installed."
  )
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

include(FetchContent)

FetchContent_Declare(
  libpto_isa_headers
  GIT_REPOSITORY https://gitcode.com/cann/pto-isa.git
  GIT_TAG 8.5.0)

FetchContent_Populate(libpto_isa_headers)

# TORCH_NPU_PATH is the location where PyTorch Ascend Adapter (torch_npu) is
# installed.
execute_process(
  COMMAND
    python3 -c
    "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_NPU_PATH)

message("***********************************************************")
message("* ASCEND_CANN_PACKAGE_PATH : ${ASCEND_CANN_PACKAGE_PATH}")
message("* TORCH_INSTALL_PREFIX     : ${TORCH_INSTALL_PREFIX}")
message("* TORCH_NPU_PATH           : ${TORCH_NPU_PATH}")
message("* TORCH_LIBRARIES          : ${TORCH_LIBRARIES}")
message("***********************************************************")

ascendc_library(
  no_workspace_kernel
  SHARED
  csrc/kernel/kernel_tri_inv_col_sweep.cpp
  csrc/kernel/kernel_abs.cpp
  csrc/kernel/kernel_simple_matmul.cpp
  csrc/kernel/kernel_batch_matrix_square.cpp
  csrc/kernel/kernel_tri_inv_trick.cpp)

ascendc_include_directories(
  no_workspace_kernel PRIVATE ${libpto_isa_headers_SOURCE_DIR}/include
  ${libpto_isa_headers_SOURCE_DIR}/include/pto/common)

pybind11_add_module(pto_kernels_ops csrc/host/pybind11.cpp)

target_compile_options(
  pto_kernels_ops PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS}
                          --cce-aicore-arch=dav-2201 -std=c++17)

target_link_libraries(
  pto_kernels_ops
  PRIVATE ${TORCH_LIBRARIES}
          c10
          torch_cpu
          torch_npu
          ascendcl
          tiling_api
          register
          platform
          ascendalog
          dl
          no_workspace_kernel)

target_link_directories(pto_kernels_ops PRIVATE ${TORCH_INSTALL_PREFIX}/lib
                        ${TORCH_NPU_PATH}/lib)

target_include_directories(
  pto_kernels_ops
  PRIVATE ${TORCH_NPU_PATH}/include ${TORCH_INCLUDE_DIRS}
          ${TORCH_INSTALL_PREFIX}/include
          ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include)

if(PIP_INSTALL)
  message(STATUS ">>> Running under scikit-build-core (pip install)")

  target_link_options(pto_kernels_ops PRIVATE "LINKER:-no-as-needed")

  # install pto_kernels_ops.so under site-packages/pto_kernels
  install(TARGETS pto_kernels_ops LIBRARY DESTINATION pto_kernels)

  # Set RPATH to $ORIGIN/lib (See Python package layout)
  set_target_properties(
    pto_kernels_ops
    PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
               INSTALL_RPATH_USE_LINK_PATH TRUE
               LINK_FLAGS "-Wl,-rpath,\${ORIGIN}/lib")

  # install dynamic libraries under site-packages/pto_kernels/libs
  install(TARGETS no_workspace_kernel LIBRARY DESTINATION pto_kernels/lib)
else()
  message(STATUS ">>>======================================================")
  message(STATUS ">>> Ignoring dynamic libraries COPY inside Python wheel.")
  message(STATUS ">>>======================================================")

  # Set RPATH to $(pwd)/build/lib
  set_target_properties(
    pto_kernels_ops
    PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
               INSTALL_RPATH_USE_LINK_PATH TRUE
               LINK_FLAGS "-Wl,-rpath,\${ORIGIN}/build/lib"
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

endif()
