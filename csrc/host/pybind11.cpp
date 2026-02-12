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
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "torch_abs.h"
#include "torch_batch_matrix_square.h"
#include "torch_simple_matmul.h"
#include "torch_tri_inv.h"
#include "torch_tri_inv_rec_unroll.h"
#include "torch_tri_inv_trick.h"

using namespace pto_isa_ops;

/**
 * @brief Pybind11 module.
 */
PYBIND11_MODULE(pto_kernels_ops, m) {
  m.doc() = "PTO-ISA Kernels";
  m.def("pto_abs", &pto_isa_ops::run_abs);
  m.def("pto_batch_matrix_square", &pto_isa_ops::run_batch_matrix_square);
  m.def("pto_simple_matmul", &pto_isa_ops::run_simple_matmul);
  m.def("pto_tri_inv_rec_unroll", &pto_isa_ops::run_tri_inv_rec_unroll);
  m.def("pto_tri_inv_trick", &pto_isa_ops::run_tri_inv_trick);
  m.def("pto_tri_inv", &pto_isa_ops::run_tri_inv);
}
