/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
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
