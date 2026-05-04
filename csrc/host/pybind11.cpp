/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "torch_abs.h"
#include "torch_batch_matrix_square.h"
#include "torch_chunk_cumsum.h"
#include "torch_csr_gather.h"
#include "torch_scan_ul1.h"
#include "torch_simple_matmul.h"
#include "torch_swiglu.h"
#include "torch_tri_inv.h"
#include "torch_tri_inv_ns.h"
#include "torch_tri_inv_rec_unroll.h"
#include "torch_tri_inv_trick.h"

using namespace pto_isa_ops;

/**
 * @brief Pybind11 module.
 */
PYBIND11_MODULE(pto_kernels_ops, m) {
  m.doc() = "PTO-ISA Kernels";
  m.def(
      "get_aic_cores",
      [](int32_t device_id) { return pto_isa_ops::GetNumCubeCores(device_id); },
      pybind11::arg("device_id") = 0);
  m.def(
      "get_aiv_cores",
      [](int32_t device_id) {
        return pto_isa_ops::GetNumVectorCores(device_id);
      },
      pybind11::arg("device_id") = 0);
  m.def("pto_abs", &pto_isa_ops::run_abs);
  m.def("pto_chunk_cumsum", &pto_isa_ops::run_chunk_cumsum, py::arg("g"),
        py::arg("batch_size"), py::arg("seq_len"),
        py::arg("cu_seqlens") = at::zeros({1}));
  m.def("pto_batch_matrix_square", &pto_isa_ops::run_batch_matrix_square);
  m.def("pto_csr_gather", &pto_isa_ops::run_csr_gather);
  m.def("pto_scan_ul1", &pto_isa_ops::run_scan_ul1);
  m.def("pto_simple_matmul", &pto_isa_ops::run_simple_matmul);
  m.def("pto_swiglu", &pto_isa_ops::run_swiglu, py::arg("x"),
        py::arg("dim") = -1);
  m.def("pto_tri_inv_trick", &pto_isa_ops::run_tri_inv_trick);
  m.def("pto_tri_inv_rec_unroll", &pto_isa_ops::run_tri_inv_rec_unroll,
        py::arg("M"), py::arg("cu_seqlens") = at::zeros({1}),
        py::arg("is_bsnd_format") = false,
        py::arg("dtype_out") = at::ScalarType::Half);
  m.def("pto_tri_inv_ns", &pto_isa_ops::run_tri_inv_ns, py::arg("M"),
        py::arg("num_iters") = 0, py::arg("scale_value") = 0.0f);
  m.def("pto_tri_inv", &pto_isa_ops::run_tri_inv);
}
