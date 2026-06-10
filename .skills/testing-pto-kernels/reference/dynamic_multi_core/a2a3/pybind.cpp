#include <torch/extension.h>
#include <cstdint>
#include <string>

namespace py = pybind11;

extern "C" void call_add(uint32_t block_dim, void *stream, uint8_t *y,
                         uint8_t *x, uint8_t *z, uint32_t n);
extern "C" void call_matmul(uint32_t block_dim, void *stream, uint8_t *a,
                            uint8_t *b, uint8_t *c, uint32_t m);

void launch_add(const at::Tensor &y, const at::Tensor &x,
                const at::Tensor &z, uint32_t n, uint32_t block_dim, uintptr_t stream) {
  call_add(block_dim, reinterpret_cast<void *>(stream), static_cast<uint8_t *>(y.data_ptr()),
           static_cast<uint8_t *>(x.data_ptr()), static_cast<uint8_t *>(z.data_ptr()), n);
}

void launch_matmul(const at::Tensor &a, const at::Tensor &b,
                   const at::Tensor &c, uint32_t m, uint32_t block_dim, uintptr_t stream) {
  call_matmul(block_dim, reinterpret_cast<void *>(stream), static_cast<uint8_t *>(a.data_ptr()),
              static_cast<uint8_t *>(b.data_ptr()), static_cast<uint8_t *>(c.data_ptr()), m);
}

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME pto_dynamic_a2a3_demo
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "A2A3 PTO demo pybind launcher";
  m.def("launch_add", &launch_add, py::arg("y"), py::arg("x"),
        py::arg("z"), py::arg("n"), py::arg("block_dim"), py::arg("stream"));
  m.def("launch_matmul", &launch_matmul, py::arg("a"), py::arg("b"),
        py::arg("c"), py::arg("m"), py::arg("block_dim"), py::arg("stream"));
}
