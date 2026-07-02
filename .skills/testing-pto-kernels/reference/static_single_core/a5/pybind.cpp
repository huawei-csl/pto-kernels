#include <torch/extension.h>

#include <cstdint>
#include <string>

namespace py = pybind11;
extern "C" void call_static_add(uint32_t block_dim, void *stream, uint8_t *out,
                                uint8_t *x, uint8_t *z);
extern "C" void call_static_matmul(uint32_t block_dim, void *stream,
                                   uint8_t *out, uint8_t *a, uint8_t *b);

void launch_static_add(const at::Tensor &out, const at::Tensor &x,
                       const at::Tensor &z, uintptr_t stream) {
  call_static_add(1, reinterpret_cast<void *>(stream),
                  static_cast<uint8_t *>(out.data_ptr()),
                  static_cast<uint8_t *>(x.data_ptr()),
                  static_cast<uint8_t *>(z.data_ptr()));
}

void launch_static_matmul(const at::Tensor &out, const at::Tensor &a,
                          const at::Tensor &b, uintptr_t stream) {
  call_static_matmul(1, reinterpret_cast<void *>(stream),
                     static_cast<uint8_t *>(out.data_ptr()),
                     static_cast<uint8_t *>(a.data_ptr()),
                     static_cast<uint8_t *>(b.data_ptr()));
}

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME pto_static_a5_demo
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_static_add", &launch_static_add);
  m.def("launch_static_matmul", &launch_static_matmul);
}
