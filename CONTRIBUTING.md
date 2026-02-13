# Contributing a new kernel

This example shows how to contribute a new custom PTO-based kernel and expose it as a PyTorch operator via `torch_npu`.

## Directory Layout

```
pto-isa-kernels/
├── csrc/
│   ├── kernel/                # PTO kernel implementation
│   └── host/                  # Host-side PyTorch operator registration
├── test/                      # Python tests
├── CMakeLists.txt             # Build configuration
└── CONTRIBUTING.md            # This document
```

## 1. Implement the kernel

Add a kernel source file under `csrc/add/csrc/kernel/` and include it in the build. For example, to build `kernel_add_custom.cpp`, add it to `pto-isa-kernels/CMakeLists.txt`:

```cmake
ascendc_library(no_workspace_kernel SHARED
    ...
    csrc/kernel/kernel_add_custom.cpp
    ...
)
```

For build options and details, refer to the Ascend community documentation: https://www.hiascend.com/ascend-c

## 2. Integrate with PyTorch (`torch_npu`)

The host-side implementation lives under `csrc/host/`. See the `torch_abs.h` example and don't forget to register your kernel on `csrc/host/pybind11.cpp`.
