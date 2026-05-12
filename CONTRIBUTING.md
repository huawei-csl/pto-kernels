# Contributing a new kernel

This example shows how to contribute a new custom PTO kernel and expose it as a PyTorch operator via `torch-npu`.


### Code conventions

Your contribution should pass all the coding conventions enforced by CI and pre-commit hooks.

Before your contribution, please run the following on your diff

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

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
