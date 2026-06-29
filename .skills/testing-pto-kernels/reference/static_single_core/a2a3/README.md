# Static Single-Core A2A3

This directory implements the PTO-ISA ST style for A2A3: fixed-shape kernels, direct `<<<1, nullptr, stream>>>` launch, a small ACL host driver, ctypes, and pybind launchers.

## Local File Roles

| Role | Source |
| --- | --- |
| Vector `add.cpp` | Static vector add kernel, shape `(64,64)`. |
| Cube `matmul.cpp` | Static cube instruction checklist. |
| Mix `matmul_add.cpp` | Static mix-kernel checklist. |
| `main.cpp` | Minimal ACL host driver using `NPU_DEVICE`. |
| `pybind.cpp` | Minimal pybind launcher for static add; public C++ API takes `at::Tensor`. |

The local files keep the template roles and do not require another PTO-ISA test checkout.

## Local Runnable Path

Real-device smoke (`run_kernel_ctypes.py` defaults to `--kernel all`: add, matmul, matmul_add):

```bash
NPU_DEVICE=npu:0 python3 run_kernel_ctypes.py
NPU_DEVICE=npu:0 python3 run_kernel_ctypes.py --kernel add
NPU_DEVICE=npu:0 python3 run_kernel_pybind.py
exe="$(bash compile.sh acl_add | tail -n 1)" && NPU_DEVICE=npu:0 "$exe" "$(pwd)/build/libstatic_add_a2a3.so"
```

A2A3 CA-model smoke via `msprof` (no physical NPU required for correctness):

```bash
./run_sim.sh msprof --kernel add
./run_sim.sh msprof --kernel all
```

Override the simulator SOC when needed:

```bash
MSPROF_SOC_VERSION=Ascend910B2 MSPROF_TIMEOUT=120 ./run_sim.sh msprof --kernel add
```

`./run_sim.sh direct` runs `run_kernel_ctypes.py` on a real Ascend910B device.
