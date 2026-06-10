# Static Single-Core A5

This directory implements the PTO-ISA ST style for A5 with fixed-shape vector add and cube matmul kernels. Run them through the CA model on servers without Ascend950 hardware.

## Local File Roles

| Role | Source |
| --- | --- |
| Vector `add.cpp` | Static vector add kernel, shape `(64,64)`. |
| Cube `matmul.cpp` | Static cube matmul kernel, shape `(16,16)x(16,16)`. |
| Mix `matmul_add.cpp` | Static C2V mix smoke, `C = A @ B + D`. |
| `main.cpp` | Self-contained ACL C++ host driver for the `-lruntime_camodel` path. |
| `pybind.cpp` | Minimal pybind launcher for static add and matmul; public C++ API takes `at::Tensor`. |
| `pybind_package/` | Minimal pip-installable pybind package for deliverable-style AOT. |

## Local Runnable CA-Model Smoke

```bash
./run_sim.sh msprof --kernel add
./run_sim.sh msprof --kernel matmul
./run_sim.sh msprof --kernel matmul_add
./run_sim.sh cannsim
./run_sim.sh linked
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_static_pybind \
  python3 run_kernel_pybind.py
cd pybind_package
python3 -m pip install --no-build-isolation -e .
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_static_pybind_pip \
  python3 ../run_kernel_pybind.py
```

`./run_sim.sh linked` builds `main.cpp`, links `-lruntime_camodel`, and runs the static add kernel directly against the CPU CA model.
