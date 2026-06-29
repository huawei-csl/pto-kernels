# PTO Kernel Launch Methods

## Quick Matrix

| Method | Best for | Build artifact | Runtime |
| --- | --- | --- | --- |
| ctypes/JIT | Fast kernel iteration | one `*.so` from `bisheng -shared` | Python + torch-npu stream and tensor pointers |
| pybind/CMake AOT | Formal deliverables | packaged Python extension | Python import with typed wrapper |
| ACL C++ | Host/runtime validation | executable + kernel `*.so` | ACL init, malloc, memcpy, stream |
| `msprof op simulator` | A2A3/A5 CA-model smoke + dumps | same torch/ctypes `*.so` | msprof injects CA model into Python |
| `cannsim record` | A5 SoC simulator trace | executable wrapper | `cannsim record -s Ascend950 ... -u "..."` |
| `-lruntime_camodel` | C++ CA-model samples | C++ executable linked to CA runtime | direct process execution on CPU model |

## ctypes/JIT

Use this by default while developing kernels.

1. Compile with `bisheng -fPIC -shared -xcce ... kernel.cpp -o build/libkernel.so`.
2. Load with `ctypes.CDLL`.
3. Set `argtypes` for `call_kernel`.
4. Pass `ctypes.c_void_p(tensor.data_ptr())` and `torch.npu.current_stream()._as_parameter_`.
5. Synchronize and compare with a CPU/PyTorch reference.

This is the pattern used by `dynamic_multi_core/*/run_kernel_ctypes.py`.

Cache the stream pointer before repeated launches. Do not call `torch.npu.current_stream()` inside an inner timing loop or per-token/per-tile launch helper.

## pybind/CMake AOT

Use pybind when the kernel is becoming a deliverable, not just a test. Expose torch C++ `at::Tensor` arguments, then obtain tensor data pointers inside the C++ binding. Do not expose Python-level `tensor.data_ptr()` integers as the public interface.

The local reference files include `dynamic_multi_core/a2a3/pybind.cpp`, `static_single_core/a2a3/pybind.cpp`, and `static_single_core/a5/pybind.cpp` as compact shapes of that workflow. The deliverable-style example is `static_single_core/a5/pybind_package`, which builds a pip-installable package linked to local kernel libraries. A pybind path is not accepted until the package/module builds, imports, launches a kernel, and verifies numerics on the target server.

When packaging with setuptools/torch extensions, use `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` so the generated module name matches names such as `pto_static_a5_demo._C`.

## ACL C++ Main

Use `main.cpp` when testing host-side ACL details: device selection, `aclrtMalloc`, host/device copies, stream creation, launch, and cleanup. This is also the natural shape for direct `-lruntime_camodel` linking.

## CA Model Modes

### `msprof op simulator`

A5 example:

```bash
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH:-}"
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=60 python3 run_kernel_ctypes.py ...
```

A2A3 example:

```bash
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/Ascend910B2/lib:${LD_LIBRARY_PATH:-}"
msprof op simulator --soc-version=Ascend910B2 --timeout=120 python3 run_kernel_ctypes.py ...
```

Local wrappers:

```bash
cd static_single_core/a2a3 && ./run_sim.sh msprof --kernel add
cd dynamic_multi_core/a2a3 && ./run_sim.sh msprof --kernel add --n 4096
```

This wraps the Python process, injects the simulator runtime, and lets torch-npu allocation/launch calls execute against the CPU CA model. Do not link the kernel `*.so` to `runtime_camodel` in this mode. Set `MSPROF_SOC_VERSION` when your install uses a different simulator package name.

#### Suggested `MSPROF_TIMEOUT` values

| Workload | SOC | Suggested timeout (seconds) |
| --- | --- | --- |
| Vector add smoke | A5 | 60 |
| Cube matmul smoke | A5 | 60 |
| Mix kernel (`rounds=1`) | A5 | 120–180 |
| Static all kernels | A2A3 | 120 |
| Dynamic add / matmul | A2A3 | 120 |

Override with `MSPROF_TIMEOUT=<seconds>` when using `./run_sim.sh`.

### `cannsim record`

Prefer the local wrappers; they handle the known teardown segfault when PASS JSON is written:

```bash
cd dynamic_multi_core/a5
./run_sim.sh cannsim --output-json outputs/add_cannsim.json
```

Raw invocation (user arguments go through `-u`, not trailing argv):

```bash
cannsim record -s Ascend950 -o outputs/cannsim \
  ./run_sim_entry.sh -u "--n 128 --block-dim 8 --output-json outputs/result.json"
```

`cannsim` is A5-focused and can generate `trace_core*.json` reports. The user process may segfault during teardown after the kernel writes valid PASS JSON; treat that as success only when the JSON file exists and contains `"result": "PASS"`.

### `-lruntime_camodel`

The C++ executable links against `runtime_camodel` and simulator libraries directly. The local portable example is `static_single_core/a5/main.cpp`:

```bash
cd static_single_core/a5
./run_sim.sh linked
```

Use this mode when future code samples already have an ACL C++ main and expect compile-time CA-model linkage.

## Host-Type Launch Rules

| Detected host | A2A3 kernels | A5 kernels |
| --- | --- | --- |
| Ascend910B (A2A3) | `direct`, ctypes, pybind, ACL | **`msprof` / `cannsim` / `linked` only** — do not use `./run_sim.sh direct` |
| Ascend950 (A5) | CA model or cross-compile only | `direct`, ctypes, pybind, ACL |
| CPU-only + CANN simulator | A2A3/A5 CA-model paths only | Same |

On a 910B host, launching A5 `dav-c310` kernels through `./run_sim.sh direct` or bare `run_kernel_ctypes.py` on a real NPU will fail with vector-core error `507035`. This is expected; use the CA-model paths above.

## Hang Detection

Use two timeout layers for real-device and direct runner invocations:

```bash
cd reference
PTO_PROCESS_TIMEOUT_S=60 ./run_with_timeout.sh python3 dynamic_multi_core/a2a3/run_kernel_ctypes.py --kernel add
```

| Layer | Tool | Default |
| --- | --- | --- |
| Whole process | `run_with_timeout.sh` / GNU `timeout` | 60 s device, 1800 s when `PTO_SIMULATOR=1` |
| Per `synchronize()` | `pto_demo_utils.synchronize_device()` | 60 s (`PTO_SYNC_TIMEOUT_S`) |

`./run_sim.sh direct`, `reproduce.sh`, and `run_sim_entry.sh` call `run_with_timeout.sh` automatically. `msprof op simulator --timeout=...` provides a separate whole-process bound for CA-model Python runs.

## TODO: ACLgraph

ACLgraph launch is intentionally not verified in this reference. Add only local templates after a PTO sample compiles and runs through ACLgraph.
