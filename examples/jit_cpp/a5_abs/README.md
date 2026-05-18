# a5_abs — JIT `vabs_fp16` with torch_npu + msprof CA simulator

Refactors the [a5 `main_abs`](../../a5/main_abs.cpp) flow: JIT-compile [`kernel_abs.cpp`](../../../csrc/kernel/kernel_abs.cpp) with `bisheng`, launch `call_vabs_fp16` via **ctypes** on **torch_npu** tensors, and profile under the Ascend950 CA model with **msprof op simulator**.

Numeric correctness is **not** validated in `run_abs.py` when using the CA simulator — `msprof op simulator` models hardware pipeline behavior only. A successful run compiles the kernel, launches it on NPU tensors, and produces profiler output under `OPPROF_*`.

## Layout

| File | Role |
|------|------|
| `jit_util_a5_abs.py` | JIT compile `kernel_abs.cpp` → `libkernel_abs_jit.so`, ctypes wrapper |
| `run_abs.py` | Build input, launch kernel, synchronize |
| `run_msprof.sh` | Wrapper: env + `msprof op simulator` |

## Prerequisites

- Docker image `agent_npu_cann_950:9.0.0` (or equivalent CANN 9.0 + torch_npu 2.9)
- Host checkout of `pto-kernels` mounted into the container

## Reproduce (CA simulator)

From the host, start the container (mount the repo parent so `pto-kernels` is visible):

```bash
cd /path/to/parent-of-pto-kernels
docker run -it --rm \
  --privileged \
  --network=host \
  --ipc=host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  --name torch_npu \
  agent_npu_cann_950:9.0.0 \
  /bin/bash
```

Inside the container:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/pto-kernels/examples/jit_cpp/a5_abs

./run_msprof.sh
```

Or manually (same as [`ca_model.md`](../../../../npu_kernels/950_setup/ca_model.md)):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
ulimit -n 65535

# msprof splits on spaces; use a script file, not python -c.
msprof op simulator --soc-version=Ascend950PR_9599 \
  python ./run_abs.py
```

### Expected success signals

1. Console: `generated .../libkernel_abs_jit.so`, then `vabs_fp16 kernel launch completed.`
2. msprof log: `Profiling on kernel: vabs_fp16_mix_aic` (or similar) and core duration table
3. New directory `OPPROF_<timestamp>_*` in this folder with parsed simulator results

### Compile flags (Ascend950 / A5)

JIT build uses Ascend950-oriented flags (aligned with tilelang-ascend PTO `A5` path):

- `--cce-aicore-arch=dav-c310`
- `-DREGISTER_BASE`
- `-std=gnu++17` (required for A5 PTO headers; `-std=c++20` fails on dav-c310)
- AICore stack LLVM options (`-cce-aicore-stack-size`, etc.)

Do not use `--npu-arch=dav-2201` here; that targets an older arch profile.

## Run without msprof (real device)

On hardware with a working NPU runtime, you can smoke-test launch only:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/pto-kernels/examples/jit_cpp/a5_abs
python ./run_abs.py
```

For numeric checks on device, use the packaged op (`tests/test_abs.py` / `pto_abs`) or the legacy ACL sample:

```bash
cd /workspace/pto-kernels
make run_abs_a5   # cannsim record + examples/a5/main_abs.cpp
```

## Relation to legacy a5 sample

| Legacy (`make run_abs_a5`) | This example |
|----------------------------|--------------|
| `g++` + `libkernel_abs.so` + ACL host buffers | `bisheng` JIT + torch tensors + ctypes |
| `cannsim record --soc=Ascend950` | `msprof op simulator --soc-version=Ascend950PR_9599` |
| `examples/a5/main_abs.cpp` | `run_abs.py` + `jit_util_a5_abs.py` |

Kernel source is shared: `csrc/kernel/kernel_abs.cpp` (`call_vabs_fp16`, shape `8×128`, `blockDim=8`).
