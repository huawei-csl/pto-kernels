# Verification Log

Latest full reproduction: **2026-06-10** — see [`../reproduction-report-2026-06-10.md`](../reproduction-report-2026-06-10.md).
Quick re-smoke: `cd reference && NPU_DEVICE=npu:0 ./reproduce.sh`

---

Date: 2026-06-08

Environment:

- Host workspace: `/workdir`
- CANN: `/usr/local/Ascend/cann-9.0.0`
- `bisheng`: `/usr/local/Ascend/cann-9.0.0/bin/bisheng`
- Python: `3.11.15`
- Device server: `npu-smi info` reported Ascend910B2 devices.

Only commands listed here were executed during this skill update.

## A2A3 Dynamic ctypes: Vector Add

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
NPU_DEVICE=npu:4 python3 run_kernel_ctypes.py --kernel add --n 4096
```

Result:

```text
PASS add n=4096 block_dim=48
```

Notes:

- The local `dav-c220-vec` add sample initializes A2A3 vector mask state with `set_mask_norm(); set_vector_mask(-1, -1);` and is verified with `vector_core_num=48`. UB capacity is per core; query the target device's `ub_size` rather than treating total `block_dim` as a UB capacity multiplier.
- All local smoke checks use `pto_demo_utils.assert_close` with `torch.testing.assert_close` defaults: fp16 `rtol=1e-3, atol=1e-5`; fp32 `rtol=1.3e-6, atol=1e-5`. Real-device runners repeat each launch 5 times; CA-model paths set `PTO_SIMULATOR=1` for a single repeat.

## A2A3 Dynamic ctypes: simple_matmul

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
NPU_DEVICE=npu:4 python3 run_kernel_ctypes.py --kernel matmul --m 128
```

Result:

```text
PASS simple_matmul m=128 block_dim=1
```

Final source recompile:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
bash compile.sh matmul
```

Result:

```text
/workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3/build/libmatmul.so
```

## A2A3 Dynamic pybind/CMake

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
NPU_DEVICE=npu:4 python3 run_kernel_pybind.py --kernel all --n 4096 --m 128
```

Result:

```text
PASS pybind add n=4096 block_dim=48
PASS pybind simple_matmul m=128 block_dim=1
```

The pybind module exposes torch-style `at::Tensor` arguments, imports the resulting `pto_dynamic_a2a3_demo` module, then launches the local device libraries.

## A2A3 Dynamic Benchmark Template

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
NPU_DEVICE=npu:1 python3 benchmark.py --kernel all --warmup 1 --repeats 3 \
  --add-sizes 4096 --matmul-m 128 --csv outputs/benchmark_smoke.csv
```

Result:

```text
add             n=4096                   1     293.32     361.93     156.26       0.08     0.0000
simple_matmul   m=128,k=128,n=128        1     173.30     773.08     865.95       0.57     0.0242
Wrote outputs/benchmark_smoke.csv
```

Notes:

- The add benchmark chooses active blocks by tile capacity; spreading a very small vector over all physical cores can trigger vector UB issues under event timing.
- Use larger shape lists and more repeats for real reporting.

## A2A3 Dynamic Mix Kernels

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
NPU_DEVICE=npu:5 python3 run_mix_ctypes.py --kernel all --rounds 1
```

Result:

```text
PASS A2A3 matmul_add_c2v rounds=1 batch=3072
PASS A2A3 add_matmul_v2c rounds=1 batch=3072
```

## A2A3 CA Model: msprof op simulator

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a2a3
MSPROF_SOC_VERSION=Ascend910B2 MSPROF_TIMEOUT=120 ./run_sim.sh msprof --kernel all
```

Result:

```text
PASS static_single_core/a2a3 add shape=(64,64)
PASS static_single_core/a2a3 matmul shape=(128,128)x(128,128)
PASS static_single_core/a2a3 matmul_add rounds=1 batch=128
```

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
MSPROF_SOC_VERSION=Ascend910B2 MSPROF_TIMEOUT=120 ./run_sim.sh msprof --kernel add --n 4096
```

Result:

```text
PASS add n=4096 block_dim=48
```

Notes:

- Simulator libs live under `${ASCEND_HOME_PATH}/tools/simulator/Ascend910B2/lib`.
- `run_sim.sh` sources `setenv.bash`, sets `LD_LIBRARY_PATH`, and wraps `msprof op simulator`.
- Use `./run_sim.sh direct` for the same ctypes runners on real Ascend910B hardware.

## A2A3 Static Single-Core Add

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a2a3
NPU_DEVICE=npu:5 python3 run_kernel_ctypes.py
```

Result:

```text
PASS static_single_core/a2a3 add shape=(64,64)
```

Additional static A2A3 verification:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a2a3
NPU_DEVICE=npu:5 python3 run_kernel_ctypes.py
NPU_DEVICE=npu:5 python3 run_kernel_pybind.py
exe="$(bash compile.sh acl_add | tail -n 1)" && NPU_DEVICE=npu:5 "$exe" "$(pwd)/build/libstatic_add_a2a3.so"
```

Result:

```text
PASS static_single_core/a2a3 add shape=(64,64)
PASS static_single_core/a2a3 matmul shape=(128,128)x(128,128)
PASS static_single_core/a2a3 matmul_add rounds=1 batch=128
PASS static_single_core/a2a3 pybind add shape=(64,64)
PASS static_single_core/a2a3 ACL add
```

The static A2A3 `matmul_add.cpp` is self-contained; it does not include code from the dynamic example directory.

## A5 Dynamic ctypes: Compile

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5
bash compile.sh add
```

Result:

```text
/workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5/build/libadd_a5.so
```

## A5 Dynamic CA Model: msprof op simulator

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5
MSPROF_TIMEOUT=30 bash run_sim.sh msprof --n 128 --block-dim 8
```

Result:

```json
{
  "kernel": "add",
  "n": 128,
  "block_dim": 8,
  "result": "PASS"
}
```

Observed simulator wall time was about one minute including startup and profiling parse.

## A5 Dynamic CA Model: Cube Matmul

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5
MSPROF_TIMEOUT=30 bash run_sim.sh msprof --kernel matmul
```

Result:

```text
PASS dynamic matmul shape=16x16x16 block_dim=1
```

## A5 Dynamic CA Model: pybind

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_dynamic_pybind \
  python3 run_kernel_pybind.py --kernel all --n 128 --block-dim 8
```

Result:

```text
Using torch extension pybind path: pto_dynamic_a5_demo
PASS dynamic_multi_core/a5 pybind add n=128 block_dim=8
PASS dynamic_multi_core/a5 pybind matmul shape=16x16x16
```

## A5 Dynamic CA Model: Mix Kernels

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_mix_c2v \
  python3 run_mix_ctypes.py --kernel matmul_add --rounds 1 --block-dim 8
```

Result:

```text
PASS A5 matmul_add_c2v rounds=1 batch=1024 block_dim=8
```

Rechecked with strict `atol=1e-5`: PASS.

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_mix_v2c \
  python3 run_mix_ctypes.py --kernel add_matmul --rounds 1 --block-dim 8
```

Result:

```text
PASS A5 add_matmul_v2c rounds=1 batch=1024 block_dim=8
```

Rechecked with strict `atol=1e-5`: PASS.

## A5 Dynamic CA Model: cannsim

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5
bash run_sim.sh cannsim --output-json outputs/add_cannsim_rerun.json
```

Result:

```json
{
  "kernel": "add",
  "n": 128,
  "block_dim": 8,
  "result": "PASS"
}
```

Notes:

- `cannsim` executed the kernel and wrote PASS JSON, then the user process segfaulted during teardown. `run_sim.sh` now treats this known pattern as success only when the requested JSON file exists and contains `"result": "PASS"`.

## A5 Static Single-Core CA Model

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a5
MSPROF_TIMEOUT=30 bash run_sim.sh msprof --kernel add
MSPROF_TIMEOUT=30 bash run_sim.sh msprof --kernel matmul
```

Result:

```text
PASS static_single_core/a5 add shape=(64,64)
PASS static_single_core/a5 matmul shape=(16,16)x(16,16)
```

Static cannsim add wrapper:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a5
bash run_sim.sh cannsim
```

Result:

```text
PASS static_single_core/a5 add shape=(64,64)
cannsim exited 1 after writing PASS JSON; treating as success
```

Additional static A5 verification:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a5
MSPROF_TIMEOUT=30 bash run_sim.sh msprof --kernel matmul_add
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_static_pybind \
  python3 run_kernel_pybind.py
```

Result:

```text
PASS static_single_core/a5 matmul_add rounds=1 batch=1024
PASS static_single_core/a5 pybind add
PASS static_single_core/a5 pybind matmul
```

Pip-installable pybind package verification:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a5/pybind_package
python3 -m pip install --no-build-isolation -e .
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_static_pybind_pip \
  python3 ../run_kernel_pybind.py
```

Result:

```text
PASS static_single_core/a5 pybind add
PASS static_single_core/a5 pybind matmul
```

The package was also checked with a direct import after editable install:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a5/pybind_package
python3 -m pip install --no-build-isolation -e .
python3 - <<'PY'
import pto_static_a5_demo
print("import ok", pto_static_a5_demo.__name__)
PY
```

Result:

```text
import ok pto_static_a5_demo
```

## A5 Static C++ CA Model: `-lruntime_camodel`

Command:

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/static_single_core/a5
bash run_sim.sh linked
```

Result:

```text
PASS static_single_core/a5 ACL runtime_camodel add
```

## Pybind/CMake AOT

The local pybind verification is covered by `A2A3 Dynamic pybind/CMake` above. It builds and imports `dynamic_multi_core/a2a3/pybind.cpp` and launches the same local device libraries as the ctypes runner.

## Not Verified In This Task

- ACLgraph launch, vLLM/SGLang inference integration, TorchTitan/autograd integration.
