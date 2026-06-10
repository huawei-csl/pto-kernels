# Dynamic Multi-Core A2A3

This directory is the real-device A2A3 jit-style demo. It keeps the file roles used across the reference tree while using small smoke shapes.

## Files

| File | Role |
| --- | --- |
| `add.cpp` | Vector kernel: `y = x + z`, dynamic length, persistent `block_dim`. |
| `matmul.cpp` | Cube kernel role, operation name `simple_matmul`: `C[M,128] = A[M,128] @ B[128,128]`. |
| `matmul_add.cpp`, `add_matmul.cpp` | Self-contained A2A3 mix kernels for C2V and V2C handoff. |
| `compile.sh` | Raw `bisheng` command for `add` or `matmul`. |
| `run_kernel_ctypes.py` | Main verified quick-test path. |
| `pybind.cpp`, `run_kernel_pybind.py` | Local pybind launcher alternative to `run_kernel_ctypes.py`; public C++ API takes `at::Tensor`. |
| `run_mix_ctypes.py` | Reduced `rounds=1` smoke for both mix directions. |
| `benchmark.py` | Concrete timing example using NPU events, warmup, repeats, median, optional cache flush, and CSV output. |
| `run_sim.sh` | `msprof` or `direct` wrapper for A2A3 CA-model / real-device ctypes smoke. |

## Real-Device Smoke

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
NPU_DEVICE=npu:0 python3 run_kernel_ctypes.py --kernel all --n 4096 --m 128
```

## A2A3 CA-Model Smoke

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a2a3
./run_sim.sh msprof --kernel add --n 4096
./run_sim.sh msprof --kernel all --n 4096 --m 128
```

Default simulator SOC is `Ascend910B2`. Override with `MSPROF_SOC_VERSION` when your CANN install uses a different A2A3 simulator package.

`./run_sim.sh direct` runs the same ctypes path on a real Ascend910B device.

Expected output:

```text
PASS add n=4096 block_dim=48
PASS simple_matmul m=128 ...
```

## Pybind AOT Smoke

This directory includes `pybind.cpp` as a local importable launcher module that calls the same device libraries as the ctypes path.

```bash
NPU_DEVICE=npu:0 python3 run_kernel_pybind.py --kernel all --n 4096 --m 128
```

Use pybind for deliverables; use ctypes for fast local kernel iteration.

## Benchmark Smoke

```bash
NPU_DEVICE=npu:0 python3 benchmark.py --kernel all --warmup 2 --repeats 5 \
  --add-sizes 4096 65536 --matmul-m 128 1024 \
  --csv outputs/benchmark_smoke.csv
```

Use `--flush-cache` for bandwidth-sensitive measurements where L2 reuse would otherwise dominate.

## Block Dim Policy

- `add.cpp` is a pure vector kernel compiled with `dav-c220-vec`. It initializes A2A3 vector mask state with `set_mask_norm(); set_vector_mask(-1, -1);`, uses only `block_idx`, and is verified with `block_dim=vector_core_num=48` on 910B2.
- `matmul.cpp` is a cube kernel and uses `block_dim=min(cube_core_num, M / 128)`.
- Mix kernels use `get_subblockid()` for Cube/Vector pairing and use Cube-oriented `block_dim`.

## Mix Kernels

Run the local mix smoke:

```bash
NPU_DEVICE=npu:0 python3 run_mix_ctypes.py --kernel all --rounds 1
```

For first smoke or CA-model attempts, reduce to one seed and `rounds=1`.
