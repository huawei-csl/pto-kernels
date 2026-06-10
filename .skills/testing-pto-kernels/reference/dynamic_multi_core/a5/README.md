# Dynamic Multi-Core A5

This directory mirrors `dynamic_multi_core/a2a3` but targets A5/Ascend950. On Ascend910B hosts, A5 is verified through CPU CA-model launch modes only.

**Do not use `./run_sim.sh direct` or bare `run_kernel_ctypes.py` on a 910B NPU.** A5 `dav-c310` kernels will fault with `507035` on A2A3 hardware. Use `msprof`, `cannsim`, or test on real Ascend950 hardware.

## Files

| File | Role |
| --- | --- |
| `add.cpp` | Vector kernel: `y = x + z`, dynamic length, A5 `dav-c310-vec`. |
| `matmul.cpp` | Tiny dynamic cube matmul smoke, shape `(16,16)x(16,16)`. |
| `matmul_add.cpp`, `add_matmul.cpp` | Self-contained A5 mix kernels for direct C2V and V2C handoff. |
| `compile.sh` | Raw A5 `bisheng` command for `add`. |
| `run_kernel_ctypes.py` | ctypes runner with simulator-safe CPU-created inputs. |
| `run_mix_ctypes.py` | Reduced `rounds=1` CA-model smoke for A5 mix directions. |
| `run_sim.sh` | `msprof`, `cannsim`, or `direct` launch wrapper. |
| `pybind.cpp`, `run_kernel_pybind.py` | Tensor-style pybind launcher for add and tiny matmul under CA model. |

## A5 CA-Model Smoke

```bash
cd /workdir/pto-skills/testing-pto-kernels/reference/dynamic_multi_core/a5
./run_sim.sh msprof --n 128 --block-dim 8
```

Optional cannsim path:

```bash
./run_sim.sh cannsim --output-json outputs/add_cannsim.json
```

`cannsim` may report a user-process segmentation fault during teardown after the kernel has written valid PASS JSON. `run_sim.sh` accepts that case only when the requested JSON exists and contains `"result": "PASS"`.

Use tiny shapes first. CA-model startup dominates wall time.

Pybind CA-model smoke:

```bash
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_pybind \
  python3 run_kernel_pybind.py --kernel all --n 128 --block-dim 8
```

## A5 Mix Notes

Run one mix direction at a time under CA model:

```bash
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_mix_c2v \
  python3 run_mix_ctypes.py --kernel matmul_add --rounds 1 --block-dim 8
```

```bash
msprof op simulator --soc-version=Ascend950PR_9599 --timeout=30 \
  --output=outputs/msprof_mix_v2c \
  python3 run_mix_ctypes.py --kernel add_matmul --rounds 1 --block-dim 8
```

CA-model mix kernels are much slower than vector kernels. Keep `rounds=1` unless you explicitly need a larger simulation.
