# PTO Kernel Testing Reference

This directory is the reproducible companion to `../SKILL.md`. Future agents should be able to test kernels from here without reading the raw material repositories.

## Reproduction Checklist

Source CANN, pick an idle NPU, then run the path that matches your host.

```bash
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
npu-smi info   # pick an idle NPU_DEVICE
export NPU_DEVICE=npu:0
```

### Ascend910B host (A2A3 hardware + A5 simulator)

```bash
cd reference

# Full sequential smoke (recommended after skill changes)
./reproduce.sh

# Or run individually:
cd dynamic_multi_core/a2a3
NPU_DEVICE=npu:0 python3 run_kernel_ctypes.py --kernel all --n 4096 --m 128
NPU_DEVICE=npu:0 python3 run_mix_ctypes.py --kernel all --rounds 1

cd ../../static_single_core/a2a3
MSPROF_SOC_VERSION=Ascend910B2 MSPROF_TIMEOUT=120 ./run_sim.sh msprof --kernel all

cd ../../dynamic_multi_core/a5
MSPROF_TIMEOUT=60 ./run_sim.sh msprof --n 128 --block-dim 8
./run_sim.sh cannsim --output-json outputs/add_cannsim.json

cd ../../static_single_core/a5
MSPROF_TIMEOUT=60 ./run_sim.sh msprof --kernel all
./run_sim.sh linked
```

Do **not** run A5 `./run_sim.sh direct` on a 910B host. A5 kernels require CA-model launch there.

### Ascend950 host (A5 hardware)

```bash
cd reference/dynamic_multi_core/a5
NPU_DEVICE=npu:0 python3 run_kernel_ctypes.py --n 128 --block-dim 8
```

Use `msprof` or `cannsim` when simulator traces are needed.

### CPU-only host (CANN simulator installed)

Use only `./run_sim.sh msprof`, `./run_sim.sh cannsim`, or `./run_sim.sh linked` paths. Do not claim real-device performance.

Record commands and results in `verification-log.md` or a dated report under the skill root.

Real-device runners should be launched through `run_with_timeout.sh` (default **60 s** on NPU, **1800 s** under `PTO_SIMULATOR=1`; override with `PTO_PROCESS_TIMEOUT_S`) in addition to the in-process `PTO_SYNC_TIMEOUT_S` guard around `torch.npu.synchronize()`.

## Directory Map

```text
reference/
  pto_demo_utils.py                 # shared helpers: assert_close thresholds, device repeats, sync timeout
  reproduce.sh                      # sequential smoke script
  run_with_timeout.sh               # whole-process timeout wrapper for runners
  third_party/pto-isa/              # how to find or pin PTO-ISA headers
  static_single_core/a2a3/          # PTO-ISA ST-style A2A3 samples
  static_single_core/a5/            # PTO-ISA ST-style A5 samples
  dynamic_multi_core/a2a3/          # jit_cpp-style A2A3 samples
  dynamic_multi_core/a5/            # jit_cpp-style A5 samples
  inference_integration/            # TODO stub
  training_integration/             # TODO stub
  launch-methods.md
  synchronization-and-architecture.md
  verification-log.md
```

Generated `build/` and `outputs/` directories are ephemeral; see `../.gitignore`.

## Which Example To Use

| Goal | Start here | Why |
| --- | --- | --- |
| Learn minimal PTO instruction/test anatomy | `static_single_core/a2a3` or `static_single_core/a5` | Fixed-shape `tadd`-style kernels are easiest to read. |
| Iterate on real A2A3 hardware | `dynamic_multi_core/a2a3` | ctypes/JIT launch on torch-npu is the fastest loop. |
| Test A2A3 kernels without a free NPU | `static_single_core/a2a3` or `dynamic_multi_core/a2a3` | `./run_sim.sh msprof` with `Ascend910B2` simulator. |
| Test A5 behavior on this A2A3 host | `dynamic_multi_core/a5` | Uses A5 flags plus CPU CA-model launch modes. |
| Compare launch methods | `launch-methods.md` | ctypes, pybind, ACL C++, `msprof`, `cannsim`, and `-lruntime_camodel`. |
| Work on Cube/Vector sync | `synchronization-and-architecture.md` | A2A3 raw FFTS and A5 direct `TMOV`/`TINSERT` patterns. |

## Quick Decision Flow

1. Learning PTO syntax? Start with `static_single_core/a2a3`.
2. Iterating on an A2A3 production-style kernel? Use `dynamic_multi_core/a2a3`.
3. Preparing a deliverable Python package? Use pybind/CMake, especially the pip package under `static_single_core/a5/pybind_package`.
4. Need A2A3 CA-model smoke without tying up a real NPU? Use `static_single_core/a2a3` or `dynamic_multi_core/a2a3` with `./run_sim.sh msprof`.
5. Need A5 behavior without A5 hardware? Use `dynamic_multi_core/a5` or `static_single_core/a5` with `msprof op simulator`.
6. Debugging Cube/Vector handoff? Read `synchronization-and-architecture.md`, then run the mix examples one direction at a time.

## Verification Policy

Only commands that were actually executed are listed in `verification-log.md`. A command shown in a demo `README.md` is a runnable target; if it was not runnable in this environment, the blocker must be recorded.

Keep simulator inputs tiny. A5 CA-model runs are much slower than real NPU runs; start with vector smoke shapes such as SiLU `T=128`, SwiGLU `batch=1,input_n=256`, stream `num_iters=4`, and mix `rounds=1`.

Run examples sequentially on a chosen `NPU_DEVICE`. Parallel tests on the same NPU can leave the device in a persistent fault state after a bad kernel launch.
