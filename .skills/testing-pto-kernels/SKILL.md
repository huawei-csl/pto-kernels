---
name: testing-pto-kernels
description: Compile, run, and verify PTO kernels on Ascend A2A3/910B and A5/950. Use when developing PTO-ISA kernels, testing jit_cpp examples, choosing bisheng flags, launching through ctypes/pybind/ACL, or running CA-model simulator smoke tests.
---

# PTO Kernel Testing

## Definition Of Done

For PTO kernel work, code is not done until all applicable checks are complete:

1. Compile the kernel with `bisheng` for the target architecture and core type.
2. Launch it through a real runtime path: ctypes/JIT, pybind/AOT, ACL C++, or CA-model simulator.
3. Compare against a PyTorch, NumPy, or golden-file reference.
4. Record exact commands and results in the relevant `README.md` or verification log.

Do not ask the user to compile or verify new kernel code manually as a substitute for running it yourself. If hardware, simulator libraries, or Python packages are missing, record the exact blocker.

## Reference Layout

Use `reference/` as the self-contained testing cookbook:

```text
reference/
  third_party/pto-isa/
  static_single_core/a2a3/
  static_single_core/a5/
  dynamic_multi_core/a2a3/
  dynamic_multi_core/a5/
  inference_integration/
  training_integration/
```

Keep the A2A3 and A5 demo directories structurally similar, but do not assume every directory supports every launch path. Real-device versus CA-model is a launch mode, not a directory name.

| Directory | Main purpose | Verified launch paths |
| --- | --- | --- |
| `static_single_core/a2a3` | Minimal fixed-shape A2A3 syntax | ctypes, pybind, ACL host, `msprof` CA model |
| `static_single_core/a5` | Minimal fixed-shape A5 + CA model | ctypes under `msprof`, pybind (+ pip package), `matmul_add` mix, linked `runtime_camodel`, `cannsim` |
| `dynamic_multi_core/a2a3` | Production-style A2A3 iteration | ctypes, pybind, mix, benchmark, `msprof` CA model |
| `dynamic_multi_core/a5` | A5 CA-model dynamic smoke | ctypes under `msprof`, pybind, `cannsim`, mix |

Read these first:

- `reference/README.md` for the map of verified examples.
- `reference/launch-methods.md` for ctypes, pybind, ACL C++, `msprof`, `cannsim`, and `-lruntime_camodel`.
- `reference/synchronization-and-architecture.md` for A2A3/A5 differences and mix-kernel sync pitfalls.
- `reference/verification-log.md` for commands that were actually run.

## Prerequisites

Before compiling or running examples:

```bash
source /usr/local/Ascend/cann-9.0.0/bin/setenv.bash
```

Confirm `bisheng`, `python3`, `torch_npu`, and either an idle NPU or the A5 simulator are available. Pick an idle device with `npu-smi info`; avoid running multiple kernel tests on the same `NPU_DEVICE` concurrently. If a kernel fault leaves an NPU in a persistent `507035`/vector-core error state, switch to a clean idle NPU or reset the device before trusting follow-up failures.

**`cannsim` teardown quirk:** On A5, `cannsim record` may segfault during process cleanup after the kernel has already written valid PASS JSON. Use `./run_sim.sh cannsim` (not a raw `cannsim` one-liner) unless you manually check the JSON output. A `[FAILED] cannsim record execution failed!` message with PASS JSON on disk is a known false alarm.

Run the bundled smoke script after skill changes:

```bash
cd reference && NPU_DEVICE=npu:0 ./reproduce.sh
```

## Static vs Dynamic Examples

`static_single_core/*` follows the PTO-ISA official unit-test style: one small kernel, fixed testcase shapes, direct `<<<1, nullptr, stream>>>` launch, and deterministic host/golden smoke. This style is good for learning instruction syntax and validating minimal ACL/C++ simulator linking, but each new shape usually requires recompiling or instantiating another kernel template.

`dynamic_multi_core/*` follows the local jit-style production-development demos: Python compiles and loads the kernel, chooses `block_dim`, allocates torch tensors, passes `torch.npu.current_stream()._as_parameter_`, and sweeps runtime shapes. Production kernels should support the dimensions that vary in real workloads to avoid recompiling for every input shape.

For LLM inference kernels, `batch` and `sequence` dimensions are usually dynamic because they change across requests and batches. `hidden_dim` or the innermost channel dimension is usually static because fixed tiling reduces scalar calculation overhead and keeps vector/cube tile shapes simple. `num_head` can be dynamic or static depending on the model family and deployment shape contract.

When reporting a delivered kernel, explicitly state which dimensions are dynamic and which are compile-time/static, for example: `B` and `S` dynamic, `H` static, `D=128` static.

## Architecture And Compiler Flags

Only A2A3 and A5 are in scope.

| Target | Device names | PTO dirs | Vector flag | Cube flag | Mix flag | Memory macro |
| --- | --- | --- | --- | --- | --- | --- |
| A2A3 / DAV_2201 | `Ascend910B*`, `Ascend910_93` | `a2a3` | `--cce-aicore-arch=dav-c220-vec` | `--cce-aicore-arch=dav-c220-cube` | `--cce-aicore-arch=dav-c220` | `-DMEMORY_BASE` |
| A5 / DAV_3510 | `Ascend950PR`, `Ascend950DT` | `a5` | `--cce-aicore-arch=dav-c310-vec` | `--cce-aicore-arch=dav-c310-cube` | `--cce-aicore-arch=dav-c310` | `-DREGISTER_BASE` |

Prefer runtime queries over hardcoded core counts or buffer sizes. Typical 910B2 has 24 Cube cores and 48 Vector cores; A5 variants differ by SKU.

## PTO-ISA Header Choice

Use CANN headers for stable APIs:

```bash
export PTO_LIB_PATH="${ASCEND_HOME_PATH}"
```

Use a pinned checkout or submodule when samples need new APIs (`TALLOC`, newer A5 helpers, etc.):

```bash
export PTO_LIB_PATH=/path/to/pto-isa
```

The actual include is usually `-I${PTO_LIB_PATH}/include`; if `pto/pto-inst.hpp` lives directly under the path, use `-I${PTO_LIB_PATH}`.

## Launch Methods

- ctypes/JIT: default for fast iteration. Build one `.so`, load with `ctypes.CDLL`, pass NPU tensor pointers and stream pointer.
- pybind/CMake AOT: preferred for formal deliverables and packaged ops. Follow the local pybind sample before claiming an AOT path works.
- ACL C++: use `main.cpp` when testing host-side ACL allocation, memcpy, stream, and `-lruntime_camodel` linking.
- CA model: use `msprof op simulator`, `cannsim record`, or direct `-lruntime_camodel` linking when the target device is not physically available, or when simulator/profiling traces are needed. If a real target NPU is available, prefer real-device execution for correctness and performance acceptance; use CA model as an additional debugging/profiling tool, not as a required substitute.

Launch syntax is `kernel<<<block_dim, nullptr, stream>>>(...)`. For persistent NPU kernels, set `block_dim` to the physical core count or a small bounded value, then loop over data inside the kernel. Do not scale `block_dim` with tensor size like CUDA grid launches.

UB capacity is a per-core hardware limit and is independent of total `block_dim`; query it for the current target instead of hardcoding it. On CANN installs with platform config files, use `grep -A 18 "AICoreSpec" "${ASCEND_HOME_PATH}/arm64-linux/data/platform_config/<soc>.ini"` and check `ub_size`; production code should prefer runtime platform APIs when available. For every A2A3 vector kernel compiled with `dav-c220-vec`, initialize vector mask state before vector instructions:

```cpp
set_mask_norm();
set_vector_mask(-1, -1);
```

This mask state affects vector instruction semantics broadly; missing it can surface as misleading runtime faults such as UB out-of-bounds, even when the real bug is not UB capacity or `block_idx` partitioning. Always document each kernel's verified `block_dim` policy in its example README or delivery report.

## Device Discovery

Use one or more:

```bash
npu-smi info
```

```python
import acl
print(acl.get_soc_name())
```

```python
import torch, torch_npu
props = torch.npu.get_device_properties(torch.npu.current_device())
print(props.name, props.cube_core_num, props.vector_core_num)
```

On some A5 systems `npu-smi` can fail while ACL and torch-npu still work; treat ACL/torch-npu success as stronger evidence.

Choose the launch method from the detected target:

| Host SOC | A2A3 kernels | A5 kernels |
| --- | --- | --- |
| Ascend910B (A2A3 hardware) | `direct` — ctypes, pybind, ACL on real NPU | **CA model only** — `msprof`, `cannsim`, or `-lruntime_camodel`. **Never** `./run_sim.sh direct` or bare ctypes on the 910B NPU (`507035` vector-core fault). |
| Ascend950 (A5 hardware) | CA model or cross-compile only | `direct` on real NPU; CA model for traces |
| No matching NPU | CA model for the target arch with tiny inputs | Same |
| CPU-only + CANN simulator | CA-model paths only | Same |

- Real A2A3 / Ascend910B available: compile with A2A3 flags and run on NPU through ctypes, pybind, or ACL C++.
- Real A5 / Ascend950 available: compile with A5 flags and run on the A5 NPU directly; use CA model only when detailed simulator traces are useful.
- No matching target NPU available: compile for the target architecture and use CA model (`msprof op simulator`, `cannsim`, or `-lruntime_camodel`) with tiny inputs.
- CPU-only host with CANN simulator installed: use only CA-model paths; do not claim real-device performance.

### CA-Model Timeouts

Set `MSPROF_TIMEOUT` when simulator runs are slow. Suggested values: A5 vector/matmul smoke 60 s; A5 mix kernels 120–180 s; A2A3 static/dynamic smoke 120 s. See `reference/launch-methods.md` for the full table.

## Correctness Rules

### Numeric Thresholds

Use the same relative/absolute tolerances as `torch.testing.assert_close` defaults. Reference runners call `pto_demo_utils.assert_close`, which applies:

| dtype | rtol | atol |
| --- | --- | --- |
| `float16` | `1e-3` | `1e-5` |
| `bfloat16` | `1.6e-2` | `1e-5` |
| `float32` | `1.3e-6` | `1e-5` |

Relax thresholds only when a kernel has a documented numerical reason (for example accumulation order or a known approximation). Relax gradually and record the exception in the deliverable README or verification log. Do not use `atol=1e-2`; loose absolute tolerance can mark incorrect kernels as PASS.

For outlier-heavy kernels, also report RMSE relative to output magnitude and R²; do not rely on max-error alone.

### Shape Coverage

Test edge shapes: one tile, two tiles, non-multiple tails, and at least one shape with more logical work than physical cores.

### Repeat Policy (Real Device vs Simulator)

| Mode | Repeats | Rationale |
| --- | --- | --- |
| Real NPU (`direct`, ctypes, pybind, ACL) | **5** (default) | Surfaces nondeterministic sync bugs such as missing `set_flag`/`wait_flag` pairs or `pipe_barrier` |
| CA model (`msprof`, `cannsim`, `-lruntime_camodel`) | **1** | Simulation is deterministic and slow |

`run_sim.sh` sets `PTO_SIMULATOR=1` so reference runners skip extra repeats under CA model. Override with `PTO_DEVICE_REPEATS=<n>` when needed.

### Hang Detection

Sync bugs can deadlock instead of producing a numeric mismatch. Use **two** timeout layers:

| Layer | What it guards | Default | Override |
| --- | --- | --- | --- |
| Whole process | compile, launch, driver/runtime stalls, hangs before `synchronize()` | **60 s** real device; **1800 s** simulator | `PTO_PROCESS_TIMEOUT_S` via `reference/run_with_timeout.sh` |
| Per-sync | `torch.npu.synchronize()` deadlocks inside a repeat loop | **60 s** | `PTO_SYNC_TIMEOUT_S` via `pto_demo_utils.synchronize_device()` |

A per-sync timeout alone is not sufficient: a bad kernel can hang during `bisheng` compile, ACL allocation, the launch call itself, or process teardown without ever reaching `synchronize()`. Wrap every real-device runner invocation with `run_with_timeout.sh` (or an equivalent whole-process `timeout`); `reproduce.sh` and `run_sim.sh direct` do this by default. CA-model paths launched through `msprof op simulator --timeout=...` already inherit a process-level bound from msprof. Raise `PTO_PROCESS_TIMEOUT_S` for large on-device benchmarks or long mix-kernel sweeps; the 60 s device default targets smoke-sized runs.

After a whole-process timeout, treat the NPU as suspect; switch `NPU_DEVICE` or reset the device before the next smoke.

## Performance Rules

- Time kernels with `torch.npu.Event(enable_timing=True)` and synchronize the end event.
- Cache `stream_ptr` outside inner timing loops and repeated launch helpers. Querying `torch.npu.current_stream()` / `.npu_stream` / `._as_parameter_` has visible runtime cost; get it once per benchmark shape or request path and pass the cached value into every launch.
- Include warmup and repeated trials; report median or distribution, not one sample.
- Flush or perturb cache when measuring HBM bandwidth-sensitive kernels.
- Compare against simple roofline expectations; results far above roofline usually mean the timer missed async work or cache reuse is being counted as bandwidth.
- Use `reference/dynamic_multi_core/a2a3/benchmark.py` as the concrete timing template: it shows warmup, repeats, `torch.npu.Event`, end-event synchronization, median/stdev reporting, optional cache flush, and CSV output.

## Current TODOs

- ACLgraph launch examples are not verified in this environment. Add a local demo only after it compiles and runs.
- End-to-end vLLM/SGLang/TorchTitan/autograd integration is intentionally left for a future environment with those packages installed.
