# General knowledge about writing, compiling, and executing kernels on NPU


## Mandatory requirements for NPU kernel tasks

These rules apply whenever you (the agent) **develop, port, or optimize NPU kernels**. They are **not optional** guidance.

**Definition of done (all are required):**

1. **Compile** the kernel with `bisheng`, following the patterns in `examples/jit_cpp` in this repo.
2. **Execute** it on a real NPU via torch-npu (PyTorch with `device="npu"`).
3. **Verify** numerical correctness against a PyTorch or NumPy reference.

Until all three succeed, the task is **not finished**. Do not treat "code written" or "compiles only" as completion.

**You MUST:**

- Run the compile and NPU execution yourself and fix compile errors, runtime errors, and test failures by iterating until the kernel and its test scripts pass.
- Record the exact reproducing commands in that subdirectory’s `README.md` when the work is done so the user can re-run and confirm.

**You MUST NOT:**

- Ask the user to manually compile, run, or verify your new, still-untested code as a substitute for doing it yourself.

The environment is assumed capable of compiling and running on NPU; lack of access is not a reason to skip the steps above—surface the failure and what blocked you instead of delegating execution to the user.

---

## Highly recommended practices

> **Highly recommended — not mandatory:** The subsections below are **strong default guidance** for NPU kernels (resources, PTO-ISA layout, buffer limits, core topology, synchronization, performance, and timing). They are **not** part of the mandatory definition of done in **Mandatory requirements for NPU kernel tasks**; follow them when they apply unless you have a documented reason to diverge.

### Pick free NPUs for execution

`npu-smi info` prints NPU availability like:

```
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2               | OK            | 103.6       50                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3441 / 65536         |
+===========================+===============+====================================================+
...
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
...
```

Pick an NPU id with "No running processes", and avoid NPU id with other processes running on, to avoid resource contention. For example, to switch to NPU id 7, set `torch.npu.set_device("npu:7")` at the very beginning of the Python test script.


### Find pto-isa doc, implementation, and unit tests

The kernels should be implemented using APIs in "PTO-ISA" C++ library, just like other existing kernel samples under `examples/jit_cpp` or `csrc/kernel` of this repo.

The "PTO-ISA" library source code is usually located in `/workdir/pto-isa-master` or `/sources/pto-isa` path. Prompt the user to check if those directories do not exist in your environment. The most important subdirectories under `pto-isa` / `pto-isa-master` are:
- ISA documentation: `docs/isa`
- C++ header implementation: `include/pto/npu/a2a3`
- Unit tests: `tests/npu/a2a3/src/st/testcase` 

(the `a2a3` subdirectory name refers to current `910B` hardware; future `950` hardware uses `a5` subdirectory)


### Plan buffer space usage

`Tile` variables live in local SRAM buffer, with limited size. 

The hardware spec can be queried by command `grep -A 20 "AICoreSpec" ${ASCEND_HOME_PATH}/arm64-linux/data/platform_config/Ascend910B2.ini`, which gives:

```bash
[AICoreSpec]
cube_freq=1800
cube_m_size=16
cube_n_size=16
cube_k_size=16
vec_calc_size=128
l0_a_size=65536
l0_b_size=65536
l0_c_size=131072
l1_size=524288
fb0_size=2048
fb1_size=1024
fb2_size=2048
fb3_size=2048
bt_size=1024
smask_buffer=0
ub_size=196608
ubblock_size=32
ubbank_size=4096
ubbank_num=64
ubburst_in_one_block=32
```

The most important pieces of information are:
- ub_size=192 KiB, for `Tile<TileType::Vec, ...>`
- l1_size=512 KiB, for `Tile<TileType::Mat, ...>`
- l0_a_size=l0_b_size=64 KiB, for `TileLeft` and `TileRight`
- l0_c_size=128 KiB, for `TileAcc`

Make effective use of those SRAM buffers. Too little usage leads to low hardware utilization, while too much usage leads to overflow error.

### Number of Cube and Vector cores

The `910B2` hardware contains 24 "Cube cores" for matrix multiplications, and 48 "Vector cores" for all the rest of vector operations.

Confirm by command `grep -A 8 "SoCInfo" ${ASCEND_HOME_PATH}/arm64-linux/data/platform_config/Ascend910B2.ini`:

```
[SoCInfo]
ai_core_cnt=24
cube_core_cnt=24
vector_core_cnt=48
ai_cpu_cnt=6
memory_type=
memory_size=68719476736
l2_type=0
l2_size=201326592
```

For complex "mix" kernels that use both Cube cores and Vector cores, one cube core is coordinated with two vector cores. `get_block_idx()` gives the logical id of Cube cores, while Vector core id is usually given by `const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();`

### Synchronization for concurrent executions

Data movement instructions (e.g. `TLOAD`/`TSTORE`/`TMOV`) and compute instructions (e.g. `TADD`, `TMATMUL`) are asynchronous. To avoid data hazards during software pipelining, need `SetFlag` & `WaitFlag` instructions in between. Check existing kernel samples under `examples/jit_cpp` or `csrc/kernel` of this repo for typical synchronization patterns.

Insufficient synchronization can lead to **indeterministic bugs** that are hard to locate. Typical error patterns:
- Same kernel sometimes deadlocks, sometimes runs through
- Same kernel sometimes passes numerical check, sometimes not.
Those are due the asynchronous nature of the execution units in hardware.

Good practices:
- Always run the same verification scripts 3~5 times, not just one time.
- Be prepared that a test script might hang -- time-out until waiting for 20~30 seconds, to avoid the agent session being stucked forever.


### Performance optimization practices

- Avoid heavy use of scalar computations + scalar for loops, as they use the very slow "Scalar core" in NPU. Use SIMD instructions like `TLOAD`, `TADD`.
- General rule of thumb: Use wide SIMD length, and use "double buffers" (with two sync event ids) to overlap compute with data movement.
- Check against ideal roofline peak. For `910B2` device, the hardware roofline is about 1.5 TB/sec for global memory bandwidth, and ~300 TFLOP/s for matmul FLOPs.
    - A kernel with less than 10% of roofline is concerning: it might be bottlenecked by scalar cores, or uses wrong benchmark timer settings. 
    - A kernel that reaches much beyond roofline means not timing async kernel launch correctly, or has L2 cache reuse across iterations (if exceeds bandwidth peak but not FLOP peak).

### NPU benchmark timer settings and caveats

A typical timing code using `torch.npu.Event` (similar to `torch.cuda.Event`) looks like:

```python
    for _ in range(repeats):
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        # can optionally clean L2 cache here
        start.record()
        custom_kernel_launch()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))
```

In most cases `torch.npu.synchronize()` can be used for the `end.synchronize()` line. But triton kernel launches (sometimes needed for perf comparison) seem to not be synchronized with `torch.npu.synchronize()`, so here we use `end.synchronize()` instead.
