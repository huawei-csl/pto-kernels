# A5 Cube/Vector Sync Demo

Ports the old DAV_2201 raw-flag Cube/Vector demos to A5 / DAV_3510 and uses the
new direct Cube/Vector data paths:

- C2V: Cube `L0C -> Vec UB` through `copy_matrix_cc_to_ub` via A5 `TMOV`.
- V2C: Vec `UB -> Cube L1` through `copy_ubuf_to_cbuf` via A5 `TINSERT`.

The code is self-contained in this directory and launches through `torch_npu`
ctypes wrappers on the real device `npu:0`.

## Run

```bash
cd /home/jzhuang/pto-kernels-fork/examples/jit_cpp/cv_sync_demo_a5
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_npu_dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
NPU_DEVICE=npu:0 ./run_all.sh
```

Individual entry points:

```bash
python3 common_build.py
python3 run_stream.py
python3 run_matmul.py
```

## Validation

Before this port, the real-device environment was validated with:

```bash
cd /home/jzhuang/ptoisa-a5-test/tests/torch_sim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_npu_dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
NPU_DEVICE=npu:0 python3 -m common.build tadd
NPU_DEVICE=npu:0 python3 tadd/test_tadd.py
```

Result: both `tadd` real-device cases passed without `msprof`.

This directory was validated with `NPU_DEVICE=npu:0 ./run_all.sh` on the real
device. Results:

- `stream_c2v`: smoke passed.
- `stream_v2c`: smoke passed.
- `matmul_add_c2v`: `30/30` correctness cases passed.
- `add_matmul_v2c`: `30/30` correctness cases passed.

## Measured Bandwidth

Old baselines are from the DAV_2201 raw-flag READMEs. A5 results are from
`WARMUP=1 REPEATS=3 ./run_all.sh` on `npu:0`.

| Kernel | Old DAV_2201 peak | A5 measured peak | Ratio |
| --- | ---: | ---: | ---: |
| `stream_c2v` | 1154.2 GB/s | 9745.6 GB/s | 8.44x |
| `stream_v2c` | 1102.8 GB/s | 8256.5 GB/s counting actual A5 bytes (`UB -> L1` only); 16512.9 GB/s using the old DAV_2201 round-trip formula (`Vec -> workspace` + `workspace -> Cube`) | 7.49x by actual A5 bytes; 14.97x by old formula |
| `matmul_add_c2v` | 1401.3 GB/s | 2003.9 GB/s | 1.43x |
| `add_matmul_v2c` | 1593.8 GB/s | 273.3 GB/s | 0.17x |

Notes:

- `stream_c2v` uses float accumulator data in UB because A5 PTO does not support
  fp32-to-fp16 quantization in dual-destination accumulator-to-Vec mode. Its
  direct byte count equals the old fp16 GM round-trip byte count.
- For `stream_v2c`, the timed loop now measures the hot direct-copy path only:
  setup loads `A` and `D`, computes `A + D`, and converts to NZ once before the
  timed loop; each timed iteration performs `TINSERT`, which uses
  `copy_ubuf_to_cbuf` for `UB -> L1`. The old DAV_2201 benchmark had two GM
  transfers per iteration: `Vec -> workspace` and `workspace -> Cube`. The README
  shows both calculations so the actual A5 transfer rate is visible and the old
  benchmark's byte-counting formula can still be compared apples-to-apples.
- `matmul_add_c2v` uses float32 `D` and `C`, matching the native direct
  accumulator-to-Vec path.
- `add_matmul_v2c` launches once for the full batch. The host launcher expands
  `block_dim` to `physical_cube_cores * num_rounds`, so each logical wave/core
  pair performs one direct `UB -> L1` handoff without a Python per-wave loop.
  This avoids the Python overhead, but still reloads the weight tile per logical
  wave; attempts to reuse one physical-core persistent L1 handoff slot across
  waves were not correct on this software stack.

