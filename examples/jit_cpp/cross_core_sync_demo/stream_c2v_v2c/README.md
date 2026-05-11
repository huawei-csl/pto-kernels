# stream_c2v_v2c — Cube↔Vector bandwidth microbenchmarks

Measures the **internal** throughput of the Cube↔Vector workspace handshake
(`ffts_cross_core_sync` / `wait_flag_dev`).  Unlike the `matmul_add` kernels,
which measure external HBM traffic (large input tensors), these kernels loop
over the **same small workspace** many times and count only the round-trip
workspace bytes.

## Kernels

| Kernel | Path | Operation |
|--------|------|-----------|
| `stream_c2v` | Cube L0C → workspace → Vec UB | Cube spills GEMM result repeatedly |
| `stream_v2c` | Vec UB → workspace → Cube L1  | Vec writes A+D sum; Cube loads + GEMMs (result discarded) |

**Effective bandwidth** (both kernels use the same formula):
```
bw_eff = 2 × num_cores × T² × sizeof(fp16) × num_iters / time
         ↑ workspace write + workspace read (round-trip)
```

## Files

| File | Purpose |
|------|---------|
| `stream_c2v.cpp` | C2V kernel — runtime `num_iters` arg |
| `stream_v2c.cpp` | V2C kernel — runtime `num_iters` arg |
| `jit_util_stream.py` | JIT compile + ctypes loaders for both |
| `run_stream_c2v_v2c.py` | Smoke check + bandwidth sweep |

## Reproduce

```bash
cd /workdir/pto-kernels-fork/examples/jit_cpp/cross_core_sync_demo/stream_c2v_v2c

python run_stream_c2v_v2c.py

# Choose a different NPU (default: npu:7)
NPU_DEVICE=npu:5 python run_stream_c2v_v2c.py
```

## Expected output (910B2, 24 Cube cores)

```
stream_c2v  (Cube L0C → workspace → Vec UB)
 num_iters      dur_us     bw_GB/s
        32       53.38        942.8
        64       94.23       1068.3
       256      348.69       1154.8
      1024     1366.12       1179.0
Peak: 1179.0 GB/s

stream_v2c  (Vec UB → workspace → Cube L1)
 num_iters      dur_us     bw_GB/s
        32       53.51        940.6
        64       94.28       1067.7
       128      181.66       1108.3
      1024     1484.46       1085.0
Peak: 1108.3 GB/s
```

C2V peak ~1179 GB/s (79% of HBM roofline); V2C peak ~1108 GB/s (74%).
V2C is slightly lower because each iteration also runs a GEMM on the loaded data.
