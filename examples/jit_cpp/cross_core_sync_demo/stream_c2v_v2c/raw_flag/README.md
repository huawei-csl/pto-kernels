# stream_c2v_v2c — Cube↔Vector bandwidth microbenchmarks

Measures the **internal** throughput of the Cube↔Vector workspace handshake
(`ffts_cross_core_sync` / `wait_flag_dev`).  Unlike the `matmul_add` kernels,
which measure external HBM traffic (large input tensors), these kernels loop
over the **same small workspace** many times and count only the round-trip
workspace bytes.

## Kernels

| Kernel | Path | Cube work | Vec work |
|--------|------|-----------|----------|
| `stream_c2v` | Cube L0C → workspace → Vec UB | Initial GEMM fills L0C once; spills it every iter | Load workspace slice into UB |
| `stream_v2c` | Vec UB → workspace → Cube L1  | Load workspace into L1, discard | Load A+D, add, write to workspace |

Note: Removing the GEMM has negligible effect on throughput because the M pipe was never on the critical path.

**Effective bandwidth** (both kernels use the same formula):
```
bw_eff = 2 × num_cores × T² × sizeof(fp16) × num_iters / time
         ↑ workspace write + workspace read (round-trip)
```

## Files

| File | Purpose |
|------|---------|
| `stream_c2v.cpp` | C2V kernel — runtime `num_iters` arg |
| `stream_v2c.cpp` | V2C kernel — runtime `num_iters` arg, no GEMM on Cube side |
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
        32       53.75        936.4
        64       95.55       1053.6
       256      355.84       1131.6
      1024     1395.43       1154.2
Peak: 1154.2 GB/s

stream_v2c  (Vec UB → workspace → Cube L1)  [no Cube GEMM]
 num_iters      dur_us     bw_GB/s
        32       53.56        939.7
        64       94.63       1063.8
       128      182.56       1102.8
      1024     1467.76       1097.3
Peak: 1102.8 GB/s
```

C2V peak ~1154 GB/s (77% of HBM roofline); V2C peak ~1103 GB/s (74%).
V2C is slightly lower because Vec must also load A and D from HBM before it can
write to workspace — this external HBM traffic sits on the critical path even
though it is not counted in the bandwidth formula.
