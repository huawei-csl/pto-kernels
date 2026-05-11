# matmul_add — C2V and V2C persistent kernels (ffts_cross_core_sync)

Two complementary kernels demonstrating Cube↔Vec handshake via
`ffts_cross_core_sync` / `wait_flag_dev`.  Both use `block_dim = num_cube_cores`
(persistent style).

## Kernels

| Kernel | Operation | Stream direction |
|--------|-----------|-----------------|
| `matmul_add_c2v` | `C = A @ B + D` | Cube GEMM → workspace → Vec add |
| `add_matmul_v2c` | `C = (A + B) @ D` | Vec add → workspace → Cube GEMM |

## Files

| File | Purpose |
|------|---------|
| `matmul_add_c2v.cpp` | C2V kernel source |
| `jit_util_matmul_add_c2v.py` | JIT compile + ctypes loader (C2V) |
| `run_matmul_add_c2v.py` | Correctness tests + benchmark (C2V) |
| `add_matmul_v2c.cpp` | V2C kernel source |
| `jit_util_add_matmul_v2c.py` | JIT compile + ctypes loader (V2C) |
| `run_add_matmul_v2c.py` | Correctness tests + benchmark (V2C) |

## Reproduce

```bash
cd examples/jit_cpp/cross_core_sync_demo/matmul_add

# C2V: C = A @ B + D
python run_matmul_add_c2v.py

# V2C: C = (A + B) @ D
python run_add_matmul_v2c.py

# Choose a different NPU (default: npu:7)
NPU_DEVICE=npu:5 python run_matmul_add_c2v.py
NPU_DEVICE=npu:5 python run_add_matmul_v2c.py
```

## Expected output (910B2, 24 Cube cores)

**C2V** (`matmul_add_c2v`):
```
Correctness: 30/30 passed — OK
Peak bandwidth: 1401.3 GB/s
```

**V2C** (`add_matmul_v2c`):
```
Correctness: 30/30 passed — OK
Peak bandwidth: 1593.8 GB/s
```
