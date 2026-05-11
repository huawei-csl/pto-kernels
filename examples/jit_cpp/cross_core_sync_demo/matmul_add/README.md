# matmul_add — Cube-to-Vector matmul+add (persistent kernel)

Computes `C = A @ B + D` (fp16) using `ffts_cross_core_sync` / `wait_flag_dev`
for Cube→Vec synchronization.  `block_dim = num_cube_cores` (persistent style).

## Files

| File | Purpose |
|------|---------|
| `matmul_add_c2v.cpp` | PTO-ISA kernel (Cube GEMM + Vec add, C2V stream) |
| `jit_util_matmul_add_c2v.py` | JIT compile + ctypes loader |
| `run_matmul_add_c2v.py` | Correctness tests + bandwidth benchmark |

## Reproduce

```bash
cd /workdir/pto-kernels-fork/examples/jit_cpp/cross_core_sync_demo/matmul_add

# Run correctness tests (30 cases) and bandwidth benchmark
python run_matmul_add_c2v.py

# Choose a different NPU (default: npu:7)
NPU_DEVICE=npu:5 python run_matmul_add_c2v.py
```

## Expected output (910B2, 24 Cube cores)

```
Correctness: 30/30 passed — OK

     batch  rounds      dur_us     bw_GB/s
------------------------------------------
      3072       1       54.10       44.21
      6144       2       52.90       89.82
     12288       4       52.39      180.77
     24576       8       52.27      361.71
     49152      16       52.07      725.64
     98304      32       57.89     1304.78
    196608      64      107.78     1401.29

Peak bandwidth: 1401.3 GB/s  (910B2 HBM roofline ≈ 1500 GB/s)
```
