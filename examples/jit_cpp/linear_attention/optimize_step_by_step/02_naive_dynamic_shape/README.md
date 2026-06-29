# Step 02: Naive Dynamic Shape

This step is the beginner-friendly dynamic-shape version.

What it teaches:
- how `B` and `L` move from compile time to runtime
- why the launch `block_dim` becomes fixed to the device core count
- how the kernel loops internally over work items when `B * H` is larger than the number of cores
- how the dynamic kernel still stays close to the simple NumPy dataflow

Files:
- `linear_attention.cpp`: minimal dynamic PTO kernel
- `jit_util_linear_attention.py`: dynamic-shape JIT helper
- `run_linear_attention.py`: correctness sweep
- `benchmark_linear_attention.py`: early benchmark script
- `numpy_sim.py`: sequential NumPy emulation of dynamic work partitioning

Suggested run order:
```bash
python numpy_sim.py
python run_linear_attention.py
python benchmark_linear_attention.py --quick
```
