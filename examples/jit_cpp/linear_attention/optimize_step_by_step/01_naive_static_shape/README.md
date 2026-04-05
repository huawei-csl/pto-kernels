# Step 01: Naive Static Shape

This is the original fixed-shape starting point from commit `c226f0a`.

What it teaches:
- the smallest end-to-end PTO-ISA linear attention example
- how workspace buffers are laid out for one fixed `(B, H, L, D, C)` configuration
- why static-shape kernels are simple but inflexible

Files:
- `linear_attention.cpp`: fixed-shape PTO kernel
- `jit_util_linear_attention.py`: JIT compile/load helper
- `run_linear_attention.py`: correctness check
- `benchmark_linear_attention.py`: simple fixed-shape benchmark
- `numpy_sim.py`: sequential NumPy emulation of the same indexing and workspace logic

Suggested run order:
```bash
python numpy_sim.py
python run_linear_attention.py
python benchmark_linear_attention.py
```
