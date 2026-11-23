# Step 03a: Fast On-The-Fly Mask Construction

This step keeps the dynamic-shape interface from step 02, but removes the
slow scalar `SetValue` / `GetValue` causal-mask loop.

What changed:
- the lower-triangular mask is synthesized on-chip with PTO-ISA vector ops
- the mask is built with the higher-level `TTRI` PTO-ISA wrapper instead of raw per-element scalar updates
- the kernel then reuses the same fast `TMUL` masking pattern as step 03
- unlike step 03, this path does not read a precomputed mask from global memory

Suggested run order:
```bash
python run_linear_attention.py
python benchmark_linear_attention.py --quick
```