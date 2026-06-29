# Step 03: Precompute And Cache Causal Mask

This step corresponds to commit `a9b54ed`. It also inherits the earlier cleanup/minimalization that made the kernel shorter and easier to follow.

What changed:
- the triangular causal mask is built once in PyTorch and passed into the kernel
- the kernel applies the mask with vector tile operations instead of a slow scalar loop

Suggested run order:
```bash
python run_linear_attention.py
python benchmark_linear_attention.py --quick
```
