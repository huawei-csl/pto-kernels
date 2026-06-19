# Depthwise causal conv1d + bias + SiLU

JIT demo for the PTO fused **depthwise causal conv1d** (width `K=4`) with
per-channel bias and SiLU activation — the Mamba / Gated-DeltaNet (GDN) short
convolution.

For each channel `c` and position `i`:

```
y[b, i, c] = silu( bias[c] + sum_{k=0..K-1} W[k, c] * x[b, i-K+1+k, c] ),   x[<0] = 0
```

- Per-channel weights `W[K, W]` and `bias[W]` are fp32 GM tensors.
- I/O is fp16 **or** bf16; accumulation is fp32.
- Entries: `fn(x, w, bias)` for a single `[L, W]` sequence, and
  `fn.batched(x, w, bias, activation=...)` for `[batch, L, W]`.

## Files

- `conv1d_dw_pto.cpp` — the PTO kernel.
- `jit_util_conv1d_dw.py` — JIT compile + ctypes bindings.
- `test_conv1d_dw.py` — correctness tests vs an fp32 reference.
- `run_conv1d_dw.py` — validate + benchmark against `aclnnConvolution`.
- `sweep_conv1d_dw.py` — performance sweep over shapes.

## Suggested workflow

```bash
pytest test_conv1d_dw.py -q --npu npu:0
python run_conv1d_dw.py
python sweep_conv1d_dw.py
```

JIT shared objects are written under `outputs/so/` (git-ignored).
