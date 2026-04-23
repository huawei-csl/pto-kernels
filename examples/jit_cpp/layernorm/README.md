# LayerNorm

JIT demo for the PTO LayerNorm kernel (`kernel_layernorm.cpp`).

The kernel performs per-row layer normalization over `fp16` inputs:

```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

It uses a two-pass anchored-stats algorithm for numerical stability:
1. **Stats pass** – anchor each row on its first element, accumulate
   centered sum and squared sum in chunks, then derive mean and `inv_std`.
2. **Output pass** – apply normalization and affine transform with
   double-buffered loads to overlap DMA with compute.

## Workflow

```bash
# Correctness tests
pytest test_layernorm.py -q --npu npu:0

# Benchmark PTO (fp16) vs F.layer_norm (fp16)
python bench_layernorm.py --npu npu:0

# Plot results
python plot_layernorm.py
```

Outputs:
- benchmark CSVs: `outputs/csv/`
- plots: `outputs/plots/`
- JIT shared objects: `outputs/so/`

Optional benchmark overrides:

```bash
python bench_layernorm.py --npu npu:0 --warmup 5 --repeats 50 --trials 3
python bench_layernorm.py --npu npu:0 --rows 64 128 --hidden-dims 4096 8192
python bench_layernorm.py --npu npu:0 --no-cache-stream
python plot_layernorm.py --csv-dir outputs/csv --plot-dir outputs/plots
```

Note: benchmark launches reuse a cached NPU stream pointer by default; use
`--no-cache-stream` to disable caching.

## Files

| File | Purpose |
|---|---|
| `kernel_layernorm.cpp` | NPU kernel source |
| `jit_util_layernorm.py` | JIT compile + Python wrapper |
| `test_layernorm.py` | Correctness tests vs. PyTorch reference |
| `bench_layernorm.py` | Benchmark PTO vs `F.layer_norm` |
| `plot_layernorm.py` | Plot duration, bandwidth, and speedup heatmap |
| `conftest.py` | pytest `--npu` fixture |

## Benchmark metrics

The benchmark reports per-shape:
- runtime in microseconds (median over trials)
- effective memory bandwidth in GB/s: `2 × rows × hidden × 2 bytes / duration`
  (counts fp16 read of x and fp16 write of y; gamma/beta are per-row amortized)
- PTO speedup over `F.layer_norm` (fp16 path)
