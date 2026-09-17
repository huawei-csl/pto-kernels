# Doubly-Stochastic Sinkhorn Normalization

PTO-ISA kernel for doubly-stochastic Sinkhorn normalization on Ascend NPU.
Implements the DeepSeek MHC pre-processing algorithm: softmax per row,
then alternating row/column normalization until the matrix is approximately
doubly-stochastic.

## Algorithm

```python
def sinkhorn_normalize(x, repeat=10, eps=1e-6):
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x
```

Input shape: `(N, K, K)` fp16 — N square matrices of dimension K.
K must be <= 128 (the full K×K matrix lives in UB as fp32).

## Files

| File | Description |
|---|---|
| `kernel_sinkhorn.cpp` | PTO-ISA C++ kernel (JIT-compiled via bisheng) |
| `jit_util_sinkhorn.py` | Python wrapper — compiles & exposes the kernel |
| `conftest.py` | pytest NPU device fixture |
| `test_sinkhorn.py` | Correctness tests vs PyTorch reference |
| `bench_sinkhorn.py` | Benchmark PTO vs PyTorch |
| `plot_sinkhorn.py` | Plot benchmark CSVs |

## Usage

```bash
# compile + test
python -m pytest test_sinkhorn.py -v --npu=npu:0

# benchmark
python bench_sinkhorn.py

# re-plot from saved CSV
python plot_sinkhorn.py
```

## Reproducing

```bash
cd examples/jit_cpp/sinkhorn

# tests (73 cases: shapes × repeats × seeds)
python -m pytest test_sinkhorn.py -v --npu=npu:0

# benchmark (batch × K grid, repeat=10, eps=1e-6)
python bench_sinkhorn.py --batches 1 4 8 16 32 64 --hidden-dims 4 8 16 32 64 128

# plots
python plot_sinkhorn.py
```
