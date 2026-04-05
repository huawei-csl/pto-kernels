Standalone linear attention PTO-ISA example.

Run:

```bash
python run_linear_attention.py
```

Benchmark:

```bash
python benchmark_linear_attention.py
```

Quick smoke benchmark:

```bash
python benchmark_linear_attention.py --quick --warmup 1 --repeats 3
```

The example:
- compiles `linear_attention.cpp` with `bisheng`
- loads the generated `.so` via `ctypes`
- runs the kernel from PyTorch on NPU
- checks correctness against a PyTorch reference with `assert_close`
- reports effective TFLOP/s and GiB/s for larger benchmark shapes