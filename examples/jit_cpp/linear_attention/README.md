Standalone linear attention PTO-ISA example.

Run:

```bash
python run_linear_attention.py
```

The example:
- compiles `linear_attention.cpp` with `bisheng`
- loads the generated `.so` via `ctypes`
- runs the kernel from PyTorch on NPU
- checks correctness against a PyTorch reference with `assert_close`