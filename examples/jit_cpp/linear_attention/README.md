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

Performance notes:
- the JIT compile path uses the same key Bisheng/AICore codegen flags as the generated TileLang PTO build (`addr-transform`, stack sizing, overflow recording, scalar DCCI disable, `L2_CACHE_HINT`)
- the current minimum direct-`pto::` kernel is substantially faster than the earlier wrapper-heavy version on the tested device while keeping the same numerical result
- larger compile-time tiles such as `C=128` or `D=256` currently exceed this minimum kernel's validated L0C budget on the tested device, so the practical high-throughput envelope remains centered on `C=64`, `D=128`

Measured results:
- command: `python benchmark_linear_attention.py --warmup 2 --repeats 5`
- device-local results will vary, but the current measured table on this machine is:

| Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | ---: | ---: | ---: |
| `(16, 20, 1024, 128, 64)` | `6.151` | `5.24` | `179.39` |
| `(16, 20, 2048, 128, 64)` | `12.216` | `5.27` | `179.86` |
| `(32, 20, 1024, 128, 64)` | `11.801` | `5.46` | `187.02` |
| `(8, 20, 4096, 128, 64)` | `12.236` | `5.27` | `179.18` |

- best measured throughput in the default table: `5.46 TFLOP/s` and `187.02 GiB/s` at shape `(32, 20, 1024, 128, 64)`