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
- builds the triangular causal mask in PyTorch once and passes it as an extra kernel argument
- checks correctness against a PyTorch reference with `assert_close`
- reports effective TFLOP/s and GiB/s for larger benchmark shapes

Performance notes:
- the JIT compile path uses the same key Bisheng/AICore codegen flags as the generated TileLang PTO build (`addr-transform`, stack sizing, overflow recording, scalar DCCI disable, `L2_CACHE_HINT`)
- the current minimum direct-`pto::` kernel applies the causal mask with a precomputed GM mask tensor plus UB vector multiply, avoiding the old scalar per-element masking loop
- the current minimum direct-`pto::` kernel is substantially faster than both the earlier wrapper-heavy version and the intermediate scalar-mask minimum version on the tested device while keeping the same numerical result
- larger compile-time tiles such as `C=128` or `D=256` currently exceed this minimum kernel's validated L0C budget on the tested device, so the practical high-throughput envelope remains centered on `C=64`, `D=128`

Measured results:
- command: `python benchmark_linear_attention.py --warmup 2 --repeats 5`
- device-local results will vary, but the current measured table on this machine is:

| Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | ---: | ---: | ---: |
| `(16, 20, 1024, 128, 64)` | `1.115` | `28.89` | `989.58` |
| `(16, 20, 2048, 128, 64)` | `2.163` | `29.78` | `1015.81` |
| `(32, 20, 1024, 128, 64)` | `2.110` | `30.53` | `1045.80` |
| `(8, 20, 4096, 128, 64)` | `2.193` | `29.38` | `999.89` |

- best measured throughput in the default table: `30.53 TFLOP/s` and `1045.80 GiB/s` at shape `(32, 20, 1024, 128, 64)`