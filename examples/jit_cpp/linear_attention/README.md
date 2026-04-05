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

Throughput hunt:

```bash
python benchmark_linear_attention.py --throughput-hunt --warmup 2 --repeats 5
```

The example:
- compiles `linear_attention.cpp` with `bisheng`
- loads the generated `.so` via `ctypes`
- runs the kernel from PyTorch on NPU
- builds the triangular causal mask in PyTorch once and passes it as an extra kernel argument
- checks correctness against a PyTorch reference with `assert_close`
- reports effective TFLOP/s and GiB/s for larger benchmark shapes

Performance notes:
- the JIT compile path keeps the proven stack sizing, overflow recording, and scalar DCCI settings, but the local flag sweep showed this kernel is faster without `addr-transform` and without `L2_CACHE_HINT`
- the current minimum direct-`pto::` kernel applies the causal mask with a precomputed GM mask tensor plus UB vector multiply, avoiding the old scalar per-element masking loop
- the current minimum direct-`pto::` kernel now reuses one shared L0C accumulator region across the serialized cube stages, which allows a valid `C=128, D=128` configuration without changing the algorithm
- the cube matmul helper now uses a real L0 double-buffered `K=128 -> 2 x 64` ping-pong path, overlapping `TEXTRACT` with cube compute via `MTE1 <-> M` events
- the current bandwidth estimate excludes workspace traffic so the reported GiB/s reflects only the external tensors plus the causal mask, not the temporary per-core scratch buffers
- `C=128, D=128` is currently the best measured point on this machine and the true double-buffered cube path lifts throughput into the high-`50 TFLOP/s` range while keeping the same numerical result

Measured results:
- command: `python benchmark_linear_attention.py --warmup 2 --repeats 5`
- device-local results will vary, but the current measured table on this machine is:

| Shape `(B,H,L,D,C)` | Median ms | TFLOP/s | GiB/s |
| --- | ---: | ---: | ---: |
| `(16, 20, 1024, 128, 128)` | `0.848` | `50.64` | `368.52` |
| `(16, 20, 2048, 128, 128)` | `1.577` | `54.48` | `396.41` |
| `(32, 20, 1024, 128, 128)` | `1.491` | `57.59` | `419.07` |
| `(8, 20, 4096, 128, 128)` | `1.556` | `55.22` | `401.79` |

- for reference, the same kernel family at `C=64, D=128` currently measures roughly `28-31 TFLOP/s` on the same benchmark shapes
- best measured throughput in the default table: `57.59 TFLOP/s` and `419.07 GiB/s` at shape `(32, 20, 1024, 128, 128)`
- a larger-shape throughput hunt reached `58.48 TFLOP/s` and `425.53 GiB/s` at shape `(12, 20, 8192, 128, 128)` with the same double-buffered kernel