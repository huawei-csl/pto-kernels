# Linear Attention Step-By-Step Optimization

This folder turns the historical optimization trail of the `jit_cpp/linear_attention` example into a runnable tutorial ladder.

Each numbered directory contains a runnable snapshot of one major optimization step, copied onto the current `linear_attn` branch for teaching purposes.

## Learning Path

Suggested reading order:
1. `01_naive_static_shape`
2. `02_naive_dynamic_shape`
3. `03_cached_mask`
4. `04_chunk128`
5. `05_l0_double_buffer`
6. `06_two_slot_cv_pipeline`
7. `07_l1_prefetching`

## What Each Step Teaches
- `01_naive_static_shape`: the smallest fixed-shape PTO-ISA kernel; easiest place to understand workspace layout and tensor indexing
- `02_naive_dynamic_shape`: move `B` and `L` to runtime, keep launch shape fixed to the number of cores, and loop over work items inside the kernel
- `03_cached_mask`: precompute the triangular mask in PyTorch and apply it with vector tile ops instead of scalar loops
- `04_chunk128`: raise chunk size from `64` to `128` to increase arithmetic intensity
- `05_l0_double_buffer`: split `K=128` into `2 x 64` cube phases and overlap extract with compute
- `06_two_slot_cv_pipeline`: let cube prepare chunk `i + 1` while vector finishes chunk `i`
- `07_l1_prefetching`: keep two `H`-state L1 tiles so the next hidden-state chunk is loaded early

## Quick Validated Progression

These are the short smoke-test numbers produced while verifying the intermediate tutorial samples on this machine. They use the small `--quick` benchmark shapes so the whole ladder can be checked end to end in a reasonable amount of time. The first two rows below are from the newly simplified beginner kernels.

| Step | Quick validated result |
| --- | --- |
| `01_naive_static_shape` | fixed-shape smoke benchmark: `(2, 2, 512, 128, 64)` in `0.275 ms` |
| `02_naive_dynamic_shape` | `5.21 TFLOP/s` at `(16, 20, 1024, 128, 64)` |
| `03_cached_mask` | `30.02 TFLOP/s` at `(16, 20, 1024, 128, 64)` |
| `04_chunk128` | `49.73 TFLOP/s` at `(16, 20, 1024, 128, 128)` |
| `05_l0_double_buffer` | `52.57 TFLOP/s` at `(16, 20, 1024, 128, 128)` |
| `06_two_slot_cv_pipeline` | `63.15 TFLOP/s` at `(16, 20, 1024, 128, 128)` |

## Final Step Full Benchmark

The final tutorial step is intentionally identical to the current main example kernel and JIT helper. It should therefore be compared with the same full benchmark command, not with `--quick`.

| Target | Command | Best validated result |
| --- | --- | --- |
| `07_l1_prefetching` | `python benchmark_linear_attention.py --warmup 2 --repeats 5` | `77.71 TFLOP/s` / `565.43 GiB/s` at `(12, 20, 8192, 128, 128)` |
| main example | `python benchmark_linear_attention.py --warmup 2 --repeats 5` | `77.97 TFLOP/s` / `567.34 GiB/s` at `(24, 20, 6144, 128, 128)` |

Those two results were measured sequentially with the same command. The small gap is normal benchmark noise; the important point is that the final tutorial step reaches the same `~78 TFLOP/s` performance class as the current main example.

## How To Read The Early Steps
- `01` and `02` include `numpy_sim.py`, which intentionally hides the real flag/synchronization details and replaces parallel core execution with a sequential loop.
- Those NumPy simulations focus on the dataflow: which tiles are loaded, how workspace is updated, and how chunked causal masking interacts with the running hidden state.

## Notes
- `01` and `02` also include NumPy simulations that explain the tensor indexing and workspace layout without the real PTO synchronization details.
- The early steps were rewritten into smaller teaching kernels so they stay close to the NumPy emulation and avoid distracting helper boilerplate.
- JIT outputs are redirected into a local `compiled_lib/` subdirectory so the tutorial folders stay tidy.
- The later steps intentionally keep the code close to the optimized working snapshots, while the step README files explain the key optimization idea.
- Benchmark large-shape steps one process at a time. Running multiple heavy NPU benchmarks concurrently can lower measured TFLOP/s and make the step-to-step comparison misleading.

## Suggested Validation Order

Inside each step directory:

```bash
python run_linear_attention.py
python benchmark_linear_attention.py --quick --warmup 1 --repeats 2
```

For the final step, run the full table instead of `--quick`:

```bash
python benchmark_linear_attention.py --warmup 2 --repeats 5
```

For `01` and `02`, also run:

```bash
python numpy_sim.py
```
