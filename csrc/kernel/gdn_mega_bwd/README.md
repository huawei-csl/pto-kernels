# Sequence-wide GDN backward

PTO-ISA implementation for packed B=1, D=C=128, FP16/BF16/FP32 inputs and
FP32 gradients. Dense matrix products execute on Cube. The C++ host binding
validates inputs, allocates six outputs and one workspace, and enqueues once
on the current Torch NPU stream. It performs no model tensor arithmetic or
device-to-host transfers.
Forward and backward are exported by the same compiled `pto_kernels` extension.
One immutable `KernelArgs` carries buffers and dimensions. Each stage receives
that block and a sequence-batch index, then reads a device metadata table.
The host selects head grouping, precision and stage count once; compute-stage
arguments contain no sequence arrays. Each core advances a sequence cursor as its
chunk tasks increase, visiting each boundary at most once per stage. Resolved
token bounds are reused across grouped heads; state stages index sequences directly.

## Interface and saved forward buffers

```python
from pto_kernels import pto_gdn_mega, pto_gdn_mega_bwd

# Model forward: create the cache once and retain these buffers for backward.
inverse_cache = torch.empty(num_chunks, H, 128, 128, device=q.device, dtype=torch.float16)
pto_gdn_mega(**forward_launch_args, inverse_cache=inverse_cache)
gate_cache = tuple(forward_launch_args[name] for name in ("g_t", "g_sum", "beta_t"))

# Model backward: no separate Torch preprocessing or forward invocation.
dq, dk, dv, dg, dbeta, dh0 = pto_gdn_mega_bwd(
    q, k, v, g, beta, dout, h0, dht, cu_seqlens,
    scale=scale, inverse_cache=inverse_cache, gate_cache=gate_cache,
)
```

`num_chunks = sum(ceil(sequence_length/128))`. Both caches are optional;
omitting them recomputes their quantities in PTO. The model autograd wrapper
remains separate work. It can use `ctx.save_for_backward` to retain inputs and
caches without copying. Buffers must belong to the same inputs and sequence
boundaries and stay unmodified until backward finishes. Backward treats them
as read-only. Missing initial state/final-state adjoint means zero.

Forward saves inverse factors `[D0,0;P,D1]` in FP32, FP16 or BF16, selected by
the caller's cache allocation. With FP32, existing vector/Cube exchanges are
redirected into the cache. With FP16/BF16, forward alternates between two
existing per-core FP32 scratch slots. After Cube finishes the factors, Vec
exports each completed stripe once while Cube computes the residual/output.
Only the selected cache is retained for backward. No extra allocations, matrix
products or kernel launches are added; forward arithmetic stays unchanged.

Backward expands smaller caches to FP32 while transposing into A.T, then fills
its upper-right block with one FP32 64x64 Cube product, `D0.T @ P.T`. This
reuses the expanded transpose without allocating a separate factor buffer.
It skips the triangular solve. The FP32 input accuracy path retains its
original completion/transposition schedule.

Existing `(g_t,g_sum,beta_t)` buffers have physical `[H,T]` layout: high/low
FP32 cumulative log gates and FP16 beta. Reusing them adds no forward work.
Exponentials and pairwise decay still execute in backward. KK/QK, W/U,
chunk-boundary states and corrected values Z are also recomputed. The original
h0 is reused. Q/K normalization and its rstd belong to the upstream operation.

At 8K/H=16, retained inverse factors cost **32 MiB** in FP16/BF16 or **64 MiB**
in FP32; gates/beta cost **1.25 MiB**.
The backward workspace still reserves **816.5 MiB**: twelve FP32 matrix slots and
seven vectors per chunk/head, plus 45 MiB of reusable block scratch. Workspace
exists only during backward. The current forward cache producer accepts FP16
q/k/v/beta and FP32 gates.

## Cache modes

There is one forward API and one backward API, each with optional cache buffers;
there is no public `cache=True/False` switch. Forward without `inverse_cache`
retains no inverse output. With it, the allocation dtype selects FP32/FP16/BF16.
Backward accepts `inverse_cache` and `gate_cache` independently and recomputes
what is absent. Native FP16/BF16 versus compensated FP32 device schedules are a
separate precision choice, not separate cached/recomputed APIs. Only one inverse
cache is retained; the temporary FP32 forward scratch is reused, not also saved.

## Device stages

| Stage | Work | Scheduling |
| --- | --- | --- |
| 0 | Pack inputs, gates, inverse, W/U; local value/state gradients | Independent chunk/key-head groups |
| 1 | Replay H/Z; reverse state adjoints, complete dU, write dh0 | Interleaved forward/reverse scans per sequence/head |
| 2 | Output/WY derivatives; fused final outputs on large native grids | Chunk/head groups when fused, otherwise individual heads |
| 3 (fallback) | Final gate scans, shared-head reductions and outputs | Small grids and FP32 inputs only |

Native-input grids use **three launches** when capacity covers at least 96 chunk/head-group
tasks; smaller grids preserve more parallelism with four launches.
FP32 inputs retain four. The fixed capacity selects the same schedule for every
batch of up to 64 packed sequences. No host chunk loops or intra-kernel global barriers.

NPU `cu_seqlens` adds one PTO metadata launch for the entire call: it reads
boundaries and computes chunk counts/cache offsets on-device in O(N) work.
Shape/dtype checks stay on the host; invalid device boundary values fail
asynchronously on-device. Packed CPU boundaries use one small PTO metadata-upload
launch per batch, with values embedded in kernel arguments. Omitted boundaries
(and a validated single CPU sequence) need no metadata table or metadata launch.

Graph replay can change NPU boundary **values** in-place with fixed tensor
shapes, addresses and cache capacities; matching caches must be updated too.
CPU boundaries are frozen at capture; replay does not retain or dereference
the original host boundary storage. Tensor addresses and the launch schedule remain fixed. To avoid readback, NPU workspace reserves
`min(T, floor((T + 127*min(N,64))/128))` chunk records per batch, capped by the
inverse cache capacity when supplied. This can exceed the actual chunk count;
single-sequence capacity is exact. Metadata adds 800 bytes per batch inside the
same workspace allocation. Backward accepts padded inverse caches and validates
that all active chunks fit on-device; forward must receive only the active view.

Fused tasks own lcm(8,Hv/Hqk) value heads when Hv is divisible by eight,
otherwise all heads. Q/K remain in Cube L1 across shared value heads. dQ/dK
sums stay in UB until the final store; dV writes directly to its output.
Aligned eight-head gate/beta transposes avoid partial-cache-line races.
State bases and elementwise/reduction chains also stay in UB, and operand
prefetches overlap earlier Cube products. There are no atomic reductions.

All schedules use explicit scratch addresses and PTO phase boundaries.
Interleaved scans pair forward Cube products with reverse vector updates and vice versa. Local
key-gradient updates and gate-energy reductions stay in UB. The original V
loads directly into UB at its consumers; no FP32 V copy is retained for inputs
with contiguous feature dimensions. Strided feature dimensions retain packing.

Full contiguous Q/K chunks use their original FP16/BF16 operands directly on
Cube for the shared Gram matrices, with FP32 accumulation. Other products keep
FP32 operands/accumulation. The final symmetric key product accumulates its two
terms in L0C, eliminating the vector transpose and its GM intermediate. FP32
inputs retain four K=32 partial products, local cancellation handling and
compensated gate scans. Their partial products use disjoint L0A/B operands and
two alternating L0C accumulators, so exports overlap the next product. The
compensated reduction can add directly into its consumer in UB, eliminating a
GM temporary. No HF32 shortcut is enabled.

Both input paths use blocked diagonal solves and Cube joins when the inverse
cache is absent. FP32 interval sums retain their original Kahan order, with 64
rows evaluated in parallel SIMD lanes on each vector subcore. Local score masks
and block weights use tiled DMA/transposes instead of scalar GM gathers.

## Validation and benchmarks

From the repository root in the configured CANN build environment:

```bash
export OMP_NUM_THREADS=4 NPU_DEVICE=npu:0
cmake --build build --target pto_kernels_ops -j 2        # Incremental build
python benchmarking/gdn_mega_bwd.py                     # Four checks + one timing
python benchmarking/gdn_mega_bwd.py --extensive --fla /path/to/gdn_ascend_benchmark.json
python benchmarking/gdn_mega_bwd.py --plot-only          # Redraw saved results
PYTHONPATH=python:.:tests python -m pytest -q tests/test_gdn_mega_bwd.py
```

The recursive Torch oracle, its CPU gradchecks and the NPU tests live together
in `tests/test_gdn_mega_bwd.py`. CPU-only checks need no compiled extension:
`python -m pytest --noconftest tests/test_gdn_mega_bwd.py -k reference -q`.

After a device edit, the same build command recompiles only affected objects
and relinks. Graph tests exercise changing boundaries, queries and upstream gradients,
FP16/FP32 inverse caches, empty sequences, more than 64 sequences, and CPU
boundary storage released after capture. They use `torch.npu.NPUGraph`, backed
by ACL graph capture/replay.

Both kernels are included in the normal CMake/pip installation;
imports and calls never invoke a compiler or require the source checkout.
For A5, configure a separate build directory with `-DSOC_VERSION=Ascend950`
(`dav-3510`). Forward and backward share the Cube/Vector synchronization helpers
in `kernel_utils.h`; inverse gathers use the target's byte/element index units.
A5 compilation is checked; numerical and performance validation currently use A2.

The extensive run saves `heatmap.png` (backward/forward ms and ratio),
`precision.png` (six gradient errors), `speedup.png` (FLA/PTO timings), and raw
`results.json` under `benchmark_data/gdn_mega_bwd/eval/`. Use `--output` to change
that directory. Inverse caching defaults to FP16; `--cache-dtype` selects the
format. `--cache-dtype none` disables both inverse and gate/beta reuse in backward
and inverse export in forward. To produce separate reports for all three modes:

```bash
for cache in none float32 float16; do
  python benchmarking/gdn_mega_bwd.py --extensive --cache-dtype "$cache" \
    --output "benchmark_data/gdn_mega_bwd/cache_comparison/$cache"
done
```

Each directory contains its own `heatmap.png`, `precision.png`, `speedup.png`
and raw timings. `--help` lists shape, profiling and optional FP64 audit options.

Each gradient must satisfy `max|kernel-golden| <=
2*max|reordered_Torch-golden| + 1e-5`. The independent recursive autograd oracle
uses FP32 for FP16/BF16 inputs and FP64 for FP32 inputs. Timing excludes
compilation/warmup: forward saves caches into preallocated buffers; backward
consumes them and includes allocation/dispatch. FLA comparisons match shapes
but differ in dtype, chunk size, inputs and allocation/autograd overhead.

Known limitation: the correlated-key case with 512 tokens, 64 value heads and
16 QK heads exceeds gradient bounds with BF16 inverse caching. Its FP16/FP32
cache variants pass. Pytest marks the BF16 case as a strict expected failure;
the evaluation report shows it as a failure and exits nonzero.

## Files

- [`torch_gdn_mega_bwd.h`](../../host/torch_gdn_mega_bwd.h): host validation, allocation, metadata and enqueue.
- [`pybind11.cpp`](../../host/pybind11.cpp): public `pto_gdn_mega_bwd` binding alongside forward.
- `kernel.cpp`, `layout.h`: explicit compensated stages, task scheduling, dispatch and buffer metadata.
- `pipeline.h`, `resident_cube.h`: native phase schedules and Cube operand reuse.
- `epilogues.h`: local reductions, shared-head sums and direct output stores.
- `device_ops.h`, `primitives.h`: shared synchronization/chunk views and PTO primitives; no operation dispatcher.
- `blocked_inverse.h`: forward-style diagonal solves and Cube block joins for recomputation.
