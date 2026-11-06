# Sequence-wide GDN backward

PTO-ISA implementation for packed B=1, D=C=128, FP16/BF16/FP32 inputs and
FP32 gradients. Dense matrix products execute on Cube. Python validates,
allocates six outputs and one workspace, then enqueues once; the native
launcher handles metadata and stream integration. Neither performs tensor math.

## Interface and saved forward buffers

```python
# Model forward: create the cache once and retain these buffers for backward.
inverse_cache = torch.empty(num_chunks, H, 128, 128, device=q.device, dtype=torch.bfloat16)
pto_gdn_mega(**forward_launch_args, inverse_cache=inverse_cache)
gate_cache = tuple(forward_launch_args[name] for name in ("g_t", "g_sum", "beta_t"))

# Model backward: no separate Torch preprocessing or forward invocation.
dq, dk, dv, dg, dbeta, dh0 = backward(
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
Only the 16-bit cache is retained for backward. No extra allocations, matrix
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

## Cache modes and reproducible reports

There is one forward API and one backward API, each with optional cache buffers;
there is no public `cache=True/False` switch. Forward without `inverse_cache`
retains no inverse output. With it, the allocation dtype selects FP32/FP16/BF16.
Backward accepts `inverse_cache` and `gate_cache` independently and recomputes
what is absent. Native FP16/BF16 versus compensated FP32 device schedules are a
separate precision choice, not separate cached/recomputed APIs. Only one inverse
cache is retained; the temporary FP32 forward scratch is reused, not also saved.

[Report commands, exact timing contracts and numerical test coverage](../../../benchmarking/gdn_mega_report.md).
The unified command creates cached backward/forward timing heatmaps, exact-shape
comparisons against saved FLA times, and a small per-gradient max-error heatmap.
It defaults to BF16 inverse factors and has `--quick`, `--only` and `--plot-only`
modes. No production kernel or tolerance changes are needed for these reports.

## Device stages

| Stage | Work | Scheduling |
| --- | --- | --- |
| 0 | Pack inputs, gates, inverse, W/U; local value/state gradients | Independent chunk/key-head groups |
| 1 | Replay H/Z; reverse state adjoints, complete dU, write dh0 | Interleaved forward/reverse scans per sequence/head |
| 2 | Output/WY derivatives; fused final outputs on large native grids | Chunk/head groups when fused, otherwise individual heads |
| 3 (fallback) | Final gate scans, shared-head reductions and outputs | Small grids and FP32 inputs only |

Native-input grids use **three launches** when at least 96 chunk/head-group
tasks exist; smaller grids preserve more parallelism with four launches.
FP32 inputs retain four. This decision applies separately to each batch of up
to 64 packed sequences. No host chunk loops or intra-kernel global barriers.

Fused tasks own lcm(8,Hv/Hqk) value heads when Hv is divisible by eight,
otherwise all heads. Q/K remain in Cube L1 across shared value heads. dQ/dK
sums stay in UB until the final store; dV writes directly to its output.
Aligned eight-head gate/beta transposes avoid partial-cache-line races.
State bases and elementwise/reduction chains also stay in UB, and operand
prefetches overlap earlier Cube products. There are no atomic reductions.

All schedules use explicit scratch addresses and PTO phase boundaries. The
`Ops` expression interface and temporary allocator have been deleted.
Interleaved scans
pair forward Cube products with reverse vector updates and vice versa. Local
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

## Current improvement (v10)

Pre-change snapshot: `2c181b2`. All remaining `Ops` calls are removed. The new
speedups below apply to the compensated FP32-input path; cached FP16 throughput
is unchanged within 0.5%. Timings alternate the snapshot/current libraries in
one process, with two warmup pairs and seven measured pairs.

| Input / cache | Tokens | Hv / Hqk | v9 ms | v10 ms | Speedup |
| --- | --- | --- | --- | --- | --- |
| float32 / none | 256 | 8 / 2 | 1.546 | 1.048 | 1.48x |
| float32 / none | 513 | 8 / 2 | 2.707 | 1.716 | 1.58x |
| float32 / none | 2048 | 16 / 2 | 12.552 | 6.869 | 1.83x |
| float16 / BF16 + gates | 2048 | 16 / 2 | 1.447 | 1.454 | 1.00x |
| float16 / BF16 + gates | 8192 | 64 / 16 | 13.610 | 13.592 | 1.00x |

All 152 backward/reference tests passed with unchanged bounds. The saved-buffer
test group then passed all 75 cases in 55.74s using session-local oracle reuse:
30 reference calls instead of 150, with all NPU executions/assertions retained.
No production preprocessing, forward/cache format or workspace sizing changed.
The previous 2x native-training-path speedup target remains unmet.

[Raw samples, commands and source hashes](../../../tests/baselines/gdn_mega_bwd_no_ops_v10.json).

## Previous cached improvement (v9)

Safe pre-change snapshot: `e82451a`. Paired measurements use the same BF16
inverse and gate/beta caches from forward, alternating implementations in one
process (3 warmup pairs, 11 measured pairs). Times include allocation/enqueue
and synchronization, excluding compilation and forward.

| Tokens | Hv / Hqk | Snapshot ms | Current ms | Speedup |
| --- | --- | --- | --- | --- |
| 2048 | 16 / 2 | 1.673 | 1.466 | 1.14x |
| 8192 | 16 / 2 | 5.171 | 4.273 | 1.21x |
| 8192 | 64 / 16 | 16.945 | 13.811 | 1.23x |
| 8192 | 48 / 16 | 13.049 | 10.752 | 1.21x |
| 8192 | 32 / 16 | 9.504 | 7.724 | 1.23x |
| 8192 | 32 / 8 | 9.420 | 7.610 | 1.24x |
| 32768 | 64 / 16 | 64.456 | 52.946 | 1.22x |

The requested **2x speedup is not achieved**; this version measures 1.14–1.24x.
At 8K/Hv64/Hqk16, warmed device stage times change from 5.137 / 3.910 / 7.129 ms
to 4.277 / 3.072 / 5.951 ms (preparation / scans / derivatives). These are kernel
durations, not utilization counters. All 152 backward/reference tests pass with
unchanged bounds. Existing heatmaps retain their historical measurements.

[Raw samples, experiments and source hashes](../../../tests/baselines/gdn_mega_bwd_explicit_v9.json).
Reproduce from the repository root after sourcing CANN:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:.:tests python benchmarking/gdn_mega_bwd_compare.py build/gdn_mega_bwd/4ef9e56e099811b5/kernel.so
```

The baseline device library must share the current Plan/workspace ABI.

## Historical cached improvement (v8)

| Tokens | Local Hv / Hqk | Cached v7 (ms) | Cached v8 (ms) | Reduction |
| --- | --- | --- | --- | --- |
| 8192 | 64 / 16 | 20.732 | 16.960 | 18.2% |
| 16384 | 32 / 8 | 21.526 | 17.577 | 18.3% |
| 16384 | 48 / 16 | 30.830 | 24.675 | 20.0% |
| 8192 | 16 / 2 | 6.048 | 5.116 | 15.4% |
| 2048 | 16 / 2 | 1.889 | 1.660 | 12.1% |

Identical inputs and both forward caches; alternating v7/v8 in one process,
three warmup pairs and eleven measured pairs. Synchronized wall time includes
backward allocation/enqueue and excludes forward execution. All 18 paired
cases improve by **11–20%**. At 8K/Hv=16/Hqk=2, a warmed hardware trace totals
**4.780 ms** for the three kernels; allocation/enqueue are excluded there.
End-to-end latency is still slightly above 5 ms, and the general 2x-forward
target remains unmet.

At 8K/Hv=64/Hqk=16, output+assembly traffic falls 47% for Cube operand loads,
44% for vector reads and 61% for vector writes. These hardware interface
counters include L2 hits, not just HBM transfers. Forward code, cache formats,
workspace size and numerical bounds are unchanged.

[Raw samples, counters and source hashes](../../../tests/baselines/gdn_mega_bwd_cached_v8.json).
[Development history and previous baselines](../../../memory_bwd.md).

## Validation and benchmarks

**152 tests passed**: 142 hardware and ten CPU reference checks. Each gradient
is compared with an independent recursive Torch/autograd golden run (FP32 for
FP16/BF16 inputs, FP64 for FP32). A reordered recurrence in the input dtype
sets a separate scalar error budget for each tensor:
`max|kernel - golden| <= 2 * max|torch_baseline - golden| + 1e-5`.
Tests also cover finite differences, cache/input integrity, packed tails,
empty sequences, shared heads, streams, cancellation and an allocation-only
Torch dispatch guard. Cache tests run the actual forward to produce caches.
FP16/BF16 caches must exactly equal rounding the FP32 cache, and cached versus
uncached forward outputs/final states must be bitwise identical. The numerical
storage experiment can be reproduced with `tests/check_gdn_inverse_precision.py
--output build/gdn_mega_bwd/inverse_precision.json`; add `--roundtrip` to isolate
rounding using test-only Torch casts instead of native 16-bit storage.
[Initial storage experiment](../../../tests/baselines/gdn_mega_inverse_precision.json)
and [completed-factor export timings](../../../tests/baselines/gdn_mega_inverse_export.json):
FP16/BF16 halve the retained inverse bytes. Exporting completed factors once
removes the earlier 6–8.5% forward penalty; in the measured shapes, 16-bit
cache-saving forward is within about 1% of FP32 caching in the tested shapes.
The host chooses a compiled cache strategy; each call still launches once.

After sourcing CANN, from the repository root:

```bash
OMP_NUM_THREADS=4 NPU_DEVICE=npu:0 PYTHONPATH=python:. python -m pytest tests/test_gdn_mega_bwd.py tests/test_gdn_mega_bwd_reference.py -q
OMP_NUM_THREADS=4 PYTHONPATH=python:. python tests/benchmark_gdn_mega_bwd.py --tokens 2048 8192 --heads 16 --key-heads 2 --inverse-cache --gate-cache
OMP_NUM_THREADS=4 PYTHONPATH=python:. python tests/benchmark_gdn_mega_bwd.py --extensive --repeats 5
```

Add `--cache-dtype bfloat16` or `--cache-dtype float16` to benchmark smaller
caches (also supported by `--extensive`). The default stays FP32 so historical
benchmark commands remain comparable. This changes cache storage only; the
forward still requires FP16 q/k/v/beta, and all inverse arithmetic stays FP32.

`--extensive` always consumes both forward caches. Every heatmap cell shows
cached backward / cache-saving forward wall time and their ratio. Backward
includes allocation; forward uses preallocated buffers. Compilation, warmup
and TP communication are excluded. Global Hv={64,48,32,16}, Hqk=16,
TP={1,2,4,8}, L={256,1024,4096,16384,32768}; both head counts divide by TP.
Sixty configurations are measured. Twenty are N/A because the forward cache
producer does not support local Hv=2,4,6; no uncached timings are substituted.

Outputs: `benchmark_data/gdn_mega_bwd/cached/{results.jsonl,heatmap.png}`;
override with `--output DIRECTORY`. Matplotlib is required. Historical
`extensive/` and `tp_v7/` plots measure recomputation; `cached_v7/` retains the
previous cached plot. The ordinary benchmark without cache flags recomputes.
Use `--profile DIRECTORY` for a hardware trace, or `--profile-once` with msprof.

## Fast development loop

The device and native launcher use separate content-addressed build caches.
A device edit compiles only `kernel.cpp`; it does not rebuild the forward or the
repository extension. Use three smoke checks while iterating, then the complete
suite once the device changes stabilize:

```bash
# Inside the existing CANN container, from the repository root:
export OMP_NUM_THREADS=4 NPU_DEVICE=npu:0 PYTHONPATH=python:.:tests
python -m pytest -q -x \
  'tests/test_gdn_mega_bwd.py::test_backward[dtype2-lengths0-2-1-both]' \
  'tests/test_gdn_mega_bwd.py::test_backward[dtype2-lengths6-8-2-zero_beta]' \
  'tests/test_gdn_mega_bwd.py::test_backward[dtype2-lengths17-8-2-replace]'
python benchmarking/gdn_mega_bwd_compare.py build/gdn_mega_bwd/45d6c0fa80997f25/kernel.so --quick
python benchmarking/gdn_mega_bwd_compare.py build/gdn_mega_bwd/45d6c0fa80997f25/kernel.so --quick --dtype float32
```

`--quick` uses two representative shapes, one warmup pair and three measured
pairs. `--shape TOKENS HV HK` selects a case and can be repeated; `--warmup` and
`--repeats` override the sample counts. FP16 compares cached backward using the
same saved BF16 inverse and gates; FP32 compares uncached backward. Short runs
are screening measurements; use more samples for the final record. No extensive
sweep or profiler is needed for every edit. The supplied baseline library is the
validated `2c181b2` snapshot and must share the current Plan/workspace ABI.

## FLA comparison heatmap

`benchmark_gdn_mega_bwd_fla.py` benchmarks the exact shapes in a supplied FLA
JSON, then plots separate forward/backward speedups. It joins on batch,
sequence length, both head counts and both head dimensions. Each cell shows
`FLA median / PTO median` (>1 means PTO is faster), followed by both times.
It uses PTO's cache-saving forward and backward consuming both caches.

In the existing container, from the repository root:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:. python tests/benchmark_gdn_mega_bwd_fla.py --fla /mounted_home/benchmarks/fla-gdn-9e82a80/sweep/gdn_ascend_benchmark.json
# Add --plot-only to redraw existing results without running the NPU.
```

The supplied FLA sweep has Q/K heads equal to value heads (16/24/32/48/64),
so it needs its own matched-shape PTO sweep; the existing TP chart remains
unchanged. Outputs go to `benchmark_data/gdn_mega_bwd/fla_comparison/`:
`speedup_heatmap.png`, `comparison.json`, raw PTO timings and logs.

The chart explicitly labels **PTO FP16/C=128 versus FLA BF16/C=64**. These
are reported latency ratios, not an identical-dtype controlled comparison.
Input distributions, allocation/autograd overhead and state outputs also
differ; `comparison.json` preserves the source metadata and timing definitions.
No forward+backward ratio is inferred from sums: FLA measured that mode jointly.

## Files

- `backward.py`, `binding.py`, `launch.cpp`: validation, allocation, cached builds and enqueue.
- `kernel.cpp`, `layout.h`: explicit compensated stages, task scheduling, dispatch and buffer metadata.
- `pipeline.h`, `resident_cube.h`: native phase schedules and Cube operand reuse.
- `epilogues.h`: local reductions, shared-head sums and direct output stores.
- `device_ops.h`, `primitives.h`: shared synchronization/chunk views and PTO primitives; no operation dispatcher.
- `blocked_inverse.h`: forward-style diagonal solves and Cube block joins for recomputation.
