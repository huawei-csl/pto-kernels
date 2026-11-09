# Sequence-wide GDN backward

PTO-ISA implementation for packed B=1, D=C=128, FP16/BF16/FP32 inputs and
FP32 gradients. Dense matrix products execute on Cube. Python validates,
allocates six outputs and one workspace, then enqueues once; the native
launcher handles metadata and stream integration. Neither performs tensor math.

## Interface and saved forward buffers

```python
# Model forward: create the cache once and retain these buffers for backward.
inverse_cache = torch.empty(num_chunks, H, 128, 128, device=q.device, dtype=torch.float16)
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
It defaults to FP16 inverse factors; the same runner provides a fast development
check, `--extensive` evaluation and `--plot-only` rendering. No production kernel or tolerance changes are needed for these reports.

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

## Validation and benchmarks

The development command runs four small checks with the independent recursive
Torch/autograd oracle, then times one cached forward/backward shape:

```bash
python benchmarking/gdn_mega_bwd.py
python benchmarking/gdn_mega_bwd.py --extensive --fla /path/to/gdn_ascend_benchmark.json
```

See [evaluation commands and outputs](../../../benchmarking/gdn_mega_report.md)
for the complete pytest suite, numerical bounds, TP/FLA comparisons, profiling
and replotting. The extensive runner writes exactly three PNGs plus raw JSON.

Known limitation: the correlated-key case with 512 tokens, 64 value heads and
16 QK heads exceeds gradient bounds with BF16 inverse caching. Its FP16/FP32
cache variants pass. Pytest marks the BF16 case as a strict expected failure;
the evaluation report shows it as a failure and exits nonzero.

## Files

- `backward.py`, `binding.py`, `launch.cpp`: validation, allocation, cached builds and enqueue.
- `kernel.cpp`, `layout.h`: explicit compensated stages, task scheduling, dispatch and buffer metadata.
- `pipeline.h`, `resident_cube.h`: native phase schedules and Cube operand reuse.
- `epilogues.h`: local reductions, shared-head sums and direct output stores.
- `device_ops.h`, `primitives.h`: shared synchronization/chunk views and PTO primitives; no operation dispatcher.
- `blocked_inverse.h`: forward-style diagonal solves and Cube block joins for recomputation.
