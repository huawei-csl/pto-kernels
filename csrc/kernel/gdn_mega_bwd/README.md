# Sequence-wide GDN backward

Packed B=1, D=C=128, FP16/BF16/FP32 inputs and FP32 gradients. The backward
can recompute from its inputs or consume an optional inverse cache from forward.
Python validates, allocates six outputs and one workspace, then enqueues once.

## Optional forward inverse cache

Allocate a separate contiguous FP32 tensor `[num_chunks, H, 128, 128]`, where
`num_chunks = sum(ceil(sequence_length/128))`, and pass it to both calls:

```python
inverse_cache = torch.empty(num_chunks, H, 128, 128, device=q.device, dtype=torch.float32)
pto_gdn_mega(**forward_launch_args, inverse_cache=inverse_cache)
grads = backward(q, k, v, g, beta, dout, h0, dht, cu_seqlens,
                 scale=scale, inverse_cache=inverse_cache)
```

The cache belongs to those exact k/g/beta and sequence boundaries. Retain it
until backward finishes; the backward reads it without modifying it. A model
autograd wrapper can retain it using `ctx.save_for_backward` alongside inputs.
Omitting it keeps the recomputation path. Other inverse scratch remains opaque and is not a substitute for this cache. Rebuild the forward host binding
and device library together for the added optional argument.

The forward does not materialize a complete inverse. For `M=I+L`, it already
computes `D0=M00^-1`, `D1=M11^-1` and `P=-D1*L10`. The cache stores
`[D0, 0; P, D1]`. Existing vector/Cube exchanges are redirected into the
cache instead of copied afterward. No additional forward matrix products,
data stores or kernel launches are introduced.
Backward completes `A10=P*D0` with one 64x64 Cube product, transposes the
result, and skips the triangular inversion. KK, W/U, states and Z are still recomputed. Gate prefixes can also be
retained as described below. The fused backward uses four stages.

The cache uses 16 MiB at 2K tokens/H=16 and 64 MiB at 8K/H=16 (plus padded
packed tails). It survives from forward to backward; the existing 816.5 MiB
backward workspace at 8K/H=16 is unchanged. Forward's current dtype contract
is FP16 q/k/v/beta and FP32 gates.

Measured at H=16/Hk=2 with FP16 inputs and FP32 gates:

| Tokens | Cube baseline backward | Cached backward | Forward device, cache off → on |
| --- | --- | --- | --- |
| 2048 | 5.534 ms | 4.529 ms | 345.271 → 348.904 us |
| 8192 | 20.481 ms | 16.641 ms | 1268.518 → 1280.014 us |

Backward uses synchronized wall time including allocation and recomputation.
Forward uses 31 graph trials of 20 calls each; allocation is excluded.
Caching adds approximately 1% forward time here, rather than being literally
free. The 8K preparation stage drops from 10.977 to 7.107 ms. **349 tests pass**,
including cached gradients at every supported head count. Measurements and
source hashes are recorded in
[`gdn_mega_bwd_inverse_cache_v3.json`](../../../tests/baselines/gdn_mega_bwd_inverse_cache_v3.json).

## Existing gate and beta buffers

Forward already leaves `(g_t, g_sum, beta_t)` in physical `[H,T]` layout.
`g_t` and `g_sum` are the high and low FP32 parts of each chunk's cumulative
log gate; `beta_t` is FP16 beta. Keep these exact buffers alive and pass
`gate_cache=(g_t, g_sum, beta_t)` to backward, optionally alongside
`inverse_cache`. No forward code or computation is added for this reuse.
Backward loads them contiguously and skips the prefix scan and beta packing;
exponentials and pairwise gate decay are still computed in PTO. These buffers
must belong to the same inputs and packed sequence boundaries, and must not
be overwritten by another forward before backward finishes.

Retaining all three costs 0.3125 MiB at 2K/H=16 or 1.25 MiB at 8K/H=16.
`ctx.save_for_backward` can retain them without copying. The original `g`
and `beta` inputs remain part of the backward API. The original `h0` is
already reused; chunk-boundary states are still replayed. This forward does
not compute an `rstd`: Q/K normalization happens upstream and its autograd
operation owns any normalization statistics.

Before fusion, H=16/Hk=2, 31-sample measurements with the same inputs gave:

| Tokens | Inverse cache only (ms) | Inverse + gate/beta cache (ms) |
| --- | --- | --- |
| 2048 | 4.670 | 4.320 |
| 8192 | 17.194 | 15.799 |

Synchronized wall time includes backward allocations. These are consecutive
processes, so clock/thermal variation can affect the comparison. The 8K trace
showed seven stages, with preparation at 5.733 ms. See the
[raw samples and source hashes](../../../tests/baselines/gdn_mega_bwd_gate_cache_v4.json).
Add `--gate-cache` to the inverse-cache benchmark command below to reproduce.

## Four device stages

| Stage | Work | Scheduling |
| --- | --- | --- |
| 0 | Pack inputs, gates, inverse, W/U; local value/state gradients | Independent chunk/head tasks |
| 1 | Replay incoming states and Z; reverse state adjoints and complete dU; write dh0 | Forward then reverse chunk loops inside each sequence/head task |
| 2 | Output derivatives and WY/inverse derivatives | Independent chunk/head tasks |
| 3 | Gate scans, shared-head reductions, all final outputs | Independent chunk/head-group tasks |

The host launches these stages once per group of up to 64 packed sequences:
`4 * ceil(sequence_count / 64)` launches, independent of token count. Local
gradients need no replayed state, so they run alongside preparation. A state
scan owns one entire sequence/head, allowing forward replay and reverse
propagation in the same kernel without a cross-block barrier. The final
stage owns complete shared-key head groups and can finish their reductions
without atomics. Assembly remains separate to preserve fine-grained
head parallelism in the expensive output/WY stage.

Final tasks split value heads in groups of at least eight, aligned with whole
shared-key groups and 32-byte scalar output spans. Irregular head counts use
one group per chunk. This preserves head parallelism where possible without
concurrent partial writes to the same output cache line. For H=16/Hk=2 there
are two final tasks per chunk. No host chunk loops or device-wide barriers
are needed; launch ordering supplies the dependencies between stages.

Fusion also removes work inside the stages: dense inputs use 64-row DMA tiles;
causal masks use broadcasts and clamps; the inverse-adjoint mask, weights and
beta reduction stay in UB; paired native Cube products accumulate directly
in L0C; shared-head sums remain in UB until their output store; and gate/beta
gradients use tiled transposes for output assembly.
The FP32 accuracy path retains its compensated and separately rounded products.
Other intermediates still use GM, and the backward workspace remains 816.5 MiB
at 8K/H=16. This is not a fully on-chip implementation.

All dense matrix products use Cube instructions. FP16/BF16 q/k/v use native
FP32 Cube accumulation. When any of q/k/v is FP32, products use four K=32
Cube partials with compensated addition of the results. That accuracy path
also sums gate intervals directly, reconstructs dQ within 16-token blocks
inside each 128-token chunk, and computes dBeta as `dot(dUpdate, Z)/beta`
where beta is nonzero (the general WY derivative handles beta=0).
No matrix product falls back to vector dot loops. All of this remains within
the same four device stages; Python only validates and allocates.

Measured with both forward caches, H=16/Hk=2, 31 synchronized wall-time
samples including allocations:

| Tokens | Seven stages, f686c16 (ms) | Four stages (ms) | Reduction |
| --- | --- | --- | --- |
| 2048 | 4.320 | 2.807 | 35.0% |
| 8192 | 15.799 | 9.922 | 37.2% |

The 8K trace records 2.291 / 2.246 / 4.233 / 0.810 ms across the four stages.
A three-stage variant with fused assembly measured 3.207 / 10.176 ms and was
rejected: its head grouping reduced useful parallelism in output/WY work.
These comparisons use consecutive development runs, so clock/thermal state
can vary. Forward and its cache contract are unchanged; backward has no Torch
numerical operations. Raw samples and hashes are in
[`gdn_mega_bwd_fused_v5.json`](../../../tests/baselines/gdn_mega_bwd_fused_v5.json).
Final validation: **107 tests passed in 191.11 s**, with the same per-gradient
error bounds, allocation-only guard and cache integrity checks.
The 2x-forward target has not yet been reached.

## Files

- `backward.py`: validation, allocation and one native enqueue.
- `binding.py`: C ABI and separately cached device/native builds.
- `launch.cpp`: packed-sequence metadata and Torch queue integration.
- `kernel.cpp`: fused stage bodies and their device task loops.
- `device_ops.h`: cube/vector cooperation and scratch helpers.
- `primitives.h`: FP32 Cube matrix products and vector elementwise/reduction operations.
- `layout.h`: stage buffer lifetimes and metadata ABI.

## Comparison with rough baseline

Commit **150fed0** preserves the original working implementation. Its
[raw measurements and source hashes](../../../tests/baselines/gdn_mega_bwd_rough_v0.json)
remain unchanged. One 1,024-token sequence previously required 934 device
launches; the fused hardware trace shows four.

Archived comparison before replacing vector matrix products ([raw samples and trace summary](../../../tests/baselines/gdn_mega_bwd_staged_v1.json)):

| Tokens | Rough native baseline (ms) | Seven stages (ms) |
| --- | --- | --- |
| 128 | 5.005 | 7.279 |
| 512 | 19.561 | 14.033 |
| 2048 | 77.748 | 41.082 |

Ascend 910B2, H=8, Hk=2; median of five calls after one warmup. Includes
allocation, recomputation and final synchronization, excluding compilation.
That version used compensated vector matrix products in stages 2 and 4 and
passed 66 checks at `atol=2e-5, rtol=2e-4`. These are historical numbers.

Archived seven-stage Cube results ([raw samples, profile and source hashes](../../../tests/baselines/gdn_mega_bwd_cube_v2.json)),
H=16, Hk=2, D=128, FP16 inputs with FP32 gates, same timing method:

| Tokens | Vector products v1 (ms) | Cube products v2 (ms) | Speedup |
| --- | --- | --- | --- |
| 2048 | 75.378 | 5.534 | 13.62x |
| 8192 | 293.984 | 20.481 | 14.35x |

That version’s 8K trace contains seven launches. Local/output gradients fell from
275.42 ms combined to 2.57 ms; preparation/recomputation now takes 10.98 ms.
These speedups describe the FP16 path; FP32 inputs use the additional accuracy
work described above. **71 tests pass** (61 hardware, 10 CPU reference) in
109.04 seconds with cached builds.

## Accuracy checks

Each current hardware case runs the independent recursive PyTorch reference
twice with identical rounded inputs and upstream gradients. The golden run
computes in FP32 for FP16/BF16 inputs, or FP64 for FP32 inputs. The baseline
computes in the input dtype, distributing beta through the state update and
scaling q before its contraction. Autograd differentiates both recurrences.
For each of dq, dk, dv, dg, dbeta and dh0, the kernel's maximum absolute error
must be at most twice the baseline's maximum absolute error plus `1e-5`. Golden
gradients retain their computation precision, since this API returns FP32
gradients. CPU tests also check that the reordered equations and derivatives
agree in FP64. Cache tests also run the forward megakernel to produce saved
buffers; recomputation tests need only the reference forward. The device
backward itself never calls the forward megakernel. The latest gate-cache
validation passes 107 checks (97 hardware, 10 CPU reference).

Sequence-wide stages use more GM storage: 12 FP32 matrices and seven FP32
vectors per chunk/head, plus 45 MiB of reusable per-block scratch (including
four Cube partial-result slots). At 8K tokens
and 16 value heads the workspace is **816.5 MiB**, versus the rough baseline's
162 MiB. Slots are reused after their last consumer (W→dW, U→Z, Q→dQ,
dOut→dV, state adjoint→dK). This workspace exists during backward only.

The model autograd wrapper remains separate work.
See [development notes](../../../memory_bwd.md) for validation and profiling.

After sourcing CANN, run from the repository root:

```bash
OMP_NUM_THREADS=4 NPU_DEVICE=npu:0 python -m pytest tests/test_gdn_mega_bwd.py tests/test_gdn_mega_bwd_reference.py -q
OMP_NUM_THREADS=4 PYTHONPATH=. python tests/benchmark_gdn_mega_bwd.py
OMP_NUM_THREADS=4 PYTHONPATH=. python tests/benchmark_gdn_mega_bwd.py --tokens 1024 --profile build/gdn_mega_bwd/profile/current
OMP_NUM_THREADS=4 PYTHONPATH=python:. python tests/benchmark_gdn_mega_bwd.py --tokens 2048 8192 --heads 16 --inverse-cache
```
