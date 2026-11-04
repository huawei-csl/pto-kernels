# GDN backward development

## Current rough baseline (v0)

This commit is a rough correctness baseline before sequence-wide stage fusion. Frozen timing samples, source hashes and validation results are in `tests/baselines/gdn_mega_bwd_rough_v0.json`; preserve that file when comparing future implementations.

User selected a staged FLA-style chunked backward instead of a token-state tape.
Sources live in `csrc/kernel/gdn_mega_bwd/`:

- `backward.py`: validation, six output allocations, one workspace allocation and one native enqueue. No equations, chunk traversal or per-primitive calls.
- `binding.py`: cached device/native compilation and ctypes ABI metadata.
- `launch.cpp`: native packed-sequence/chunk traversal, WY/backward equations and reusable workspace.
- `kernel.cpp`: native FP32 PTO matrix products, elementwise operations, gate scans, reductions, triangular inverse and grouped-head sums.

The API is `backward(q,k,v,g,beta,dout,initial_state=None,dfinal_state=None,cu_seqlens=None,scale=1)` and returns FP32 `(dq,dk,dv,dg,dbeta,dh0)`. Inputs can be FP16, BF16 or FP32. It supports packed B=1, D=128, grouped heads, tails and empty sequences. Gates are natural-log retention; raw output scale defaults to 1.

Chunks are 128 tokens, with 128x128 FP32 tiles. Each matrix product, scan, elementwise operation or reduction is a PTO launch, coordinated in C++. This is a correctness baseline, not FLA's exact fusion boundaries or a throughput-optimized megakernel.

The existing forward megakernel is unchanged and is **not called**. Nothing is saved by it for this backward. The native replay loop reconstructs incoming chunk states from the inputs, then the reverse pass rebuilds gates, Gram matrices, the inverse and W/U. Only FP32 boundary states survive across chunks; inverses are temporary and recomputed. The replay skips the final chunk update because backward needs its incoming state, not its outgoing state.

All tensor computation is PTO, including casts, strided packing, padding, shared-head expansion, triangular masks, zero initialization and gradient assembly. Python validates shape/stride and sequence-boundary metadata, allocates exactly seven tensors, and enqueues the complete backward once. Allocation count is independent of sequence length. An NPU `cu_seqlens` is copied to CPU solely for host scheduling; model tensors stay on device. Tests use `TorchDispatchMode` to reject any Torch operation other than allocation during backward (with CPU boundary metadata) and assert exactly seven allocations.

The C++ entry uses `OpCommand::RunOpApi`, matching the existing forward launcher, and captures input descriptors, output pointers and sequence metadata by value. The operation runs through Torch's host task queue on the current NPU stream. Workspace reuse and subsequent Torch producers/consumers are ordered on that stream; no entry, exit or chunk-boundary device synchronization is needed. For the synchronization-free path, supply CPU `cu_seqlens` or omit it; NPU boundary metadata still needs the compatibility copy above. This is not yet a registered autograd operator or a verified graph-capture implementation.

## Algorithm

1. Replay preceding chunks: prefix gates, causal key products, triangular inverse, W/U, state update. Retain incoming H only.
2. Traverse 128-token chunks backwards; recompute the inverse and W/U from inputs.
3. Compute local value gradients and propagate state adjoints to the preceding chunk.
4. Compute output/state contributions to dQ, dK and dW.
5. Differentiate WY and the inverse (`dL=-A^T dA A^T`, strict lower), obtaining dV, dBeta and remaining dK contributions.
6. Reverse-scan the gate gradient using the state-energy identity below, reduce shared-head dQ/dK and assemble outputs in PTO.

Gate-gradient identity: with `B_t = dS_t + scale*q_t*do_t^T`,
`dg_t = <B_t,S_t> - <dv_t,v_t> = dg_(t+1) + <dq_t,q_t> - <dv_t,v_t>`.
At each chunk boundary, initialize the reverse scan with `<H_end,dh_end>`.
We compute that endpoint as `exp(G_end)*<H_start,dh_end> + <z@dh_end.T, exp(G_end-G)*k>`.
Here dQ and Q are the ungrouped, consistently scaled intermediates. This removes several separately cancelling WY gate-gradient terms without changing the derivative.

Reference algorithm: FLA commit `12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e`, `fla/ops/gated_delta_rule/chunk.py` and `wy_fast.py`, including the Ascend backend. The inverse uses the repository's rank-one back-substitution algorithm; it stores A transposed to avoid an extra transpose.

## Fast iteration

Reuse `gdn-mega-opt`, repository `/mounted_home/autokernel_gdn/pto-kernels`.

```bash
docker exec -w /mounted_home/autokernel_gdn/pto-kernels \
  -e OMP_NUM_THREADS=4 -e NPU_DEVICE=npu:0 gdn-mega-opt bash -c \
  'source /usr/local/Ascend/cann/set_env.sh; python -m pytest tests/test_gdn_mega_bwd.py tests/test_gdn_mega_bwd_reference.py -xq'
```

Device code and the native launcher have separate content-addressed caches. The device translation unit builds with bisheng (4.71 s previously); the native launcher builds with g++ and Torch/NPU headers (12.09 s initially). Editing native scheduling does not recompile device kernels or rebuild the repository extension. Content-addressed libraries under `build/gdn_mega_bwd/` are reused unchanged. Run only one NPU test job at a time.

## Validation and numerical findings

- CPU FP64 oracle: `tests/gdn_mega_bwd_reference.py`; independent finite differences: `tests/test_gdn_mega_bwd_reference.py`.
- Hardware tests compare all six gradients to `.backward()` on the exact rounded inputs, with fixed elementwise `atol=2e-5, rtol=2e-4`. They also check input integrity.
- The discarded token-recurrence prototype passed short cases but retained `4*T*Hv*128*128` bytes of token states (8 GiB at 8K tokens, 16 heads). A 513-token weak-decay case narrowly failed dK. An expm1-style decay update helped but did not fully resolve it.
- The original C=128 implementation failed one dQ element for repeated identical keys, beta=1 and g=0. C=64 was a temporary workaround. The current implementation stays at C=128 with the same fixed tolerances.
- Ordinary cube accumulations and then ordinary vector tree sums were insufficient for that cancellation case. Selected sensitive products now use a vector TwoSum tree, carrying low parts until final rounding. Other products retain the faster native FP32 cube path. Products themselves are still FP32; this is not FP64 arithmetic.
- Extending repeated-key tests to 513 tokens exposed a dK cancellation, addressed by compensated products in the local adjoint and state-related derivatives. A remaining dG cancellation disappeared using the state-energy identity above. No input-specific dispatch or tolerance relaxation is used.
- Packing uses contiguous DMA rows when possible and device scalar gathers for arbitrary strides. Output assembly writes contiguous disjoint blocks, avoiding races between heads on shared cache lines.
- PTO column vectors require aligned physical storage even for a scalar reduction (8x1 physical, 1x1 valid). TMUL requires row-major views; row/column aliases can share UB storage.
- UB is 192 KiB on 910B2, with the upper 8 KiB reserved by PTO. Half-matrix stripes keep inverse and reduction temporaries below that limit.

## Next optimizations after correctness

A 1,024-token sequence currently launches **934 kernels**: 2 initialization + 7×21 state-replay + 8×98 reverse-chunk + 1 final state-gradient write. This count comes from the native control flow, not a hardware profiler. One Python enqueue is not one device launch. The next target is roughly seven sequence-wide stages: chunk preparation, forward state replay, local output backward, reverse state propagation, output/state derivatives, WY backward, and final gradient assembly. Local stages can parallelize across chunks; state dependencies remain inside scan kernels.


Fuse adjacent operations and reuse workspace; profile compensated products before optimizing them; accept explicitly exported forward states/inverses; batch independent chunk work; replace native host chunk loops with persistent device state scans. Preserve the independent oracle, allocation-only guard and fixed acceptance bounds.

## Native scheduling and timing

The previous Python implementation allocated one tensor and made one ctypes call per primitive, then synchronized each chunk. It has been removed, including `pto_ops.py`. Moving the equations into a native callback alone is not kernel fusion: the device still executes the staged primitives.

Hardware wall-time benchmark (Ascend 910B2, H=8, Hk=2, D=C=128, FP16 Q/K/V/beta, FP32 gates, zero initial/final adjoints): one warmup, five measured calls, median, including allocation and final synchronization, excluding build. Same inputs and unchanged device code:

| Tokens | Python orchestration (ms) | Native orchestration (ms) | Speedup |
| --- | --- | --- | --- |
| 128 | 6.020 | 5.005 | 1.20x |
| 512 | 25.434 | 19.561 | 1.30x |
| 2048 | 102.138 | 77.748 | 1.31x |

Reproduce with `PYTHONPATH=. python tests/benchmark_gdn_mega_bwd.py` in the container after sourcing CANN. The benchmark also reports enqueue time separately from completion time. A second run measured enqueue medians of 0.352 / 0.356 / 0.338 ms at 128 / 512 / 2048 tokens, including Python validation and allocations; completion remained 5.022 / 19.562 / 77.777 ms. These are hardware wall times, not simulator pipeline metrics.

## Verification results

- Prior all-PTO device baseline: **322 passed in 220.19 s** (48 backward, 8 CPU reference, 266 forward).
- Native orchestration: initial **48 hardware cases passed in 58.75 s**, including the 12.09 s host build. Final expanded validation: **57 passed in 41.51 s** (49 hardware tests and 8 CPU-reference tests), including queued producer/consumer and workspace reuse; fixed seven-allocation assertions pass for every case.
- Backward coverage: all six gradients in FP16/BF16/FP32, packed and empty sequences, exact chunk boundaries, weak decay, repeated keys through 513 tokens, missing states, noncontiguous model/adjoint/state tensors, a nondefault stream and rejection of Torch computation.
- Reference tests include finite differences. Forward regression includes sequences up to 32K; backward hardware coverage currently reaches 513 tokens. No claim of validation on A5 or of throughput optimization.

Full regression command:

```bash
docker exec -w /mounted_home/autokernel_gdn/pto-kernels \
  -e OMP_NUM_THREADS=4 -e NPU_DEVICE=npu:0 -e PYTHONPATH=python \
  -e GDN_REFERENCE_CACHE=benchmark_data/gdn_mega/reference_cache gdn-mega-opt bash -c \
  'source /usr/local/Ascend/cann/set_env.sh; python -m pytest tests/test_gdn_mega_bwd.py tests/test_gdn_mega_bwd_reference.py tests/test_gdn_mega.py -qs --durations=5'
```

Boundary states cost approximately `4 * ceil(T/128) * Hv * 128 * 128` bytes for one sequence: **64 MiB at 8K tokens and 16 heads**. This is allocated during backward, not retained from forward. Inputs, gradients and per-chunk temporary workspace are additional; there is no per-token state tape or retained inverse tape.

The native workspace additionally reserves 96 matrix-sized scratch slots and two state-adjoint slots, all reused across chunks/sequences. At 8K tokens and 16 heads the combined workspace is **162 MiB** (64 MiB boundary states + 96 MiB scratch + 2 MiB adjoints). Packed sequences reuse checkpoints sized for the longest sequence. There are no per-chunk allocations or frees.
