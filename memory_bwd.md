# GDN backward development

## Remove the remaining operation scheduler (2026-11-06, v10)

Committed the validated v9 work first as `2c181b2`. Its device library is
`45d6c0fa80997f25/kernel.so`. The native launcher, Plan/workspace layout,
forward/cache format and per-gradient error bounds remain unchanged.

Deleted `Ops`, its allocator, generic `call(kind,op)` dispatcher, nested tensor
expressions and implicit joins. This is not a relocation of the old class.
`kernel.cpp` now has explicit compensated stages with fixed scratch addresses;
`pipeline.h` retains the already optimized FP16/BF16 schedules. Small inline
PTO tile helpers are instructions inside a stage, never additional launches.

Concrete changes to the formerly generic FP32 path:

- Pack/scale Q together and batch independent vector preparation. Fuse the
  compensated product's rounded result directly into its consumer in UB,
  preserving the order of additions while eliminating intermediate GM matrices.
- Evaluate 64 gate-interval Kahan sums in SIMD lanes, preserving each row's
  descending summation order, and use both vector subcores for the two stripes.
  Keep the compensated prefix/reverse scans required by the FP32 error budget.
- Replace local-score scalar masking with four 16x16 tiled loads per subcore.
  Load block weights through aligned DMA, transpose and tile insertion.
- Replace the remaining serial rank-one inverse with the existing blocked
  diagonal solver and Cube joins. The full FP32 suite passed unchanged bounds.
- Give all four K=32 products disjoint L0A/B operands and alternating 64-KiB
  L0C slots. MTE1, M and FIX overlap; FIX->M events protect accumulator reuse.
  Four partial results still receive the original compensated reduction.
- Replace scalar beta-gradient division with masked PTO comparison/selection
  and vector division; beta=0 retains the general derivative without dividing
  by zero. Remove the unused generic GEMM modes, scalar inverse and reducer.

Validation: **152 tests passed in 359.94s**, before changing reference reuse.
The subsequently optimized saved-buffer test group passed **75 tests in
55.74s**. Instrumentation verified **30 oracle calls for 15 unique input cases**,
versus 150 calls before. Only fixed-seed CPU oracle results are shared within
one process. Every kernel call, tolerance assertion, allocation-only guard,
input/cache integrity check and stream check still runs; nothing is saved to
disk or added to the production path.

The benchmark now supports `--quick`, repeated `--shape T HV HK`, `--dtype`,
`--warmup` and `--repeats`. Quick runs use two shapes, one warmup and three pairs.
Development used small smoke tests and quick comparisons; the full numerical
suite ran once after kernel changes stabilized, followed by the affected test
cache group. No repeated full sweeps, forward rebuilds or profiler runs.
Device/native builds remain separately content-addressed. A tile-typed helper
must be guarded by `__CCE_AICORE__`, including its signature, because the
bisheng host pass does not define device tile aliases.

Final paired measurements, alternating libraries on identical data, two warmup
pairs and seven measured pairs. FP32 recomputes; FP16 uses actual saved inverse
and gate/beta buffers. Wall times include allocation/enqueue and synchronization,
excluding build and forward:

| Input / cache | Tokens | Hv / Hqk | v9 ms | v10 ms | Speedup |
| --- | --- | --- | --- | --- | --- |
| float32 / none | 256 | 8 / 2 | 1.546 | 1.048 | 1.48x |
| float32 / none | 513 | 8 / 2 | 2.707 | 1.716 | 1.58x |
| float32 / none | 2048 | 16 / 2 | 12.552 | 6.869 | 1.83x |
| float16 / BF16 + gates | 2048 | 16 / 2 | 1.447 | 1.454 | 1.00x |
| float16 / BF16 + gates | 8192 | 64 / 16 | 13.610 | 13.592 | 1.00x |

FP32 improves **1.48–1.83x**. The optimized FP16 path stays within **0.5%** of
v9; these are not additional training-dtype speedups. The previous 2x native
backward target remains unmet. Raw samples, source/library hashes, commands
and development comparisons are in
`tests/baselines/gdn_mega_bwd_no_ops_v10.json`.

## Explicit native schedules (2026-11-06)

Snapshot requested before optimization: `e82451a` (clean tree after commit).
Baseline device library: `4ef9e56e099811b5/kernel.so`; native launcher:
`4b84318d17296a18/launch.so`. No forward, launch ABI, saved-cache format,
workspace sizing or numerical tolerance changes in this iteration.

The FP16/BF16 schedules in `pipeline.h` no longer use `Ops`, its temporary
allocator, or generic operation dispatch. `cube_load`/`cube_product` are inline
PTO tile instructions, never launches. The generic algebra remains for the
compensated FP32 accuracy path. This change alone does not explain the measured
gain: the useful changes alter data movement and overlap.

- Pack Q and apply its scale in the same UB pass. Full contiguous Q/K chunks
  produce their shared Gram matrices directly from the original 16-bit inputs
  on Cube, accumulating in FP32; tails/strides retain the FP32 packing path.
- Share Q as well as K across value heads when fused outputs keep dQ elsewhere.
- Weight all preparation RHS matrices in UB. Reuse inverse and dOut operands
  across consecutive products. Preparation uses five phases instead of six.
- Interleave independent forward replay and reverse adjoint scans. Forward
  state bases occupy UB 96..128 KiB, reverse bases 128..160 KiB. Four phase
  boundaries per paired step replace eight; the previous reverse update is
  completed alongside the next forward product. Arithmetic order is retained.
- Fuse the three local dK contributions into one UB lifetime; no intermediate
  dK materialization. Delay independent key products into the WY vector phases.
  Combine beta terms afterward in their original rounding order.
- Form `(dKK+dKK.T) @ K` using two Cube products and one L0C accumulator. This
  replaces vector transpose/add and a full GM write/read; operands stay in L1.
  It adds one Cube product (nine -> ten in the derivative stage), but runs faster.
- Load contiguous original V into UB at its two consumers, eliminating its
  retained FP32 GM copy. Strided feature dimensions keep the existing fallback.
- Combine gate energy terms in one UB pass instead of two GM vector operations.

One implementation hazard: consecutive Cube products need M->MTE1 protection
for L0A/B and FIX->M protection for L0C. A missing M->MTE1 dependency produced
an L0B conflict during development; the final helper fences both resources.

Rejected experiments:

- Three BF16 products on high/low operand pairs passed 37 targeted cases but
  slowed backward 12-16%. Extra packing/GM copies outweighed faster products.
  No such approximate arithmetic remains enabled.
- Replacing matrix gate exponentials with bounded outer products of vector
  exponentials passed 19 targeted cases, but had no consistent timing gain.
  Reverted to the original gate arithmetic, including its large-span behavior.

All timings compare cached backward on identical inputs with the same actual
BF16 inverse factors and gate/beta buffers produced by `gdn_mega` forward.
Three warmup pairs, eleven measured pairs, alternating launch order in one
process. Synchronized wall latency includes allocation/enqueue; compilation
and forward are excluded. Baseline and candidate share the native launcher.
Raw paired samples, source/library hashes and experiments are retained in
`tests/baselines/gdn_mega_bwd_explicit_v9.json`; the comparison is reproducible
with `benchmarking/gdn_mega_bwd_compare.py BASELINE_KERNEL_SO`.

| Tokens | Hv / Hqk | Snapshot ms | Current ms | Speedup |
| --- | --- | --- | --- | --- |
| 2048 | 16 / 2 | 1.673 | 1.466 | 1.14x |
| 8192 | 16 / 2 | 5.171 | 4.273 | 1.21x |
| 8192 | 64 / 16 | 16.945 | 13.811 | 1.23x |
| 8192 | 48 / 16 | 13.049 | 10.752 | 1.21x |
| 8192 | 32 / 16 | 9.504 | 7.724 | 1.23x |
| 8192 | 32 / 8 | 9.420 | 7.610 | 1.24x |
| 32768 | 64 / 16 | 64.456 | 52.946 | 1.22x |

All **152 backward/reference tests passed in 379.56s**, with unchanged per-gradient
`2 * reordered_torch_error + 1e-5` bounds. Forward was not modified; its saved
cache paths are exercised by the backward suite. Long timing cases check finite
gradients; the full numerical oracle coverage reaches 513 tokens.

Matched warmed hardware traces at 8K/Hv64/Hqk16:

| Stage | Snapshot ms | Current ms |
| --- | --- | --- |
| Preparation | 5.137 | 4.277 |
| Interleaved state scans | 3.910 | 3.072 |
| Derivatives/output | 7.129 | 5.951 |

These are kernel durations, not utilization counters. All three stages improve,
but the total gain is **1.14–1.24x**, not the requested 2x. The derivative stage
still contributes about 45% of device time; preparation and state scans account
for the rest. No single wrapper removal can eliminate those remaining products
and transfers. Existing heatmaps have not been regenerated.

## Export completed inverse factors once (2026-11-05)

Replaced the initial low-precision export's duplicate writes. Forward now builds
FP32 factors in two reusable per-core scratch slots and exports each completed
FP16/BF16 factor stripe once. Only the 16-bit tensor survives for backward.
The slots reuse `o_workspace_qs/gated` and the previously idle
`wy_workspace_a1/a2`; each pair supplies two FP32 64x128 halves. This is
128 KiB/core (3 MiB for 24 cores), independent of sequence length. No allocations
or kernel launches were added; the matrix arithmetic remains FP32.

Cube writes/reads its joins through the current scratch slot. Vec prepares the
next chunk in the other slot. After completing all joins, Cube signals flag 8;
Vec then converts/exports the completed current slot while Cube runs the
factored RHS solve and output/state products. Existing flag 9/6/7 handshakes
finish output/state and protect slot reuse. The export uses UB at 96 KiB and
0 KiB, preserving the live FP32 recurrent state at 64–96 KiB.

The old export wrote 32 KiB of incomplete 16-bit factors and then overwrote
14 KiB through seven small Cube stores. The new export writes 32 KiB once,
removing those Cube exports and putting conversion in the vector idle window.
A temporary FP32 factor is working scratch, not a second saved activation.

With identical build flags, before/after checks load and assert library hashes.
Within each process, rotate cache dtype order; 3 warmups + 15 timed samples:

| Tokens / Hv / Hk | Old BF16 FWD ms | New BF16 FWD ms | New FP32-cache FWD ms |
| --- | --- | --- | --- |
| 2K / 16 / 2 | 0.423 | 0.400 | 0.398 |
| 8K / 16 / 2 | 1.438 | 1.347 | 1.342 |
| 8K / 64 / 16 | 4.322 | 4.107 | 4.109 |
| 32K / 64 / 16 | 17.233 | 16.643 | 16.664 |

FP16 behaves similarly. The measured smaller-cache overhead versus FP32 cache
saving is now at most about 0.7%, replacing the old 5–8% in this controlled
build comparison (6–8.5% in the original run). Absolute comparisons across
processes include clock/thermal variation; the dtype ratios within one process
are the better overhead measure. Retained inverse bytes remain halved.

The full run passes 418 unique tests (30 low-precision cases, then the other
388), including exact cache rounding and bitwise forward output/state.
A follow-up compile-time export strategy is selected in the host launch shim,
keeping slot selection and export handshakes out of the FP32/no-cache loop.
There is still exactly one forward launch. Selecting both expanded pipelines
inside one device kernel was slower and was rejected.

Final same-process comparison, with all three libraries deep-bound and the
same input/workspace pointers; rotate/reverse launch order, 4 warmups + 20
samples. These are synchronized raw-launch wall times, excluding Torch enqueue:

| Tokens / Hv / Hk | Old FP32-cache ms | Old BF16-cache ms | Final FP32-cache ms | Final BF16-cache ms |
| --- | --- | --- | --- | --- |
| 2K / 16 / 2 | 0.3940 | 0.4235 | 0.3970 | 0.3961 |
| 8K / 16 / 2 | 1.3611 | 1.4804 | 1.3650 | 1.3657 |
| 8K / 64 / 16 | 4.2030 | 4.4511 | 4.2060 | 4.1924 |
| 32K / 64 / 16 | 16.2060 | 16.9874 | 16.2132 | 16.1718 |

BF16 is now within 0.6% of the original FP32-cache latency at these shapes;
the final FP32 path stays within 0.8% of the original. There is no measured
material default-path regression. All 341 affected forward/cache tests pass
in 374.53s after the host-dispatch change; the other 77 tests exercise the
unchanged backward/reference implementation. Full records and library/source
hashes are in `tests/baselines/gdn_mega_inverse_export.json`.

## Initial lower-precision export (2026-11-05, superseded above)

Committed the complete v8/FLA-comparison snapshot first: **c2a9836** (clean
working tree before this experiment). Added optional **FP16 and BF16** cache
storage, selected by the dtype of `inverse_cache`; FP32 remains supported and
is still the benchmark default. Forward q/k/v/beta remain FP16.

- Both smaller formats pass the unchanged `2 * Torch error + 1e-5` gradient
  bounds. In the 15-case rounding screen, the worst error/budget ratio was
  **0.02801 FP16**, **0.20618 BF16**, versus **0.001319 FP32**. The first native
  45-case screen exactly reproduced these max errors. The final implementation
  passes **152 backward/reference tests**, including 30 actual 16-bit cache
  cases. The forward suite also passes **266 tests in 164.65s**. No tolerance
  changes.
- Retained inverse storage is halved: **64 -> 32 MiB at 8K/H16**, or
  **256 -> 128 MiB at 8K/H64**. Gates and backward workspace are unchanged.
- Forward's cache is also its working exchange in the FP32 path. Rounding that
  exchange would change forward numerics. The 16-bit path instead uses the
  original FP32 scratch and additional PTO exports: Vec `TCVT` for initial
  stripes, Cube `TSTORE`/FIX for completed inverse blocks. It adds no GEMMs or
  launches. Tests verify bitwise-identical output/final state and exact equality
  of the saved cache with rounding the FP32 factors.
- Backward converts directly into the existing FP32 transposed-factor buffer,
  then completes `D0.T @ P.T` on Cube using that buffer. A first version also
  wrote a separate expanded factor copy; removed that redundant GM write before
  final regression/timing. Python and C++ launchers only validate/allocate/pass
  dtype metadata; all production conversions run in PTO.

Same-process, rotating dtype order; 3 warmups + 15 measured synchronized-wall
samples. Backward includes allocations; forward buffers are preallocated:

| Tokens / Hv / Hk | FP32 FWD / BWD ms | FP16 FWD / BWD ms | BF16 FWD / BWD ms |
| --- | --- | --- | --- |
| 2K / 16 / 2 | 0.394 / 1.643 | 0.426 / 1.641 | 0.425 / 1.644 |
| 8K / 16 / 2 | 1.334 / 4.957 | 1.446 / 4.982 | 1.445 / 4.979 |
| 8K / 64 / 16 | 4.099 / 16.411 | 4.367 / 16.463 | 4.365 / 16.476 |
| 32K / 64 / 16 | 16.197 / 64.046 | 17.225 / 64.244 | 17.224 / 64.209 |

This is a **memory option**, not a speedup: forward is about **6.3–8.4% slower**
than FP32 cache export; backward differs by less than 0.5% in this sample.
Keep FP32 for the fastest forward; select BF16/FP16 when retained activations
matter more. `--cache-dtype bfloat16` / `float16` works in the existing benchmark
and extensive sweep; heatmaps identify and reject mixed cache dtypes.

Reproduce accuracy with `tests/check_gdn_inverse_precision.py --output <json>`;
`--roundtrip` isolates numerical storage error with test-only Torch casts.
Raw errors, timings, benchmark script and final source hashes are saved in
`tests/baselines/gdn_mega_inverse_precision.json`.

## FLA comparison chart

Added `tests/benchmark_gdn_mega_bwd_fla.py` and
`benchmark_data/gdn_mega_bwd/fla_comparison/speedup_heatmap.png/.svg`.
FLA commit 9e82a80 uses equal Q/K and value heads (16/24/32/48/64), so only
five shapes matched our prior TP sweep. A separate 25-shape PTO sweep now
matches batch, length, both head counts and both head dimensions exactly,
with 30 measured samples per direction and both forward caches enabled.
The existing TP heatmap is unchanged. `--plot-only` redraws without NPU work.

Cells show FLA/PTO median wall-time ratios and both latencies. The comparison
is explicitly labeled PTO FP16/C128 versus FLA BF16/C64. Input distributions,
allocation/autograd overhead and final-state/dh0 outputs also differ; this
is not a controlled same-dtype performance claim. No joint forward+backward
ratio is inferred from separate measurements. Source metadata, all 50 ratios
and raw PTO samples are retained in
`tests/baselines/gdn_mega_bwd_fla_v8.json`. Both kernels remain unchanged.


## Cached backward: local reuse and output fusion (v8)

All numerical changes use PTO-ISA, with native FP32 Cube products. Forward,
retained-cache format and workspace size are unchanged. Python/native launch
code still only validates, allocates and submits; no Torch computation was added.

The inverse cache removes the solve, but not KK/QK, W/U, local adjoints, state
replay or output derivatives. The cached v7 8K/Hv=64/Hqk=16 hardware trace was
5.626 / 4.632 / 7.764 / 2.278 ms for preparation, state scans, gradients and
assembly. Thus gradients plus assembly accounted for about half the call;
even eliminating all preparation would leave most of the backward work.
The saved inverse is a factorization, completed with one 64x64 Cube product,
not the full triangular solve. Caching was working; subsequent data movement
and synchronization limited the end-to-end gain.

A separate alternating comparison on the final code confirms that caches
still help: at 8K/Hv=64/Hqk=16, full recomputation is **19.506 ms** versus
**16.878 ms** cached (13.5% lower); at 8K/Hv=16/Hqk=2, **5.939 versus 5.104 ms**
(14.1% lower). Same inputs, three warmup pairs and eleven measured pairs.
The blocked recomputation solve is already much cheaper than the original
serial inverse, so skipping it cannot produce the early expected large gain.

Implemented:

- Native cached preparation transposes the read-only factors into A.T, then
  completes its upper-right block as D0.T @ P.T. No temporary factor copy.
- Both vector subcores construct their own decay stripes. Gate/beta prefix
  buffers remain read-only; both caches are still passed to every timed call.
- State scans retain the scaled incoming state in UB across their two products.
  Cube prefetches the next operand while the preceding product runs.
- `resident_cube.h` retains operands in seven L1 slots. Shared Q/K remain
  resident across the value heads sharing that key head. Matrix products keep
  native FP32 operands and accumulation; no lower-precision shortcut is enabled.
- `epilogues.h` fuses weighting, additions and reductions in UB. Independent DMA
  loads queue together before their consumer fence. dV energy is computed from
  the actual rounded dV and reused for dg, preserving the rounding contract.
- Large native-input grids fuse final output assembly into the gradient stage.
  Shared dQ/dK sums live at UB 120..184 KiB across value heads; helpers use the
  lower 120 KiB, leaving PTO's reserved 184..192 KiB untouched. dV writes directly
  to its final strided output. This removes the final dQ/dK/dV matrix round trips
  through GM.
- Fused tasks own lcm(8,Hv/Hqk) value heads (all heads if Hv is not divisible by
  eight), preserving aligned scalar writes and whole shared-key reductions.
  Three launches are used when at least 96 such chunk/group tasks exist.
  Smaller native grids retain four launches and omit unnecessary assembly
  cross-core joins; FP32 inputs retain the compensated four-stage path.

Hardware `msprof op --aic-metrics=Roofline` counters at 8K/Hv=64/Hqk=16,
comparing v7 gradients+assembly against the final fused output stage:

| Traffic | v7 (MiB) | v8 (MiB) | Reduction |
| --- | --- | --- | --- |
| GM to Cube L1 | 4608 | 2432 | 47.2% |
| GM to vector UB | 10548 | 5934 | 43.7% |
| Vector UB to GM | 5028 | 1956 | 61.1% |

These are transfers at the named interfaces, including L2 hits, not measured
HBM-only bandwidth. Mean per-Cube cross-core waiting for gradients+assembly
falls from 4.080 to 2.804 ms; state-scan waiting falls from 1.614 to 1.185 ms.
Output-stage Cube activity is still only 24.8% of the profiler denominator;
these improvements do not imply peak Cube FLOP utilization or a 2x-forward
result. Vector work and state dependencies remain substantial costs.

Development checks caught a matrix-only copy used on a vector and a compact
reduction scratch tile incorrectly reused for a full KK load; both were fixed.
An HF32 experiment failed the existing repeated-key FP16 dQ bound
(0.118866 error versus 0.0788064 budget) and was removed. No tolerances changed.
Packed `[129]+[1]*10` at H=64/Hk=16 and `[1]*48` at H=48/Hk=16 now exercise the
fused dispatch, state transitions, partial chunks and group sizes four/three.

Paired synchronized wall times on identical inputs, alternating old/new order,
three warmup pairs and eleven measured pairs; both paths consume the same
forward inverse and gate/beta caches. Allocation/enqueue is included, forward
execution excluded. The before library is `b54cbc2527d02828` (cached v7), not
v6 or uncached recomputation; the after library is `ade915863a5ac992`.

| Tokens | Local Hv / Hqk | Cached v7 (ms) | Cached v8 (ms) | Reduction |
| --- | --- | --- | --- | --- |
| 8192 | 64 / 16 | 20.732 | 16.960 | 18.2% |
| 16384 | 32 / 8 | 21.526 | 17.577 | 18.3% |
| 16384 | 48 / 16 | 30.830 | 24.675 | 20.0% |
| 8192 | 16 / 2 | 6.048 | 5.116 | 15.4% |
| 2048 | 16 / 2 | 1.889 | 1.660 | 12.1% |

All 18 paired cases improve, with reductions of 11.3–20.0%. At 8K/Hv=16/Hqk=2,
the end-to-end 5 ms target is still narrowly missed (5.116 ms). A separate
warmed hardware trace sums to **4.780 ms** across three kernels
(1.405 / 1.343 / 2.032 ms), excluding host allocation/enqueue and launch gaps.
This is a device trace, not an end-to-end median. The general 2x-forward
objective also remains unmet; do not describe this as a 2x end-to-end speedup.
Final validation: **122 passed in 276.73 s**, including 112 hardware and ten
CPU reference checks, with unchanged per-gradient bounds, cache/input integrity,
allocation-only guard, streams, empty sequences, packed tails and cancellation.
All 60 supported cached sweep medians improve over the recorded v7 sweep
(new/old latency 0.735–0.938, median 0.859). These separate-process comparisons
supplement the paired measurements above. All 80 cells were checked for both
cache flags, cache-saving forward denominators and explicit N/A handling.
The final heatmap and raw samples use both caches; old cached v7 data remain in
`tests/baselines/gdn_mega_bwd_cached_v7.json` and the `cached_v7/` plot directory.

## Cached training heatmap baseline (v7)

The previous `extensive/` and `tp_v7/` charts measured full recomputation.
The default `--extensive` sweep now explicitly passes both inverse and
gate caches produced by the forward call. Its forward denominator is
`forward_cache_ms`, so inverse-saving cost is included. Cache creation
happens before backward timing. Those results are preserved under
`benchmark_data/gdn_mega_bwd/cached_v7/`; unsupported cache producers (local
Hv=2,4,6) produce N/A cells in both directions, never uncached substitutes.
The plot rejects mixed cache modes and labels the saved buffers explicitly.
All 60 supported configurations completed with five samples per direction;
validated every ratio against `forward_cache_ms` and both cache flags.
Raw samples and source hashes are preserved in
`tests/baselines/gdn_mega_bwd_cached_v7.json`.

## TP=1/2 preparation optimization (v7)

The full-recomputation TP heatmap exposed a different bottleneck from cached
v6: at 4K/H=64/Hk=16, preparation consumed **12.898 ms** of a roughly 20.7 ms
backward. The uncached path still used 127 serial rank-one inverse updates on
one vector subcore and scalar GM gathers for each gate/beta token.

Implemented in PTO, without changes to forward, cache contracts or allocations:

- `blocked_inverse.h` reuses forward's eight 16x16 diagonal substitutions on
  both vector subcores, followed by 32/64/128 block joins on Cube. This computes
  the complete inverse for recomputation; cached inverse completion is retained.
- Native preparation now pipelines every cache mode, not just both-caches mode.
- Gate/beta packing loads aligned head panels and transposes them, replacing
  128 scalar GM reads per head/vector. Arbitrary strides retain the fallback.
- Native gate prefixes use a seven-step FP32 vector tree. FP32 q/k/v inputs
  retain the original compensated prefix, inverse and gradient schedule.
- When there are at least 24 chunk/key-head tasks, preparation owns an entire
  shared-key group. QK and KK are computed once for that group. K and KK remain
  read-only aliases of the leader's storage throughout backward, avoiding copies.
  Raw QK temporarily occupies the leader's incoming-state slot, which stage 1
  overwrites only after all preparation finishes. Workspace size is unchanged.

The shared-gram experiment with copied K/KK gave little improvement. Retaining
read-only aliases reduces traffic; it yields only modest cached-path gains.
This is mainly a **recomputation improvement**, not a solution to every cached
backward bottleneck. Native accumulation still uses FP32 Cube operations.

Paired measurements against rebased snapshot **f33546c** (original **2b6bd11**),
identical inputs in one process, alternating old/new order, three warmup pairs
and eleven measured pairs; synchronized wall time includes allocations:

| Global Hv / Hqk | TP | Tokens | Snapshot (ms) | Current (ms) |
| --- | --- | --- | --- | --- |
| 64 / 16 | 1 | 16384 | 80.359 | 45.188 |
| 64 / 16 | 2 | 16384 | 42.512 | 24.652 |
| 48 / 16 | 1 | 16384 | 61.123 | 34.012 |
| 48 / 16 | 2 | 16384 | 31.098 | 17.453 |

At 8K/H=64/Hk=16 with both caches, the corresponding comparison is only
20.674 -> 20.244 ms. Do not present the 40–44% uncached improvement as a cached
speedup. The two-times-forward objective remains unmet. The unchanged gradient
and state stages account for much of the remaining time.

The original 107 checks passed after the new solve/panel/prefix work; nine
focused checks passed after shared storage was added. New packed
`[17,0,19]`, H=64/Hk=16 cases exercise the group scheduling and aliases across
all three dtypes and cache modes, retaining the same numerical bounds.
Final validation: **113 tests passed in 205.35 s** with the existing
`max_error <= 2 * reordered_torch_error + 1e-5` assertions, allocation guard,
cache integrity, arbitrary strides, packed sequences and stream checks.

Matched 4K/H=64/Hk=16 uncached hardware traces (stage durations, not counters):

| Stage | Snapshot (ms) | Current (ms) |
| --- | --- | --- |
| Preparation | 12.898 | 4.203 |
| State replay + adjoints | 2.349 | 2.269 |
| Output/WY gradients | 3.989 | 3.953 |
| Assembly | 1.198 | 1.191 |

Preparation is 67% faster; the other stages are essentially unchanged. These
traces do not measure Cube utilization percentages. Raw profiles are under
`build/gdn_mega_bwd/profile/tp1_before` and `tp1_final`. The paired driver,
measurements and final validation log are retained under
`build/gdn_mega_bwd/tp_v7/`. The refreshed 80-case backward sweep reuses the
unchanged forward measurements from v6 and labels their source explicitly.
All 80 refreshed backward medians improve versus the recorded v6 sweep
(current/old latency range 0.553–0.943). These separate-process sweep results
are supplementary to the paired comparisons above. Its heatmaps are under
`benchmark_data/gdn_mega_bwd/tp_v7/`.
The compact comparison and full samples are in
`tests/baselines/gdn_mega_bwd_tp_v7.json`.

## Extensive sweep snapshot before further optimization

`tests/baselines/gdn_mega_bwd_extensive_v6.json` preserves all 80 backward
configurations and the 60 supported forward measurements used by the TP
heatmaps. These ratios measure **full-recomputation backward**, unlike the
cached v6 paired benchmark. Both use synchronized wall time; backward includes
allocation, forward uses preallocated buffers. TP communication is excluded.
The plotting code can regenerate the heatmap from the snapshot's `results`
list. No kernel changes were made after the 107-test validation below.

## Pipelined backward and direct corrected-value derivatives (v6)

The native FP16/BF16 path now overlaps independent Cube/vector phases within
four launches. Forward and cache contracts are unchanged. The FP32 input path
retains its compensated mathematical schedule. All numerical work remains PTO.
Final validation: **107 tests passed in 188.16 s**, with unchanged per-tensor
`max_error <= 2 * reordered_torch_error + 1e-5` bounds, allocation guard,
cache integrity, packed tails, cancellation, beta=0 and stream coverage.

Paired comparison against commit **300100c**, H=16/Hk=2, FP16 inputs, both
forward caches: identical inputs in one process, alternating old/new order,
three warmup pairs and 31 measured pairs. Times include allocation/enqueue
and device synchronization:

| Tokens | Committed fused baseline | Current pipeline | Time reduction |
| --- | --- | --- | --- |
| 2048 | 2.783 ms | 1.783 ms | 35.9% |
| 8192 | 9.877 ms | 6.038 ms | 38.9% |

The **5 ms target is not reached**. Raw samples, source hashes, profile paths,
validation and the paired-driver hash are in
`tests/baselines/gdn_mega_bwd_pipeline_v6.json`. The driver is retained at
`build/gdn_mega_bwd/pipeline_ab.py`; it loads the original content-addressed
kernel library alongside the current one. Standalone 8K timing was 5.930 ms;
prefer the paired comparison when reporting the speedup.

### Remove products before scheduling the remaining work

For one chunk, let `A=(I+L)^-1`, `B=diag(beta)`, `E=diag(exp(G))`, and `H`
be its incoming state. Corrected values satisfy `Z=A(BV-BEKH)`. Given the
complete corrected-value gradient `D=dZ`, compute:

```text
dVB = A.T @ D
dKB = -dVB @ H.T
dL  = -dVB @ Z.T       # retain the strictly lower triangle
```

This avoids materializing `dA` and applying the two-sided inverse derivative.
It removes four output-stage products. Forming `dKK+dKK.T` before multiplying
by K removes one more: **14 -> 9 Cube products per chunk/head** in stage 2.
The beta derivative retains its general formula without dividing by beta.
Independent vector epilogues run alongside these products.

### Scheduling and scalar work

- `pipeline.h` explicitly groups independent Cube and vector operations.
  Full joins remain where a vector consumes the other vector's stripe.
  Elsewhere Cube releases vectors early while waiting for both producers;
  alternating event epochs prevent a fast subcore from reusing a pending flag.
- Cached preparation uses six phases. KK overlaps gate preparation, inverse
  completion overlaps RHS weighting, and QK overlaps inverse transpose.
- State replay weights Z instead of K before `K.T @ Z`; reverse propagation
  weights the product `K @ dH`. Residuals, decay and copies overlap Cube work.
- Native Cube operand loads are queued together, with dependency-specific
  MTE2/MTE1/M/FIX events. FP32 compensated products keep the original path.
- Three-argument integer TCI generates causal indices on vectors. Missing
  initial/final states use tiled zero packing. Native gate gradients use a
  seven-step FP32 vector reverse scan across transposed groups of eight heads;
  FP32 inputs retain their compensated scan.

An intermediate build hit one UB bounds exception late in a full suite;
its isolated case passed. Explicit vector mask initialization at kernel entry
and restoration after causal-index conversion were added. This is consistent
with stale mask state, but the original fault's cause was not conclusively
established. Two subsequent complete suites passed, including the final build.
Do not assume mask state survives arbitrary low-level vector primitives.

### New hardware counters and remaining limits

Hardware Roofline replay at 8K, both caches, 1800 MHz:

| Stage | Time (ms) | Cube active ratio on active blocks | Mean Cube cross-core wait (ms) |
| --- | --- | --- | --- |
| Preparation + local gradients | 1.458 | 20.9% | 0.607 |
| Forward + reverse state scans | 1.530 | 19.4% | 0.574 |
| Output + WY gradients | 1.949 | 22.9% | 0.528 |
| Assembly | 0.621 | 0% (vector work) | 0.506 |

Output/WY drops from 4.010 to 1.949 ms and mean Cube waiting from 1.598 to
0.528 ms. Its GM/on-chip traffic falls from 5.70 to 5.07 GiB. This includes
L2-served transfers, not just HBM. Sum **all** wait-ID columns: new phases
use alternating IDs 2/3 and 4/5 in addition to legacy 0/1. Active-cycle ratios
are not peak FLOP/s, and overlapping counters are not additive fractions.

The state scan still uses only 16 of 24 blocks for a single sequence/H=16.
Many intermediates still travel through GM; workspace stays **816.5 MiB** at
8K/H=16, with **65.25 MiB** retained forward caches. Remaining candidates are
retaining shared matrix operands on chip, splitting state work across value
columns to improve occupancy, and avoiding replay by saving chunk states.
The latter changes forward storage/performance and has not been implemented.

## Hardware profiling after 300100c

Commit **300100c** preserves the validated four-stage fused kernel and v5
measurements. `msprof op --aic-metrics=Roofline` collected all four kernels
at 8K/H=16/Hk=2, FP16 inputs, both forward caches, 1800 MHz on Ascend910B2.
This is counter collection with profiler replay, not a new wall-time benchmark.
The profiled stage times sum to 9.177 ms; the saved synchronized wall baseline
is 9.922 ms. Source hashes match the committed runtime.

| Stage | Time (ms) | Mean reported Cube active ratio, active blocks | Active blocks / 24 | Mean Cube cross-core wait / block time |
| --- | --- | --- | --- | --- |
| Preparation + local gradients | 2.161 | 14.1% | 24 | 51.8% |
| Forward + reverse state scans | 2.226 | 13.3% | 16 | 53.1% |
| Output + WY gradients | 4.010 | 17.3% | 24 | 40.8% |
| Assembly | 0.780 | 0% (vector work) | 24 | 93.7% |

Cube ratios count active pipeline cycles, not achieved peak FLOP/s. Waiting
and other pipe counters can overlap and must not be added as disjoint fractions.
For the state stage, averaging over all 24 blocks gives only 8.9% Cube activity;
13.3% above excludes the eight idle blocks.

The primary issue is serialized execution, not saturation of Cube arithmetic.
`Ops::call` ends every primitive with a two-vector/Cube `join()`. `load`,
`store` and `matmul` additionally place `PIPE_ALL` barriers around DMA, operand
extraction, multiply and stores. Cube waits for vectors for 1.10 / 1.16 / 1.60 ms
on average in stages 0 / 1 / 2. Vector subcores spend much of those stages
waiting for Cube in return. These observations identify the opportunity; they
do not imply all reported wait time can simply be removed.

Output/WY performs approximately **5.70 GiB of GM/on-chip transfers**:
1792 MiB GM→L1, 768 MiB L0C→GM, 2188.5 MiB GM→UB and 1092.5 MiB UB→GM.
This includes L2-served traffic, so it is not HBM traffic or proof of HBM
bandwidth saturation. It does demonstrate repeated materialization of matrix
intermediates. Retaining operands/intermediates and overlapping DMA with math
is more promising than removing another host launch.

Scalar findings: stage 3's vector0 has a 41.1% reported scalar ratio versus
5.9% on vector1. The compensated 128-element reverse gate scan runs serially
on vector0; the two-argument PTO `TCI` used by `causal_mask` also expands to
scalar stores in CANN 9.1's `pto/npu/a2a3/TCI.hpp`. The three-argument TCI
provides a vector implementation. Missing h0/dhT use the rowwise packing
fallback because their descriptors have zero strides. Conversely, FP32-only
interval loops/local dQ reconstruction and inverse recomputation are **not**
executed in this cached FP16 profile; optimize the actual path first.

Priority for the 5 ms target:

1. Pipeline stage 2's independent chunk/head work with alternating UB/L1
   buffers; retain common operands and use producer/consumer events at real
   dependencies. Batch consecutive operations on the same engine. Do not
   blindly delete cross-core waits: vector broadcasts/reductions cross stripes.
2. Apply the same scheduling to preparation. Replace scalar TCI and use a
   compensated vector tree for gate scans, retaining the current error bounds.
3. Address state-stage occupancy with independent value-column tiles when
   sequence/head count is too small, or save incoming chunk states to skip
   replay (extra forward stores and 64 MiB at 8K/H=16). Measure these tradeoffs.

Even eliminating the entire 0.780 ms assembly stage would leave over 8 ms
in this profile. Scalar cleanup alone cannot reach 5 ms. The counters show
headroom, but do not establish that a particular pipeline will meet the goal.
Raw CSV locations/hashes and per-stage summaries are retained in
`tests/baselines/gdn_mega_bwd_profile_v5.json`.

The benchmark now accepts `--profile-once` to launch one forward for caches
and one backward, skipping timing graphs and cache comparison runs. For
hardware counters, use `msprof op --kernel-name="*gdn_bwd_stage*" --launch-count=4
--aic-metrics=Roofline --output=build/gdn_mega_bwd/profile/roofline_once python
tests/benchmark_gdn_mega_bwd.py --tokens 8192 --heads 16 --key-heads 2
--inverse-cache --gate-cache --profile-once` after sourcing CANN and setting
`PYTHONPATH=python:.`. Numerical tests remain separate from profiler replay.

Simulator note: a C=128/H=16/core0 output/WY capture reached its 10-minute
application limit during stage 2. It saved a **partial** instruction trace under
`build/gdn_mega_bwd/profile/simulator_stage2_v5/`; it is not a complete kernel
measurement and is not used for the hardware percentages above. Early products
show `WAIT_FLAG_DEVI` stalls. The release binary lacks source-line debug info.
The initial runner had still executed four forward/cache-check calls; the final
`--profile-once` path performs only one cache-producing forward and one backward.
For a future simulator run, prepend CANN's
`tools/simulator/Ascend910B2/lib` to `LD_LIBRARY_PATH` and use
`msprof op simulator --kernel-name="_Z13gdn_bwd_stageILi2EEv4Plan"
--launch-count=1 --aic-metrics=PipeUtilization --core-id=0 --timeout=10`
with that lean runner. The complete 8K hardware profile is the primary evidence. The lean runner
completed hardware smoke checks at 128 and 8192 tokens with both caches; runtime
source hashes remain identical to the committed 107-test-passing kernel.

## Four-stage fusion after f686c16

Commit **f686c16** preserves the validated seven-stage inverse/gate-cache
implementation and its v3/v4 raw results. Fusion leaves forward code and cache
contracts unchanged. The backward now submits four stages per group of up to
64 packed sequences, independent of token/chunk count:

1. Preparation and local gradients, parallel across chunk/head tasks. Local
   gradients depend on Q/K/dOut and gates, not the replayed state.
2. Forward state/Z replay, then reverse state propagation in the same task,
   which owns one sequence/head. No cross-task state dependence is introduced.
3. Output/WY gradients, parallel across individual chunk/head tasks.
4. Gate scans and final assembly. Tasks own whole shared-key groups; value-head
   groups split only at 32-byte output boundaries. H=16/Hk=2 has two final tasks
   per chunk. Irregular head counts use one task per chunk.

Simply calling the old helpers in fewer launches is insufficient. The fused
implementation also loads dense inputs in 64-row DMA tiles, builds causal masks
with broadcasts/clamps, keeps inverse-adjoint weighting and beta reduction in
UB, accumulates pairs of native matrix products in L0C, transposes gate/beta
outputs in eight-head tiles, and retains shared-head dQ/dK sums in UB until the
output store. No dense product uses vector dot loops. FP32 inputs retain the
compensated product path and its rounding order.

The first four-stage candidate measured 2.805 / 9.890 ms at 2K / 8K, H=16/Hk=2,
with both caches (nine samples). Fusing final assembly with one task per whole
chunk hurt head parallelism, motivating grouped final tasks. Even with grouped
heads and on-chip reductions, the three-stage candidate measured 3.207 / 10.176 ms
(31 samples), so it was rejected in favor of separate assembly. Fewer launches
alone are not a useful optimization when the expensive stages lose parallelism.
These were exploratory runs, not the final benchmark. Workspace remains 816.5 MiB at 8K/H=16; many
intermediates still use GM. There is no whole-kernel on-chip residency claim.

Final 31-sample synchronized wall medians with both caches: **2.807 ms at 2K**
and **9.922 ms at 8K**, H=16/Hk=2. The committed v4 baseline measured
4.320 / 15.799 ms: **35.0% / 37.2% less time**. The final 8K trace confirms
four launches at 2.291 / 2.246 / 4.233 / 0.810 ms. State scans and output/WY
products remain bottlenecks; the 2x-forward target is not reached. The raw
samples, rejected three-stage experiment and source hashes are retained in
`tests/baselines/gdn_mega_bwd_fused_v5.json`. Comparisons are across separate
development runs, so clock/thermal variation can affect the numbers.

Final validation: **107 passed in 191.11 s** (97 hardware, 10 CPU reference).
All six gradients keep the same `2 * reordered_PyTorch_max_error + 1e-5` bound.
The seven-allocation guard, nondefault-stream tests, cache immutability,
packed/empty sequences, irregular heads and FP32 cancellation cases pass.
Only the device library needed rebuilding (7.96 s); host ABI and Python hotpath
are unchanged. Forward numerical code was not changed after f686c16.

## Reusing existing gate and beta outputs

The inverse export below did change forward code; it redirects existing
exchanges to a retained buffer. Gate/beta reuse needs no further forward
numerical changes: `gate_scan.h` already writes the high FP32 chunk prefix
into `g_t`, its low FP32 part into `g_sum`, and FP16 beta into `beta_t`.
All three have physical `[H,T]` layout, despite g_sum's historical logical
`[1,T,H]` allocation. Preserve their lifetime and pass
`gate_cache=(g_t,g_sum,beta_t)` to backward. Never share these buffers with
a subsequent forward until their backward consumer has completed.

PTO loads these rows contiguously, skipping raw gate/beta packing and the
prefix scan. It forms exp(G) as exp(high)*exp(low), and intervals from the
differences of both parts, matching the forward's compensated representation.
Tail prefixes repeat the last valid value, with padded beta zero. Exponentials,
pairwise decays, KK, W/U, chunk states and Z remain recomputed. Both caches
are optional and read-only; the seven-stage schedule and seven allocations
remain unchanged. Retaining gates/beta adds 0.3125 / 1.25 MiB at 2K / 8K,
H=16, with no new forward writes. The original h0 is already consumed;
retaining chunk-boundary states would be a separate 64 MiB cache at 8K/H=16.
There is no normalization rstd in this forward; normalization is upstream.

Validation: **107 passed in 185.35 s** (97 hardware, 10 CPU reference),
including 36 saved-forward cases covering inverse-only, gates-only and both.
These exercise all seven supported head counts, packed tails, empty sequences,
weak decay, repeated keys, strided inputs and the 64-sequence grouping boundary.
All six gradients retain the `2 * reordered_PyTorch_max_error + 1e-5` bound;
read-only cache checks and the seven-allocation/no-Torch-computation guard pass.
Forward numerical code was unchanged for this addition; the preceding inverse
export had already passed all 266 forward checks.

Consecutive 31-sample runs at H=16/Hk=2 measured inverse-only backward at
**4.670 / 17.194 ms** for 2K / 8K, and inverse+gates at **4.320 / 15.799 ms**
(7.5% / 8.1% lower wall time). The latter 8K trace has seven launches, with
preparation taking 5.733 ms. Separate processes can have different clock/thermal
state; raw samples, profile and hashes are retained in
`tests/baselines/gdn_mega_bwd_gate_cache_v4.json`. Forward timing in this record
still compares the existing inverse-cache switch; gate retention itself changes
no forward work. The earlier v2/v3 records remain untouched.

## Optional forward inverse cache

Commit **c03edb2** preserves the validated Cube baseline before this work.
The optional `inverse_cache` argument is shared by `pto_gdn_mega` and
`backward`. It is a dedicated contiguous FP32 `[num_chunks,H,128,128]` tensor
in packed sequence/chunk order; hold it with the original inputs until
backward completes. The backward does not modify it. Caller allocation and
the seven-stage backward launch structure are unchanged.

The forward's final 64x64 join deliberately stops short of a full inverse.
It holds D0=M00^-1, D1=M11^-1 and P=-D1*L10 for M=I+L. Exporting a tensor
called “inverse” without accounting for this would produce wrong gradients.
The cache contract is `[D0,0;P,D1]`; backward computes only the missing block
P*D0 on Cube, then transposes the result for its existing WY equations.
It skips weighted-triangular construction and back substitution. It still
recomputes gates, KK, W/U, forward states and Z.

The first implementation mirrored the existing exchange stores into the
cache and passed all 344 then-existing forward/backward/reference checks.
It added about 1.5–2% forward device time. The final approach redirects those
existing exchanges into the cache and reloads the same data from it; this
adds no matrix products, stores or launches. Next-chunk preparation writes
a different cache record from the current Cube solve. Existing cross-core
flags order producers/consumers. The final lower-left join stays factored.

At H=16, Hk=2, D=128, the final 31-trial paired graph measurements (20 forward
calls per trial) were 345.271 → 348.904 us at 2K and 1268.518 → 1280.014 us at
8K with caching enabled: **1.05% / 0.91% overhead**, not literally zero.
Synchronized backward wall time was **4.529 ms / 16.641 ms**, versus the
frozen Cube baseline's 5.534 / 20.481 ms. The 8K preparation stage fell from
10.977 to 7.107 ms; inversion was only part of its cost. Forward output and
final state are bit-identical with/without caching in the cache tests and
the 2K/8K benchmark. The cache occupies 16 / 64 MiB respectively, surviving
from forward to backward; backward's original workspace remains 816.5 MiB
at 8K/H=16. Reusing an unrelated opaque scratch tensor is not supported.

Final validation: **349 passed in 293.74 s** (266 forward, 73 backward,
10 CPU-reference checks). Twelve cached-backward cases cover all seven
supported head counts, grouped heads, weak gates, repeated-key cancellation,
packed tails, empty sequences, strided backward inputs, and cache offsets
across a 65-sequence batch. Both forward outputs are checked unchanged with
caching enabled; the cache itself is checked unchanged after backward.
The backward tolerance remains `2 * reordered_PyTorch_max_error + 1e-5`.
The all-head device candidate built in 11.02 s. The existing forward benchmark
builder was reused; the local development binding was rebuilt for only the
GDN entry, avoiding compilation of unrelated device kernels. The validated
device library is also installed in the ignored `build/lib/` location used
by the default local binding.

## Cube baseline (c03edb2)

All dense matrix products now use Cube. FP16/BF16 q/k/v use one native FP32
Cube product; any FP32 q/k/v selects four K=32 Cube products, followed by
compensated addition of their matrix results. The old vector dot-product
implementation is removed. Vector operations still handle packing, masking,
gate scans, triangular back substitution, elementwise arithmetic and reductions.
There remain seven sequence-wide stage launches and no Torch numerical hotpath.

FP32 cancellation required algebraic improvements as well as shorter dot
accumulations. Subtracting rounded large gate prefixes loses interval accuracy,
so this path sums gate intervals directly. dQ is reconstructed in 16-token
subblocks inside each 128-token chunk, reducing cancellation between distant
updates. Since Z=beta*residual and dV/beta=dUpdate, dBeta can be computed as
dot(dUpdate,Z)/beta; this avoids cancellation between WY derivative paths.
The general derivative is retained at beta=0. FP16/BF16 keep the simpler path.

Tests use identical rounded inputs and upstream gradients in a golden recursive
PyTorch run, a reordered run in the input dtype, and the PTO backward. For each
of six gradients, max error must be <= 2 * baseline max error + 1e-5 (the
absolute allowance was explicitly requested). The golden computation is FP32
for FP16/BF16, FP64 for FP32; golden gradients retain that precision because
the kernel returns FP32 gradients. Query/value/upstream inputs use unit-variance
randn; keys remain normalized for a stable delta recurrence. Both references
use autograd. FP64 tests independently verify equivalence of the reordered
equations and their gradients. Mixed FP32 gates/state are covered too.

Development notes: unpartitioned Cube products passed all FP16/BF16 cases even
without the absolute allowance. FP32 exposed cancellation; changing tolerances
alone was insufficient. K=8 extraction is only supported in compatible L0
layouts and initially required explicit transposes. Local dQ and direct dBeta
made that experiment unnecessary; the final K=32 path needs no GM transposes.
The workspace now reserves 45 MiB of block scratch (four additional partial
products), totaling 816.5 MiB at 8K tokens / 16 heads. Forward caches are still
unused, and sequence-wide intermediate matrices still occupy GM.

Validation: **71 passed in 109.04 s** with cached builds (61 hardware, 10 CPU).
Final FP16 timings at H=16, Hk=2, D=128: **5.534 ms at 2K** and **20.481 ms
at 8K**, versus 75.378 / 293.984 ms in v1: **13.62x / 14.35x faster**.
Medians include allocation, recomputation, enqueue and synchronization, with
one warmup and five samples. At 8K, the seven hardware stage times were
10.977, 0.949, 0.691, 1.146, 1.882, 3.500 and 0.997 ms. Local/output gradient
stages together dropped from 275.42 ms to 2.57 ms. Preparation/recomputation
now dominates; fusion and inverse preparation are the next profiling targets.
FP32's accuracy path is correctness-tested, but these timing improvements
must not be attributed to it. Raw samples, profile timings, exact config,
validation criterion and source hashes are in
`tests/baselines/gdn_mega_bwd_cube_v2.json`; earlier artifacts remain unchanged.

## Archived sequence-wide v1 implementation

### 16-head performance diagnosis (2026-11-04)

Measured B=1, H=16, Hk=2, D=128, FP16 inputs, zero initial state, no final-state gradient: 2,048 tokens **75.378 ms**, 8,192 tokens **293.984 ms**, five samples after warmup. Enqueue medians were 0.339 and 0.407 ms. Raw samples and the 8K hardware trace timings are preserved in `tests/baselines/gdn_mega_bwd_staged_v1_heads16.json`; the benchmark now accepts `--heads` and `--key-heads`. The user's approximate forward times were 1 and 2.5 ms, but forward was not rerun and its exact key-head count/timing scope was unspecified.

Seven-stage device timings at 8K were 11.031, 1.021, 77.882, 1.253, 197.539, 3.695 and 1.082 ms. Local and output gradients (stages 2 and 4) consume **93.8%** of device time. Their `score_product` helper computes matrix products on vector cores with a seven-level compensated TwoSum reduction, repeated operand loads, transposes and barriers. This preserves strict cancellation-sensitive gradient accuracy but is much slower than the Cube products used by the forward. The two state scans total only 2.274 ms. Neither seven launch overheads nor state recurrence explains the slowdown. Next priority is accurate Cube implementations of these products, preserving the existing gradient tests, followed by reducing GM temporaries and synchronization. Forward recomputation adds work but is a small fraction of this measured runtime.

`kernel.cpp` launches seven mixed cube/vector kernels: prepare, forward replay, local gradients, reverse state propagation, output derivatives, WY derivatives, and final assembly. Independent stages distribute chunk/head tasks across blocks; state stages loop over chunks on device. The host has no token-chunk loop. At most 64 sequence descriptors fit in one inline plan; larger packed batches submit groups of seven stages and reuse the same workspace. A hardware trace for a 1,024-token sequence contains exactly seven stage launches.

Device primitives moved to `primitives.h`. `device_ops.h` coordinates each cube/vector block. Both vector subcores split disjoint rows/stripes; weighted triangular preparation and multiply/reduction chains keep intermediate products in UB. General matrix temporaries still use per-block GM scratch inside the stages. This establishes the FLA-like scheduling structure while preserving the independently tested FP32 equations.

State buffers and other stage outputs have explicit lifetimes in `layout.h`. Twelve matrix slots and seven vector slots are retained per chunk/head. W is reused for dW, U for corrected values, Q for dQ, dOut for dV, and state adjoints for dK. Per-block scratch has 24 temporary matrix slots (WY peaks at 19 matrices plus vectors) and two state-adjoint slots; 24 blocks reserve 39 MiB. Workspace at 8K tokens/H=16 is 810.5 MiB, compared with 162 MiB in the rough baseline. The additional sequence-wide storage trades memory for chunk parallelism.

### Findings during stage conversion

- A launch-only conversion passed all 57 original checks but was slower at short lengths: 13.77 / 27.00 / 80.00 ms for 128 / 512 / 2048 tokens. Compensated dot products lost their previous row-level distribution across blocks.
- The trace attributed most time to stages 2 and 4. Splitting these products across both vector subcores reduced timings to 7.27 / 14.08 / 41.27 ms. The 128-token case still trails the 5.00 ms rough baseline; long-sequence throughput improves.
- A two-vector data hazard initially corrupted one row: signalling the cube release before waiting for both vectors let one vector read its peer's unfinished rows. `join()` now waits for both vector producers before releasing either vector to the next operation.
- Grouped-head reduction reuses one scaled-query temporary, so scratch use is bounded even when many value heads share one key head.
- Pointer values in the inline plan use `uint64_t`; the Ascend compiler rejects converting generic pointer fields directly into GM pointers inside device functions.
- The host launcher receives a device entry-function pointer. Device-only edits now rebuild only the bisheng library; they do not recompile Torch headers.
- Expanded coverage includes non-power-of-two head counts, packed tails, a 65-sequence metadata batch, and a 64-to-1 shared-head case. Final validation: **66 passed in 70.11 s**, including 6.00 s device and 11.60 s native builds. This comprises 58 hardware checks and 8 CPU-reference checks, at unchanged tolerances. Native-only and device-only builds are cached independently.

### Final comparison record

`tests/baselines/gdn_mega_bwd_staged_v1.json` contains final raw samples, source hashes, hardware stage timings and configuration; v0 remains unchanged.

| Tokens | Rough native v0 (ms) | Seven stages v1 (ms) | Speedup |
| --- | --- | --- | --- |
| 128 | 5.005 | 7.279 | 0.69x |
| 512 | 19.561 | 14.033 | 1.39x |
| 2048 | 77.748 | 41.082 | 1.89x |

At 1,024 tokens, wall time was 20.893 ms and the trace contained exactly seven device launches. Stage durations in order were 0.680, 0.117, 5.446, 0.160, 13.736, 0.258 and 0.163 ms. Enqueue medians were about 0.29–0.31 ms. These timings include no model forward pass.

The benchmark's `--profile DIR` option records a hardware trace and `stages.json`. It is not a simulator run. Stage labels are `gdn_bwd_stage<0>` through `<6>`. Keep the frozen v0 JSON unchanged; newer measurements belong in a separate artifact.

## Archived rough baseline (150fed0)

The descriptions and measurements in this section refer to commit 150fed0, before the seven-stage implementation.

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
