# GDN mega optimization notebook

Target: 3x faster than commit `10748eb`, with correctness preserved.
Hardware: Ascend910B2, 24 cube / 48 vector cores. CANN 9.1.0,
PyTorch / torch-npu 2.10.0. Container: `gdn-mega-opt`.

## Reproduction

```sh
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh timeout 120 python -m pytest -q tests/test_gdn_mega.py
bash scripts/gdn_mega_docker.sh timeout 120 python benchmarking/gdn_mega.py --output benchmark_data/gdn_mega/current.json
```

Build options: `GDN_MEGA_ONLY=ON` excludes unrelated kernels and bindings.
`GDN_MEGA_HEAD=16` compiles one head specialization for experiments (~12s);
clear it for full acceptance tests. Full compilation is ~35s, initial host
binding compilation ~44s. CMake reconfiguration adds ~10s; ordinary source
edits do not reconfigure. Fixed pre-existing CANN include-path loss on
reconfiguration and incorrectly shell-expanded `$ORIGIN` linker flags.

The benchmark allocates once and uses device graphs with 20 calls, 15 event
timing trials, five correctness/warmup calls. References run on CPU.
Warm-cache timing; report output AND state errors. Current acceptance requires
finite values and relative RMSE < 0.005 for both output and state. This aggregate
budget accounts for FP16 intermediate matrix products versus the FP32 reference;
the former 0.01 absolute-error shortcut and 5% RMSE budget were removed.
Actual observed errors are reported alongside the threshold.

## Baseline

H=16, Hg=4, D=C=128; no initial state; normalized q/k, random value and beta,
logsigmoid gates. All 7 original tests pass. Expanded suite also passes all
17 tests, adding packed ragged sequences, lengths 1/127/129/257, initial state,
weak decay gates, and five consecutive calls.

| T | Baseline us | Parallel scan us | Scan + exact doubling us |
|---|---:|---:|---:|
| 128 | 104.892 | 80.348 | 76.662 |
| 256 | 141.561 | 108.560 | 104.495 |
| 1024 | 306.957 | 269.530 | 234.732 |
| 4096 | 863.501 | 827.078 | 759.152 |

Doubling measurements include optional stage-counter instrumentation.
Baseline binary and raw JSON retained under ignored `benchmark_data/gdn_mega/`.
Baseline output relative RMSE ~0.00038; final-state relative RMSE ~0.00032.

## Findings / experiments

- Original gate prefix sum has 128 serial row operations with extra row copies,
  then a global barrier and separate transpose pass. Seven-step parallel scan
  fuses both transposes and removes one global barrier. Verified on H=16
  expanded suite (14 selected tests); full head coverage remains pending.
- Original inverse constructs I and zero using matmuls and repeatedly copies
  X through X@I. Exact geometric doubling uses A^128=0, 14 matmuls, and retains
  the running X in FP32 L0C; no convergence assumption or truncation.
- Stage instrumentation (`GDN_STAGE_TIMING`) stores seven core-0 timestamps
  in the otherwise unused A_inv_f32 workspace. Differences include stage
  barriers; optional debug path, not production timing.
- At T=4096 after scan, stage tick counts: gates 917, KKT 5400, inverse 14150,
  WY 3389, recurrent state 10116, output 7331. Inverse is largest but not enough
  alone to deliver 3x. Need reduce data transfers, synchronization, and state
  critical path too.
- Two consecutive global barriers remain between inverse and WY (old cast
  stage removed but barrier retained).
- Compiler ASC language defaults to automatic synchronization in addition
  to explicit set/wait/barriers. Experiment with disabling automatic sync
  requires full correctness validation: some vector dependencies may currently
  rely on compiler-inserted synchronization.
- Potential next algorithm: block triangular inverse with TINSERT Acc->Mat,
  performing only nonzero recursive blocks instead of dense sparse matmuls.
- chunk_h serializes W@S, vector residual, Ktilde^T@Vnew, vector state update,
  with GM roundtrips on every chunk. State could remain resident in L1; vector
  preparation can be moved ahead of the recurrent dependency.

## First full acceptance

All 21 tests pass, including H=8/12/16/24/32/48/64 and packed tails.
Full-dispatch build timings: 93.739 / 125.622 / 258.306 / 779.279 us for
T=128/256/1024/4096. These are the honest production comparison: the
single-head experimental binary is ~20 us faster due to code/dispatch layout.
Future changes must measure full supported dispatch too. Investigate separate
kernel entrypoints per head, chosen by the host, to avoid putting all seven
large specializations in a single device entrypoint.

## Ongoing experiments after 6453a98

- Shared KKT Gram computation once per key head, concurrent with gate scan;
  vector gating uses all cores. Tests pass for H16 including ragged/weak gates.
- Blocked inverse (two 64x64 diagonal inverses + one offdiagonal join) is faster
  than four 32x32 blocks: fewer tiny matrix instructions outweigh extra FLOPs.
- WY input-side scaling overlaps vector preparation with cube inverse work.
  Numerical output relative RMSE rises slightly from .00038 to .000435.
- Cube-only state recurrence keeps state in L1. Uses negative W, U@I + (-W)@S,
  then S@decayI + Ktilde.T@Vnew. Prep stores scaled K in A_inv_f32 scratch and
  decay matrices temporarily in s; cube overwrites s with actual incoming states.
  All 14 selected H16 tests pass. No tolerance relaxation. Timings with all
  experimental macros: 54.288 / 78.317 / 187.010 / 623.784 us. Long workload
  stage ticks: gates1082, KKT3541, inverse8756, WY3100, state7322, output6926.
- Current experiment CMake config: HEAD=16, MANUAL_SYNC=OFF,
  DEFINITIONS=GDN_SHARED_GRAMS;GDN_BLOCKED_INVERSE;GDN_INVERSE_BASE64;GDN_WY_INPUTS;GDN_CUBE_STATE;GDN_STAGE_TIMING.
  These are experimental flags, not defaults yet. Full-head acceptance pending.
- Simulator profile running with kernel-name `gdn_mega_specialized<16>`, using
  `benchmarking/gdn_mega.py --profile-once --lengths 128`; output in
  benchmark_data/gdn_mega/profile_state, process log /tmp/gdn-simulator.log.

## Fusion checkpoint

Full production acceptance after shared-output work: 23 tests pass, all heads,
T up to 1024 with weak gates and initial state. H16/Hg4 latencies
52.708 / 75.084 / 169.278 / 553.351 us. Simulator profile_output also passes
numerics after creating masks/identity on CPU and checking raw output on CPU
(msprof's kernel filter otherwise skips NPU setup/check operations).
Traceviz plot: benchmark_data/gdn_mega/pipeline.png; matched profile root
profile_output/OPPROF_20261022133337_UMHOVJSARRABVLPS. Trace reveals many serial
MTE2 and FIX transfers, substantial wait time, low cube utilization.
Disabling compiler automatic sync made no measurable difference; keep OFF for
GDN_MEGA_MANUAL_SYNC normally. Static full-tile DMA descriptors and overlapping
X/power matmuls in the blocked inverse gave negligible measured changes.

New optional experiment GDN_FUSED_PIPELINE (fused_pipeline.cpp):
- raw KK stored in v_new scratch (NOT A_inv_f32, which scaled K would overwrite
  before other grouped heads consume raw KK); raw QK in second half-span of
  A_inv_f32, scaled K in first half-span.
- one persistent worker per (sequence, value head), vector subblocks prepare
  all chunk factors together, sharing decay matrix coefficients across KK/QK.
- ready flags 5/6 and free flags 7/8 make a two-credit producer/consumer loop.
  Cube seeds both free flags, waits ready, replenishes free immediately; vector
  drains two leftover free signals at end of each head. No global barriers in
  the recurrent loop. Buffers are per chunk, so ahead-of-consumer writes do not
  alias older chunks.
- cube retains inverse, W/U/residual, state, and output accumulators on chip,
  avoiding intermediate GM stores/loads. L1 state lives at 8*32768, other
  operands at 9..13*32768; inverse scratch is below 6*32768, total <512KiB.
- direct fused path returns after initial gate scan/shared-Gram barrier and
  skips the staged path. Currently compiling/testing H16 with this flag.
- source defaults now enable all validated staged improvements unless
  GDN_LEGACY_STAGES is defined. Fused pipeline is NOT default yet.

## Paused checkpoint and commit discipline

Commit every measured improvement once its correctness tests pass, and record
the commit ID, validation scope, and measurements in success.md immediately.
Do not accumulate further validated gains without recording them.

Optional fused pipeline with contiguous head-major intermediates passed 16
H=16 tests: 39.185 / 51.217 / 124.193 / 411.824 us at T=128/256/1024/4096.
The two free-slot credits are replenished only AFTER the cube loads both slot
matrices into L1. This supersedes the earlier immediate-replenishment note.
The default staged implementation passed the full 23-test suite.

Optimization is paused by the user. The commit checkpoint excludes the
adaptive-inverse experiment, which remains in the working tree. Its latest
build failed with two tile-constructor argument errors; two new correlated-key
tests remain unrun. Do not treat this experiment as a validated improvement.
CMake cache currently selects HEAD=16 and definitions
GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE;GDN_FUSED_TIMING. The existing shared
library is from the preceding exact fused timing build, not the failed build.
On resumption, inspect /tmp/gdn-adaptive-build.log inside gdn-mega-opt.
The adaptive helper adds max_terms and four conditional blocks; their changes
are intentionally unstaged relative to the exact-inverse checkpoint.

## Resumed: bounded adaptive inverse

User resumed optimization. Fixed the two dynamic-tile constructors (one
dynamic dimension takes one constructor argument). Full-head acceptance:

```sh
bash scripts/gdn_mega_docker.sh bash -lc 'cmake -S . -B build -DGDN_MEGA_HEAD= -DGDN_MEGA_DEFINITIONS="GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE" && bash scripts/build_gdn_mega.sh'
bash scripts/gdn_mega_docker.sh timeout 120 python -m pytest -q tests/test_gdn_mega.py
bash scripts/gdn_mega_docker.sh timeout 120 python benchmarking/gdn_mega.py --output benchmark_data/gdn_mega/adaptive_acceptance.json
```

25 tests pass. H16/Hg4 latencies: 39.143 / 49.481 / 112.795 / 372.280 us.
During single-head testing, use `-k 'not head_configs or H16'`; matching `16`
alone accidentally includes H64-Hg16, unsupported by a HEAD=16 binary.
Counter build: inverse 6470 ticks vs exact 8754, rest ~9572 at T4096; vector
preparation ~11427 ticks. Cube remains critical. Full uninstrumented build
is ~3us faster. Next test: full128 polynomial avoids two block inversions and
join overhead when only a few terms are needed; retain blocked fallback for
high norm.

Full128 adaptive polynomial experiment rejected: all 18 selected tests passed,
but latencies 41.222 / 51.750 / 117.201 / 383.706 us were slower than the
blocked adaptive inverse. Larger matrix arithmetic outweighed reduced dispatch.
Next: overlap chunk RHS and output operand DMA with inverse computation;
current PIPE_ALL barriers unnecessarily include independent MTE2 transfers.

Direct residual polynomial (no dense inverse/W/U) passed 18 selected tests,
but initial-state timings 41.805 / 53.032 / 119.838 / 392.198 us are slower.
More full128 matmuls outweigh avoided smaller inverse operations. Rejected;
source snapshot in benchmark_data/gdn_mega/direct_residual.cpp.
Initial-state baseline now measured explicitly: 107.532 / 145.567 / 309.691 /
863.765 us. Next exact algebraic change concatenates shared-factor products:
[U,-W] = inverse [betaV,-beta exp(g)K]; [O;Snext] = [Qexp;decayI]S + [Att;Kt^T]R.
Both fit physical L0 buffers (128KiB L0C, 64KiB L0A/L0B).

Joint output/state products pass 18 selected H16 tests and reduce T4096 to
359.178us WITH initial state. A first attempt also concatenated two L0B tiles
by simple pointer offset; this is invalid because L0B is row-major in its
outer blocks. Removed that branch. Concatenate W/U in column-major L1 instead
and use one full-width TMOV to correctly interleave L0B. Current build flags
GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE;GDN_JOINT_PRODUCTS, HEAD=16.

Correctly interleaved joint W/U GEMM passes 18 tests but no meaningful gain:
358.615us vs359.178us atT4096, with short cases slightly slower. Removed it.
New exact simplification: vector emits -A directly; cube inverts its diagonal
blocks in place using +I, then consumes the untouched offdiagonal -A in the
join. Removes a full128 negative-identity GEMM and two L1 copies per chunk.

Prefetch operands into L1 buffers0/1/9..12 while in-place inverse uses2/5.
Inverse helper synchronizes only MTE1/M/FIX; final PIPE_ALL waits for operand
loads before padding and releasing slot credit. 18 tests pass, initial-state
latencies37.458/45.992/96.432/299.105us. GDN_PREFETCH requires GDN_NEGATIVE_A.
Next algebraic reduction: exp(g_i-g_j) can be computed as the outer product
exp(g_i-center)*exp(center-g_j). Check range <=120 to keep individual factors
finite; otherwise retain original matrix exponential. Reduces vector exp work
from8192 elements per stripe to192 without dropping matrix entries.

Factored centered decay passes 24 selected tests (adding gate_scale4 fallback),
but does not materially improve runtime:38.129/46.269/95.580/298.966us. Dropped
it; keep exp(g_i-g_j) for its simpler numerical behavior. Extra extreme-gate
tests retained. Next remove dead initial-state identity multiply and prepare
I-A in vector registers, replacing one identity GEMM per diagonal inverse.

Vector inverse seed experiment passes24tests but regresses to320.483us atT4096:
extra vector/DMA work outweighs removed cube identity products. Removed it.
Kept exact removal of dead initial-state*I accumulator copy. Tightening all
reference checks to0.5% relativeRMSE (from5%) and removing1e-2 absolute-error
early exit; no more near-zero-output false passes. Next inverse version uses
only the specific producer/consumer edges per operation, avoiding redundant
MTE1/M/FIX round trips while preserving operand lifetimes.

Strict checks found packed weak-decay short-tail errors in some heads. Full64
terms fixes them; CPU norms of the actual half A are2.7–3.6, requiring64.
Failing heads vary between runs. CANN TSYNC_CVID publishes scalar GM stores
with volatile,dcci,dsb(DSB_DDR); our producer lacked the dsb. Added producer
and consumer DDR fences and volatile accesses. All31 full-head tests now
pass, including50 repeats of the packed regression. Tight per-operation
inverse synchronization was not faster and was reverted.

Full acceptance command flags:GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE;
GDN_JOINT_PRODUCTS;GDN_NEGATIVE_A;GDN_PREFETCH, HEAD empty. Initial-state
latencies39.651/48.429/98.719/301.066us. No-state38.417/47.582/97.472/299.847us.
Next exact transfer reuse: after W*S, L0B already contains S; the joint output
product need not load it again. For a half decay matrix that is exactly zero,
its L1/L0A copies can also be reused across chunks (not a state-zero assumption).

Operand reuse passes25selectedstrict tests. Only L1 decay is reusable across
chunks; L0A bottom half gets overwritten by Ktilde^T in second joint product.
Caught that mistake in initial attempt and fixed it. S in L0B survives from
W*S to first joint product. Added finite/zero-decay alternation regression.
Latencieswithinitial37.468/46.568/95.733/292.657us. Next interleave independent
64x64 diagonal inverses, preserving arithmetic but amortizing pipeline flags.

Paired diagonal inverses pass25strict tests but save only~2us at4096 and
regress128 startup; reverted extra scheduling code. Next vector trim caches
exp(g) once for W/Q/decay and constructs decayI directly inhalf, avoiding
float conversions of an identity matrix. Exact-zero decay uses vector fill.


Latest scheduling reductions: reuse exp(g) without factoring exp(g_i-g_j);
omit S*0 when the stored FP16 decay is exactly zero; seed first vector credits
before cube setup. Both initial-state and no-state H16 benchmark sweeps now
exceed 3x at all four standard lengths. All32 all-head/default tests pass. Production medians regress slightly versus
the single-head build (T128 initial36.354us), so3x is not universal. New user
target: another2x from this production checkpoint through structural changes.
Default promotion enables fused/adaptive/joint/negative-A/prefetch/reuse/
zero-decay/early-preparation. GDN_STAGED_PIPELINE selects the older staged path;
GDN_LEGACY_STAGES retains the earlier legacy comparison path. For individual
fusion ablations, GDN_STAGED_PIPELINE disables automatic fused definitions so
specific GDN_MEGA_DEFINITIONS can be selected explicitly.

Structural pass baseline f71a883 (full-head): initial-state36.354/44.813/
92.734/284.608us. Simulator T256 trace shows14% cube utilization and vector
producer waits. Profile:profile_structural/OPPROF_20261022152508_YOOUMPNXUMCJBEFC;
plot:structural_pipeline.png. Flattening matrix TCVT views passes25tests but
has no meaningful speedup (284.885us at4096), discarded. Three-half-buffer
streaming preparation also passes25tests, but only startup improves slightly
(35.000/42.714/91.256/283.335us). It does not reduce the long steady-state cost.
Need distinguish simulator serialization from hardware critical path using
phase counters before further scheduling changes.

Hardware counter correction: streaming producer cube [wait221,inverse3353,
rest8403] ticks; vector prepare8017/8190ticks, waits3374/3200. At32chunks,
cube wait is only~2%; simulator's startup producer stall is not the hardware
steady-state bottleneck. Dropped streaming preparation (snapshot ignored).
Six-product cube schedule now261.891us versus284.608us, with unchanged math.
Next work must remove matrix operations or state dependencies, not just flags.

Real-device Roofline profile of recurrence schedule1290ec8 atT4096:
device_roofline/OPPROF_20261022154557_CGPVAFKBAWGHKVLN. Active-core medians:
cube busy32.47%, MTE1 busy28.64%, MTE2 busy18.45%, FIX busy20.99%; cube wait
58.34%, MTE1 wait56.43%. Profiler diagnoses pipeline latency, not bandwidth.
Command: sourceCANN; timeout180 msprof op --kernel-name='*gdn_mega_specialized*'
--aic-metrics=Roofline --output=benchmark_data/gdn_mega/device_roofline
python benchmarking/gdn_mega.py --initial-state --profile-once --lengths4096.
Matmul probe confirms high movement/feedback penalty. Interleaved inverses
help weak gates~9.6%, slightly regress ordinary gates; optional, arithmetic
unchanged. User explicitly requests general pipeline/throughput optimization,
not additional decay-conditioned specializations. Next compare explicit
scheduling with compiler auto-sync disabled, then broaden recurrence scheduling
to finite decay and profile independent matrix-product transfer/compute overlap.

General recurrence schedule1e9fee7: finite decay included, no new decay shortcut.
25strict tests pass, initial ordinaryT4096=260.210us, weakT4096=286.148us.
Moving the state refill earlier gives only noise-scale differences; reverted.
Matmul store probe (256 stores) median ticks:contiguous3831,strided5506,L12048.
Only~1.44x output-store penalty; a new vector scatter stage would add substantial
traffic for a small possible saving, so not pursued. Probe raw:store_probe.jsonl.
RHS-first reassociation [betaV,-beta expK]*[I;S], then inverse*RHS passes25strict
tests but is slower263.750/288.095us atT4096 ordinary/weak and increases weak
outputRMSE0.000537→0.000575. Reverted; snapshot:rhs_recurrence.cpp. The original
residual arithmetic remains. Full-head acceptance of general schedules running.

Cross-chunk inverse lookahead (upper L0 banks, alternate state L1, next operands
prefetched during inverse) passes25 selected tests but is slower:265.828us
ordinary/286.730us weak at4096. Rejected; ignored snapshots preserve the experiment.
Reordering W before U to overlap W export also produces no robust improvement,
especially for weak decay; rejected. Shared Gram pipelining does help: full32
strict tests pass, fresh same-device ordinary261.734->254.176us and weak
283.597->276.511us. Single-wave Gram work uses original setup to avoid startup
regression. Next structural experiment: use otherwise idle cube cores for
output products alongside the state recurrence, with full per-chunk buffers.

2026-10-22 structural experiments after the Gram pipeline (all rejected):
- Offload output to the eight idle cube cores for a single H16 sequence.
  State producers publish rounded residual/state buffers; output workers run
  Q*S + attention*R. Selected25 tests pass with unchanged errors, but ordinary
  T4096 is265.334us and weak286.147us versus production~254/277us.
  Counters show producer waits~1861ticks, inverse~1946, recurrence~7456;
  vector preparation~10846/11011; output workers~7900work+3700wait ticks.
  This moved the bottleneck instead of removing it.
- Restoring contiguous two-slot attention plus explicit consumer acknowledgments
  does not recover the vector slowdown (preparation~10800ticks), disproving a
  store-layout-only explanation. Earlier residual publication still gives
  265.170/290.575us ordinary/weak. Ring snapshots:output_workers_ring/.
- Decouple preparation across all48 vector cores, with16 recurrence and8 output
  cube workers. Reuse each saved-state slot for its decay matrix before state
  publication. First attempt bypassed vector mask initialization and faulted;
  fixing initialization makes25selected tests pass onNPU1. NPU1 T4096 timings
  253.782us ordinary/305.935us weak are not a useful general improvement.
  No cross-device speedup claim. Snapshots:global_prepare/. Device0 subsequently
  reported healthy and passed25 tests against the frozen known-good library.
- Direct aligned load2d operand transfers instead of general img2colv2 loaders
  pass25 selected tests onNPU1;257.588/277.452us ordinary/weak. No demonstrated
  gain; rejected. Snapshots:load2d/.
- Split value columns into64-wide tiles, retaining four FP32 accumulators and
  interleaving residual/state exports with independent products. All25 selected
  tests pass with unchanged errors, but255.942/276.574us ordinary/weak are not
  better than production. Snapshot:recurrence_tiled.cpp.
- Let an output FIX store overlap the next inverse. Missing M->MTE1 ordering
  first caused an L0A conflict; adding the dependency and retaining full tail
  barriers passed25selected tests, but the4096 benchmark then failed with
  output relativeRMSE0.065378. This is NOT accepted. Full chunk-boundary barriers
  remain in production. Snapshots:deferred_output/.

All experimental files are ignored snapshots, not production source. Final
production rebuild uses all head specializations, empty experiment definitions,
and compiler automatic synchronization. No new decay-dependent shortcuts were
introduced in this structural pass; only the tested Gram pipeline is retained.

Final production recovery and acceptance: all32 tests passed onNPU0 after the
experimental faults; all-head build, no extra definitions, automatic syncON.
Initial32.912/40.444/83.016/254.862us; weak90.317/276.833us at1024/4096.
No-initial33.323/40.772/83.348/253.609us. All actual errors unchanged.
Device profile:device_gram_pipeline/OPPROF_20261022172022_WQDJDGCBKTHCAPFG,
1800MHz; active cube0..15 median busy31.06%,MTE126.67%,FIX21.26%; waits
cube54.57%,MTE151.50%. Summary:gram_profile_summary.json. The further2x target
is not achieved; source and build are restored to the tested Gram improvement.
