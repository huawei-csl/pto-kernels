# Validated GDN improvements

## 905ef0d — parallel gate scan and exact inverse doubling

Replaced 128 serial prefix-sum steps with seven parallel vector passes and
fused layout conversion. Replaced recursive dense inverse bookkeeping with
an exact nilpotent polynomial doubling algorithm that retains the FP32
accumulator, avoiding repeated multiplication by identity matrices.

All 21 tests pass on Ascend910B2, covering every supported value-head count,
initial states, ragged packed sequences, weak decay, and repeated execution.
Production build supports all heads; H=16, Hg=4 benchmark, device graph median:

| Tokens | Baseline us | New us | Speedup |
|---|---:|---:|---:|
| 128 | 104.892 | 93.739 | 1.119x |
| 256 | 141.561 | 125.622 | 1.127x |
| 1024 | 306.957 | 258.306 | 1.188x |
| 4096 | 863.501 | 779.279 | 1.108x |

Output relative RMSE remains ~0.00038; final-state relative RMSE ~0.00032.
Commands and setup are in memory.md. Raw results: baseline.json and
first_acceptance.json under benchmark_data/gdn_mega/ (local ignored artifacts).
The 3x target has not yet been reached.

## 6453a98 — host dispatch

Choose one templated kernel entrypoint per head count on the host. All seven
specializations remain available, but unused specializations no longer enlarge
the launched kernel. All 21 tests pass. Latencies: 76.919 / 98.894 / 228.839 /
750.282 us at T=128/256/1024/4096, respectively (1.36x / 1.43x / 1.34x / 1.15x
against the original baseline). Numerical errors unchanged.

## 4a68a0b — shared products, cube recurrence, and optional fused pipeline

Share KK/QK Gram products across grouped value heads; invert two 64x64
diagonal blocks and join them exactly; overlap WY and state-input preparation
with inversion; retain recurrent state on the cube. The default staged path
passed all 23 tests across all supported head counts on Ascend910B2.

The optional `GDN_FUSED_PIPELINE` path uses persistent head workers, two
producer/consumer slots, contiguous intermediate layouts, and L1/L0-resident
inverse/WY/state/output intermediates. It passed 16 selected H=16 tests,
including packed tails, initial states, weak gates, and repeated execution.
Other head counts have not yet been validated for this optional path.

H=16, Hg=4 device-graph medians (microseconds):

| Tokens | Original baseline | Default staged | Optional fused | Fused speedup |
|---|---:|---:|---:|---:|
| 128 | 104.892 | 52.708 | 39.185 | 2.68x |
| 256 | 141.561 | 75.084 | 51.217 | 2.76x |
| 1024 | 306.957 | 169.278 | 124.193 | 2.47x |
| 4096 | 863.501 | 553.351 | 411.824 | 2.10x |

Fused output relative RMSE is at most 0.000432 and final-state relative RMSE
at most 0.000389 on these benchmarks. Existing acceptance thresholds were
unchanged. Raw records: `benchmark_data/gdn_mega/output_acceptance.json` and
`benchmark_data/gdn_mega/fused_contiguous.json` (local ignored artifacts).
These measurements and device tests completed before the user paused work;
this commit records that validated checkpoint. The 3x target remains unmet.
The unfinished adaptive-inverse experiment and its unrun tests are excluded.

## e900fcd — bounded adaptive inverse

`GDN_ADAPTIVE_INVERSE` measures the infinity norm of the actual FP16 triangular
matrix and selects 2/4/8/16/32 polynomial terms only when a geometric tail
bound is at most 3.2e-6 per diagonal block. Otherwise the exact 64-term block
inverse is retained. This is a truncation bound; floating-point rounding is
validated separately by reference comparisons.

Full-head build with `GDN_FUSED_PIPELINE;GDN_ADAPTIVE_INVERSE` passed all 25
tests on Ascend910B2, including new zero-beta and correlated-key, zero-decay
cases, with five repeated executions. Benchmark medians at T=128/256/1024/4096:
39.143 / 49.481 / 112.795 / 372.280 us. The long case improves 9.6% over the
previous fused checkpoint; actual output/state error metrics are unchanged.
Raw results: `benchmark_data/gdn_mega/adaptive_acceptance.json`. No tolerance
relaxation; 3x target remains unmet. This path remains optional at this commit.

## 3a19c00 — joint output/state products

Stack output and next-state equations into two shared-RHS matrix products:
`[O; Snext] = [Qscaled; decayI] S + [attention; Ktilde^T] R`. This replaces four
separate GEMMs with two taller GEMMs, sharing RHS transfers and synchronization.
The algebra is unchanged and works with nonzero initial state. Optional flag:
`GDN_JOINT_PRODUCTS`, together with fused/adaptive flags.

18 selected H16 tests pass, including initial state, weak gates, packed tails,
and correlated keys. Initial-state benchmark: 39.382 / 50.061 / 111.211 /
359.178 us at T=128/256/1024/4096. Original initial-state baseline: 107.532 /
145.567 / 309.691 / 863.765 us (2.73x / 2.91x / 2.78x / 2.40x). Actual
output/state error remains approximately .00043/.00039. No tolerances changed.
Raw records: joint_output_initial.json and baseline_initial.json under
benchmark_data/gdn_mega/. All-head acceptance remains pending for this flag.

## d41e12a — in-place signed inverse

The vector preparation now emits -A directly. The cube inverts the two
diagonal blocks in place with +I and then overwrites the untouched lower
offdiagonal block with its inverse join. This removes a full128x128 GEMM
against -I and two intermediate copies per chunk. The inverse algebra and
measured numerical errors are unchanged. Enable with `GDN_NEGATIVE_A` in
addition to fused/adaptive/joint-product flags.

All 18 selected H16 tests pass. Initial-state medians: 38.915 / 48.909 /
104.472 / 331.824 us; T4096 improves 7.6% over the preceding joint-product
checkpoint and is 2.60x faster than the original initial-state baseline.
Raw results: benchmark_data/gdn_mega/negative_a_initial.json. Full-head
acceptance is still pending for these optional fused changes.

## f666654 — overlap operand loads with inversion

The in-place inverse frees L1 operand buffers early. Load W/U, scaled keys,
queries, attention, and decay during inversion; narrow inverse synchronization
to MTE1/cube/FIX instead of blocking independent MTE2 transfers.
`GDN_PREFETCH` requires `GDN_NEGATIVE_A`.

18 selected H16 tests pass. Initial-state medians: 37.458 / 45.992 / 96.432 /
299.105 us, with unchanged numerical errors. T4096 improves 9.9% over the
preceding checkpoint. Speedups over original initial-state baseline:
2.87x / 3.17x / 3.21x / 2.89x. Full-head acceptance is pending. Raw results:
benchmark_data/gdn_mega/prefetch_initial.json.

## 3eae108 — adaptive-depth publication fix and strict full acceptance

A stricter reference check exposed a race in scalar depth metadata publication.
The vector's dcci cache flush lacked dsb(DSB_DDR) before the ready notification;
the cube could consequently use an insufficient depth for a short packed tail.
Forcing64 terms removed the failure. The affected matrices have row norms
2.7–3.6 and should always select64, so this was an implementation bug rather
than a legitimate bounded truncation. Added volatile metadata accesses and
DDR fences. Removed the dead initial-state identity product for nonempty
sequences.

All31 tests pass in the full-head build, including50 repeated executions of
the weak-decay packed regression. Every reference comparison now requires
finite results and relativeRMSE<0.005; the previous0.01 absolute-error early
exit and5% relative threshold are removed. Earlier entries reported passes
under those older checks; this acceptance supersedes them.

With initial state:39.651/48.429/98.719/301.066us atT128/256/1024/4096.
Without initial state:38.417/47.582/97.472/299.847us. All optional fused flags
(fused/adaptive/joint-products/negative-A/prefetch) are included. Output RMSE
~0.00043, state~0.00039 on benchmark inputs. Raw records:fenced_acceptance*.json
under benchmark_data/gdn_mega/. 3x is not yet achieved across all lengths.

## 9e91150 — reuse resident operands

Reuse S already resident in L0B after W*S. When a chunk's stored half decay
matrix is exactly zero and the preceding L1 copy is also zero, retain that
L1 copy instead of reloading it. This does not assume S is zero and introduces
no additional approximation. L0A must still reload decay because the second
joint product overwrites that region with Ktilde^T.

25 selected strict H16 tests pass, including a new test alternating finite
and zero half-decay matrices. Initial-state medians:37.468/46.568/95.733/
292.657us; actual errors unchanged. T4096 is2.95x faster than the original
initial-state baseline. Flag:GDN_REUSE_OPERANDS (requires adaptive metadata
and joint products). Raw results:benchmark_data/gdn_mega/reuse_initial.json.

## 073ae91 — share prefix exponentials and simplify decay construction

W, Q, and final decay reuse one exp(g) vector. Construct decayI directly in
half precision instead of converting the entire identity to float and back;
fill zero directly when its half value is zero. A vector-to-scalar event
orders exp(g) before short-tail scalar reads. exp(g_i-g_j) remains unchanged.
25 selected strict tests pass. Initial-state medians:36.726/45.199/94.346/
292.951us; shorter cases improve, long case is unchanged. Actual errors are
unchanged. Raw results:benchmark_data/gdn_mega/vector_trim_initial.json.

## 28929db — omit products with exactly zero half decay

When exp(g_last) rounds to FP16 zero, omit decay-matrix construction/load
and the full S*0 cube product. Use three ordinary products for output/state;
retain joint products when decay is nonzero. This supports nonzero incoming
state and adds no approximation to the existing half-decay calculation.
25 selected strict H16 tests pass, including changing decay and the repeated
packed regression. Initial-state medians:35.877/44.158/91.991/285.193us.
Speedups:2.997x/3.296x/3.367x/3.029x. T128 remains just below target.
Raw results:benchmark_data/gdn_mega/skip_decay_initial.json.
Validation: bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh;
then timeout 120 python -m pytest tests/test_gdn_mega.py -q -k
'not head_configs or H16'; then timeout 120 python benchmarking/gdn_mega.py
--initial-state --output benchmark_data/gdn_mega/skip_decay_initial.json.

## b315f03 — overlap first chunk preparation with cube setup

Seed the first producer credits before loading identity/state, since vector
preparation has no dependency on those cube tiles. Subsequent head credits
retain their existing ordering. 25 selected strict tests pass.
Initial-state medians:35.337/42.725/90.698/283.634us, exceeding3x against the
original initial-state baseline at all four lengths. Full-head default-build
acceptance remains pending. Raw:benchmark_data/gdn_mega/early_initial.json.
Commands match preceding entry with GDN_EARLY_PREPARE added to definitions.

## f71a883 — enable validated fused pipeline by default

Full production build (all head specializations, empty experiment definitions)
passes all32 strict tests on Ascend910B2. Initial-state medians:
36.354/44.813/92.734/284.608us; no-state:35.750/43.582/91.639/285.068us.
Unlike the earlier single-head experimental build, this production measurement
falls short of3x atT128; long-sequence speedup remains above3x. No claim of
universal3x is made. Raw results:final_initial.json,final_no_initial.json.
Commands: cmake -S . -B build -DGDN_MEGA_HEAD= -DGDN_MEGA_DEFINITIONS=
-DGDN_MEGA_MANUAL_SYNC=OFF; bash scripts/build_gdn_mega.sh; timeout120
python -m pytest tests/test_gdn_mega.py -q; benchmark with/without --initial-state.
User requests the next structural optimization target: another2x from this
checkpoint, emphasizing cube utilization, global pipeline, and movement reuse.
Frozen library:benchmark_data/gdn_mega/structural_baseline_lib/.

## 1290ec8 — pipeline the six recurrence products

Use two L0A/L0B tiles and explicit producer/consumer dependencies to overlap
Q*S with residual export, load subsequent operands during accumulator transfers,
and overlap state export with the final attention product. Arithmetic is
unchanged. GDN_PIPELINED_RECURRENCE selects the new zero-half-decay branch;
finite-decay and first zero-state chunks keep the existing path.
25 selected strict H16 tests pass. Initial-state medians:
33.705/41.230/85.323/261.891us. T4096 improves8.0% versus the production
structural baseline284.608us; this is1.087x, not the requested additional2x.
Numerical errors are unchanged. Raw:benchmark_data/gdn_mega/recurrence_initial.json.
Commands: configure HEAD=16, DEFINITIONS=GDN_PIPELINED_RECURRENCE; build script;
timeout120 pytest tests/test_gdn_mega.py -q -k 'not head_configs or H16';
timeout120 python benchmarking/gdn_mega.py --initial-state --output above.

## a8fe7ff — interleave independent inverse solves

GDN_INTERLEAVED_INVERSE queues the two diagonal solves together, overlapping
cube products with FIX exports using separate operand/accumulator tiles. It
uses exactly the same polynomial terms and rounding; no decay threshold is
added. All25 selected strict tests pass. With gate_scale0.01 and nonzero initial
state, T1024/4096 improve105.278/337.722us to99.331/308.240us on the same npu:0
(1.060x/1.096x). Ordinary-gate timings36.052/42.528/86.675/264.372us are slightly
slower than the preceding recurrence-only build, so this remains optional.
Actual output and state errors are unchanged. Raw files:interleaved_initial,
interleaved_weak,structural_baseline_weak_npu0.json under benchmark_data/gdn_mega.
Build:HEAD=16,DEFINITIONS=GDN_PIPELINED_RECURRENCE;GDN_INTERLEAVED_INVERSE,
MANUAL_SYNC=OFF. Same selected pytest and benchmark commands; benchmark now
accepts --gate-scale0.01 (with a space before the numeric argument).

Added a diagnostic GDN_MATMUL_PROBE build and benchmarking/gdn_matmul_probe.py.
All24 cores return exact identity products across five runs. For512 products,
median resident/reload/feedback ticks:64x64=913/4310/5748;
128x128=7284/15800/18676. Operand reload plus feedback costs6.30x and2.56x
resident arithmetic respectively. Probe command: configure DEFINITIONS=
GDN_MATMUL_PROBE, build, timeout120 python benchmarking/gdn_matmul_probe.py.
This diagnostic replaces GDN execution and must not be used for GDN results.

## 1e9fee7 — extend recurrence scheduling to finite decay

The pipelined recurrence now includes the incoming-state decay product when
its coefficient is nonzero. It no longer selects the optimized schedule based
on decay. Track the final accumulator layout explicitly instead of inferring
it from the coefficient. No additional approximation or term truncation.
25 selected strict tests pass; numerical errors unchanged. Initial-state
T128/4096:32.934/260.210us. Weak-decay T1024/4096:93.816/286.148us, improving
7.2% at4096 over308.240us before generalizing the schedule, and18.0% speedup
(1.180x) versus337.722us structural baseline. Ordinary4096 is1.094x versus
284.608us. The additional2x target remains unmet.
Raw:general_initial.json,general_weak.json. Build and validation commands match
preceding entry. Disabling compiler auto-sync did not materially help; restored
MANUAL_SYNC=OFF. Full-head acceptance for these optional schedules is pending.

## 3e4c336 — production acceptance of general matrix schedules

Enable pipelined recurrence and interleaved inverse by default. Full production
build (all head specializations, no experiment definitions) passes all32 strict
tests. Initial-state T128/256/1024/4096:33.582/40.720/85.264/263.232us.
Weak-decay T1024/4096:91.686/282.969us. Errors unchanged; ordinary output RMSE
~0.00043, weak~0.00054, both below0.005. Ordinary4096 is1.081x and weak4096
1.193x versus the new structural checkpoint; another2x is still the goal.
Raw:general_full_initial.json,general_full_weak.json. Configure HEAD empty,
DEFINITIONS empty,MANUAL_SYNC=OFF; build; timeout120 pytest -q
 tests/test_gdn_mega.py; benchmark with --initial-state and optionally
--gate-scale 0.01. Added validated store probe:contiguous/strided/L1 ticks
3831/5506/2048 for256 stores (24-core diagnostic). Changing output layout is
not justified by this modest penalty. Rejected RHS-first reassociation is
not included in production.

## 8cfbfa4 — overlap shared Gram transfers and products

Pipeline shared K*K^T and Q*K^T across tiles: prefetch the next K/Q pair while
current products run, retain K^T, and export the first result during the second
product. The single-wave path keeps its shorter setup. Arithmetic, grouped-head
reuse, decay handling, and error thresholds are unchanged.

All 32 strict tests pass in the full-head production build. Fresh same-device
before/after T4096 initial-state medians: ordinary 261.734 -> 254.176 us (1.030x),
weak decay 283.597 -> 276.511 us (1.026x). T128 is 32.402 us versus 32.896 us.
An earlier A/B run without single-wave dispatch confirmed long-sequence gains
but regressed startup; that regression is avoided by the final dispatch.
Raw results: gram_before_initial.json, gram_before_weak.json,
gram_dispatch_initial.json, gram_dispatch_weak.json under benchmark_data/gdn_mega.

Validation: configure GDN_MEGA_HEAD and GDN_MEGA_DEFINITIONS empty,
GDN_MEGA_MANUAL_SYNC=OFF; `bash scripts/build_gdn_mega.sh`;
`timeout 120 python -m pytest tests/test_gdn_mega.py -q`;
`timeout 120 python benchmarking/gdn_mega.py --initial-state --lengths 128 4096`
and `--initial-state --gate-scale 0.01 --lengths 4096`, all through
`scripts/gdn_mega_docker.sh`. The additional 2x target remains unmet.

Final revalidation of 8cfbfa4 after rejecting all further experiments: full-head
production build passes all32 tests. Initial-state T128/256/1024/4096 medians
32.912/40.444/83.016/254.862us; no-state33.323/40.772/83.348/253.609us.
Weak-decay initial-state T1024/4096:90.317/276.833us. Initial-state gains versus
original10748eb are3.267x/3.599x/3.730x/3.389x. Gains versus the newer f71a883
checkpoint at4096 are1.117x ordinary and1.220x weak; the further2x is not achieved.
Raw files:production_final_initial.json,production_final_no_initial.json,
production_final_weak.json. No relaxed checks or experimental compile flags.

Real-device profile of this exact production build:
`msprof op --kernel-name="*gdn_mega_specialized*" --aic-metrics=Roofline
--output=benchmark_data/gdn_mega/device_gram_pipeline python benchmarking/gdn_mega.py
--initial-state --profile-once --lengths 4096` (source CANN first, timeout180).
Profile OPPROF_20261022172022_WQDJDGCBKTHCAPFG; active-core medians: cube busy31.06%,
MTE1 busy26.67%, FIX busy21.26%; cube wait54.57%, MTE1 wait51.50%. The recurrence
still has substantial transfer/dependency stalls. Larger core-splitting,
column-tiling, direct-load, and deferred-store experiments are recorded with
rejection evidence in memory.md; none are in production.

## 5141f56 — production cleanup and saved 8K/16K/32K comparison

Removed private legacy mega stages, unused inverse variants, matmul_probe and
its runner, timing/early-exit hooks, and algorithm tuning preprocessor switches.
The validated production schedule is now unconditional; architecture guards and
runtime shape/numerical decisions remain. Removed the experiment-only CMake
options; retained GDN_MEGA_ONLY for fast builds. Mega source drops from 19 files /
7,126 lines to seven used files / 1,104 lines (84.5% fewer). One launch contains
two global phases: concurrent gate/layout + shared KK/QK, then persistent
preparation/inverse/WY/state/output. Source map and buffer roles are documented.

Public signature, shapes, dtypes, and allocation sizes are unchanged. Corrected
host comments: named intermediate buffers are opaque scratch; reserved arguments
remain for compatibility. This is a maintainability improvement, not a claimed
new algorithmic speedup. Ordinary latency varies -2.75% to +1.16% against 9ea497c,
and its reported numerical errors match the pre-cleanup checkpoint.

Final full-head production build: 32 tests passed in 5.85 s on Ascend910B2,
CANN9.1.0 / torch and torch-npu2.10.0, NPU0. Benchmarking checks output and final
state against the CPU FP32 reference for five launches per case, with unchanged
finite-value and relative-RMSE <0.005 requirements. Ordinary lengths
128/256/1024/4096/8192/16384/32768 and weak-decay lengths4096/8192/16384/32768 all
pass with a nonzero initial state. No relaxed checks. Source include/guard audit,
Python syntax, report links, and git diff --check pass.

Original10748eb -> cleaned5141f56 medians (ms), H16/Hg4,D=C128:

| Tokens | Ordinary original | Ordinary cleaned | Speedup | Weak original | Weak cleaned | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0.8688 | 0.2576 | 3.37x | 0.8663 | 0.2780 | 3.12x |
| 8192 | 1.7197 | 0.5346 | 3.22x | 1.6848 | 0.5826 | 2.89x |
| 16384 | 3.5022 | 1.0938 | 3.20x | 3.3755 | 1.1761 | 2.87x |
| 32768 | 6.9650 | 2.2583 | 3.08x | 6.8061 | 2.4120 | 2.82x |

The weak-decay long-sequence results do not meet a universal3x target. The 32K
ordinary output/state relative RMSE is0.000430/0.000381; weak is0.000538/0.000487.

Validation commands, through scripts/gdn_mega_docker.sh:

```bash
bash scripts/build_gdn_mega.sh
timeout 120 python -m pytest tests/test_gdn_mega.py -q
timeout 300 python benchmarking/gdn_mega.py --initial-state \
  --lengths 128 256 1024 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/branch_report_20261023/cleanup_initial.json
timeout 300 python benchmarking/gdn_mega.py --initial-state --gate-scale 0.01 \
  --lengths 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/branch_report_20261023/cleanup_weak.json
```

Tracked report and all tables: docs/gdn_mega/optimization_report.md,
checkpoint_latency.csv, checkpoint_stages.csv, production_latency.csv, and
measurements.json. JSON preserves raw timing/error metrics, environment, dense
matrix FLOP accounting, and measured kernel-library SHA-256 fingerprints.

## 2a5cce3 — checked shape sweep, heatmap, and execution explanation

Added a reusable benchmark sweep and a separate CPU plotting script. The sweep
varies H=8/12/16/24/32/48/64 and T=128/256/1024/4096/8192/16384/32768 with fixed
GQA group4, D=C128, ordinary logsigmoid gates, seed0, and nonzero initial states.
Original10748eb and cleaned5141f56 libraries run in separate processes. Their
actually loaded SHA-256 fingerprints match the saved branch-report artifacts.
Each passing implementation/shape checks output and final state over five
warmups and after timed graph replay; medians use15trials of20calls. Allocation
and reference checks are outside timing. The runner releases per-shape graphs
and tensors, checkpoints results, and supports validated resume.

All49 optimized shapes pass (largest output/state relative RMSE0.000434155,
threshold0.005 unchanged). There are48 valid speedups ranging2.90654x–7.52798x.
The original baseline fails output correctness atH64/Hg16/T32768 in two processes:
relative RMSE0.087156 then0.087424. Its cell is explicitly gray/failed, without
baseline latency or a speedup. The optimized kernel passes that shape at7.51260ms,
with output/state relative RMSE0.000430010/0.000377453. The baseline failure's
cause has not been diagnosed; no tolerance relaxation or baseline modification.

Validated real-device normal/failure paths, matching CSV/JSON shape grids and
ratios, resume without retiming completed rows, rejection of incomplete plotting
grids, PNG visual inspection, SVG XML parsing, Python syntax, and git diff
--check. No production kernel or interface changes. Existing32-test acceptance
remains in the previous entry; this change adds the larger benchmark coverage.

Commands (repository root):

```bash
bash scripts/gdn_mega_docker.sh timeout 1800 python benchmarking/gdn_mega_sweep.py \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib \
  --output docs/gdn_mega/heatmap.json
bash scripts/gdn_mega_docker.sh timeout 900 python benchmarking/gdn_mega_sweep.py \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib \
  --output docs/gdn_mega/heatmap.json --resume
bash scripts/gdn_mega_docker.sh python benchmarking/plot_gdn_mega.py \
  docs/gdn_mega/heatmap.json
```

Saved docs/gdn_mega/heatmap.{png,svg,csv,json,md}. The Markdown explains executed
GFLOPs versus TFLOP/s, the two global phases inside one launch, head-persistent
chunk processing, and its head-count scheduling implications. Initial sweep
stopped on the baseline numerical failure; the runner now records numerical
failures explicitly and the validated resume completed the remaining row.
