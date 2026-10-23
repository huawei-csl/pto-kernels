# GDN speedup heatmap and execution model

The heatmap compares original checkpoint **10748eb** with the cleaned production
kernel **5141f56**. The later acceptance commit 1e92737 changes documentation only.
Measurements use Ascend 910B2 / CANN 9.1.0 / torch and torch-npu 2.10.0, NPU 0.

![GDN speedup by sequence length and value heads](heatmap.png)

Each passing cell is `original median latency / current median latency`. Gray
cells indicate a numerical correctness failure and have no speedup. Rows vary all
supported value-head counts H; columns vary sequence length. The grouping ratio
is fixed at four value heads per key head: Hg=H/4. D=chunk size=128. Inputs use
seed 0, normalized Q/K, ordinary logsigmoid gates (scale 1), and a nonzero initial
state. This is a controlled synthetic workload, not a guarantee for every gate
or key distribution. Other gate scales and grouping ratios can be swept with
the same script.

Both output and final state must pass against the CPU FP32 reference before a
shape receives a speedup. Passing shapes use five checked warmups and another
check after graph replay. Failed shapes record the assertion and skip timing. Timing uses the median
of 15 trials of 20 graph-replayed calls, warm cache, with allocations and checks
outside the measured interval. Baseline/current run in separate processes;
measurement order alternates across head counts. The worker reports the SHA-256
of the actually loaded kernel library, which the sweep verifies against each
requested library. Raw numerical errors and latency ranges accompany the CSV.

- [Annotated PNG](heatmap.png) and [scalable SVG](heatmap.svg)
- [Latency and speedup CSV](heatmap.csv)
- [Raw results and measurement settings](heatmap.json)

## Results and validation

The optimized kernel passes **all 49 shapes**. The **48 valid comparisons** range
from **2.91x to 7.53x**. Its largest output/state relative RMSE is 0.0004342,
well below the unchanged 0.005 threshold.

At **H=64, Hg=16, T=32768**, the original baseline fails output correctness in
two separate processes: relative RMSE 0.087156 and 0.087424. The optimized
kernel passes that shape (output/state relative RMSE 0.0004300/0.0003775,
latency 7.5126 ms). The gray cell therefore has **no speedup**, and the CSV leaves
baseline latency/speedup empty. This failure was discovered by the extended
sweep; its cause has not been diagnosed. The comparison preserves the original
baseline rather than modifying its algorithm or relaxing tolerances.

Validation also covers plotting the complete grid, rejecting incomplete grids,
resuming without rerunning completed rows, matching loaded library hashes to
the previously recorded baseline/production artifacts, CSV ratio consistency,
and Python syntax. The measurement runner releases each shape's graph and
tensors before allocating the next shape. No production kernel changes were
made for this sweep.

## Reproduce

The two libraries must have the same public GDN ABI. The original library was
built and frozen from 10748eb during the branch baseline measurements; the current
library is built from the production tree. Library binaries are local artifacts,
not committed. To compare a future change, copy its predecessor's built library
to a separate directory before rebuilding the current version.

Run from the repository root:

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh timeout 1800 python benchmarking/gdn_mega_sweep.py \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib \
  --current-lib build/lib \
  --output docs/gdn_mega/heatmap.json
bash scripts/gdn_mega_docker.sh python benchmarking/plot_gdn_mega.py \
  docs/gdn_mega/heatmap.json
```

The sweep defaults to H=8/12/16/24/32/48/64 and T=128/256/1024/4096/8192/16384/32768.
For a small comparison:

```bash
bash scripts/gdn_mega_docker.sh timeout 300 python benchmarking/gdn_mega_sweep.py \
  --baseline-lib benchmark_data/gdn_mega/baseline_lib \
  --heads 8 16 32 --lengths 128 4096 --output benchmark_data/gdn_mega/small.json
```

Use `--gate-scale 0.01` for weak decay, `--group-size 1` for equal query/value
heads, or `--no-initial-state` for zero incoming state. Head counts must divide
cleanly by the chosen grouping ratio. The benchmark always runs correctness
checks. It writes a combined JSON and CSV after each completed head-count pair;
the plot refuses an incomplete grid and marks explicit correctness failures gray.
Use `--resume` with the same command to keep completed head rows; it checks
measurement settings and library fingerprints before reusing them. Numerical
failures are saved as failed cells, while crashes, timeouts, and incomplete worker
results abort the sweep. Plotting needs only NumPy and Matplotlib,
not an NPU, torch, or the kernel library. `--output path/name` chooses a different
PNG/SVG filename stem.

## Why fewer TFLOP/s can still mean faster execution

**GFLOPs measure how much arithmetic is executed. TFLOP/s measures its rate.**
The earlier 4K comparison (H=16, Hg=4, ordinary decay, nonzero initial state) was:

| Version | Executed matrix GFLOPs | Full-kernel latency ms | Matrix GFLOPs / full-kernel ms = TFLOP/s |
|---|---:|---:|---:|
| Original 10748eb | 68.921 | 0.8688 | 79.3 |
| Optimized checkpoint 9ea497c | 18.128 | 0.2546 | 71.2 |

The optimized implementation executes **3.80x less matrix work** and finishes
**3.41x faster**, despite an approximately 10% lower whole-kernel arithmetic rate:

```text
speedup = (old work / new work) × (new rate / old rate)
        = 3.80 × (71.2 / 79.3)
        ≈ 3.41
```

These rates divide executed dense GEMM FLOPs by total elapsed kernel time,
including vector work, data movement, and waiting. They are not the hardware's
matrix throughput while its cube pipeline is busy. Old dense products on zero
or identity blocks inflate the arithmetic count while accomplishing little.
Removing them is beneficial even if the reported arithmetic rate decreases.
Sharing Gram products, using 64×64 inverse blocks, and bounded truncation reduce
work; fusion and overlap also reduce memory traffic and stalls. The speedup
cannot be attributed solely to launch overhead: both versions already used one
launch. Nor does this comparison show that every individual matrix operation
has lower throughput: fused/overlapping stages have different timing boundaries.

If both versions were assigned the same *useful problem FLOP count*, useful
TFLOP/s would increase exactly in proportion to the speedup. The reported table
instead counts each implementation's actual dense matrix instructions. These
are different metrics and should not be mixed.

## What two phases and persistent execution mean

There is **one device kernel launch** containing **two global execution phases**:

1. **Shared preparation.** Vector cores compute gate prefix sums and layouts
   while cube cores compute KKᵀ and QKᵀ once per key head. A global barrier makes
   both sets of results available to the next phase.
2. **Persistent fused computation.** A cube core owns a sequence/value-head work
   item and walks through its chunks in order. Two vector producer slots prepare
   chunk operands while the cube computes the inverse, WY products, recurrent
   state update, and output. State and cube intermediates remain in on-chip
   storage across chunks; prepared vector operands still use global scratch.

Conceptually, the cube part does this inside one launch:

```text
for each sequence/head assigned to this cube core:
    load initial state (or initialize zero state)
    for each chunk, in order:
        consume prepared operands
        compute inverse and WY
        compute output and update the resident state
    store final state
```

“Persistent” describes that long-lived worker loop and retained state. It does
not mean execution persists after the call, that state is retained across
separate launches, or that preallocation is new. Successive calls still use the
explicit initial/final-state interface.

A phase is not a file or a separate launch. Seven source files provide helpers
for these two phases. The old six conceptual stages were gate preparation, KK,
inverse, WY, state, and output (seven calls when transpose was counted
separately). Their mathematics still exists, but their scheduling boundaries
have changed.

Persistence also explains why speedup can depend on head count: recurrence
within a head is sequential across chunks. On this 24-cube-core device, a single
sequence with eight heads occupies only eight cube cores during the persistent
phase; 24 heads can occupy all 24; larger head counts require additional work
items on some cores. Shared preparation has a different distribution of work.
The heatmap measures the combined effect rather than assuming a uniform speedup.
