# GDN mega-kernel branch report

Measured on 2026-10-23. This report preserves the comparison of the starting
checkpoint **10748eb** with optimized checkpoint **9ea497c**, then records the
production cleanup and longer-sequence measurements below. The 4,096-token
kernel improved from **0.8688 ms to 0.2546 ms (3.41x)** with no additional
allocations required.

## What changed

1. **Fused inverse → WY → state update → output into a persistent pipeline.**
   State and intermediate operands stay on-chip longer, reducing global-memory
   round trips. Both versions already used one kernel launch; this improves
   execution inside that launch.
2. **Shared KKᵀ and QKᵀ across grouped-query heads.** These products now run once
   per key head. With 16 value heads and 4 key heads, this removes 75% of their
   matrix work.
3. **Reworked the inverse algorithm.** Two 64×64 diagonal inverses plus an
   off-diagonal combination replace wasteful full 128×128 operations. Several
   products used for identity copies or negation were eliminated.
4. **Skipped negligible high powers using a numerical bound.** The measured
   matrix norm determines how many inverse-series terms are needed. The ordinary
   4K benchmark uses 8 terms in 364 head-chunks and 16 terms in 148; weak decay
   uses all 64 terms in all 512. The numerically stable `g_i − g_j` calculation
   remains. This is bounded truncation, not a blind assumption about decay.
5. **Improved overlap throughout the pipeline.** Parallel gate scans, fused
   transposes, double buffering, operand prefetching, and interleaved matrix
   operations reduce idle time. A state product is also skipped when its decay
   coefficient rounds to exactly zero in FP16. This works with nonzero initial
   states.

## Full-kernel benchmark at the optimized checkpoint

Ascend 910B2, CANN 9.1.0, PyTorch/torch-npu 2.10.0, NPU 0, 24 cube/48 vector
cores. One packed sequence, 16 value heads, 4 key heads, head/chunk size 128.
Every row uses a **nonzero initial state**. Inputs use deterministic normalized
Q/K and logsigmoid gates; weak decay multiplies log gates by 0.01. FP16
Q/K/V/beta/state, FP32 gates. All supported head specializations are built.

Five checked warmups, then 15 trials of 20 graph-replayed calls; median device
latency, warm cache. Allocation and CPU FP32 reference checks are outside timing.
The output's existing `128**-0.5` reference scaling is also outside kernel timing.
These synthetic inputs do not represent every model's gate distribution.

| Tokens | Decay workload | Original ms | Optimized ms | Speedup |
|---:|---|---:|---:|---:|
| 128 | Ordinary | 0.1112 | 0.0353 | 3.15x |
| 256 | Ordinary | 0.1474 | 0.0421 | 3.50x |
| 1,024 | Ordinary | 0.3123 | 0.0851 | 3.67x |
| 4,096 | Ordinary | 0.8688 | 0.2546 | 3.41x |
| 4,096 | Weak decay | 0.8663 | 0.2776 | 3.12x |

Weak decay retains all inverse terms and the decay-state product, showing that
substantial gains survive without those shortcuts.

## Per-stage breakdown: 4,096 tokens, ordinary decay

GFLOPs count **executed dense matrix arithmetic**, using `2MNK` per
multiplication, including identity/zero products. Vector arithmetic is excluded;
“—” denotes a phase without matrix products, not zero total arithmetic.

| Version | Stage | Matrix GFLOPs | Device phase ms |
|---|---|---:|---:|
| Original | Gate prefix scan | — | 0.0155 |
| Original | Transposes | — | 0.0073 |
| Original | KKᵀ | 2.147 | 0.1144 |
| Original | Inverse | 51.741 | 0.2838 |
| Original | Additional barrier | — | 0.0024 |
| Original | WY | 4.295 | 0.0706 |
| Original | State update | 4.295 | 0.2066 |
| Original | Output, including QKᵀ | 6.442 | 0.1566 |
| Optimized | Scan/transposes overlapping shared KKᵀ + QKᵀ | 1.074 | 0.0206 |
| Optimized | Fused preparation/inverse/WY/state/output | 17.054 | 0.2265 |

WY, state, and output overlap inside the fused pipeline, so they no longer have
independent stage latencies. Phase times are the maximum across cores of the
per-core median over 15 instrumented launches, including waits/barriers. The
counter frequency was queried from the driver: 50,003,000 Hz. These diagnostic
measurements do not sum exactly to uninstrumented full-kernel latency. Diagnostic
instrumentation is absent from production source.

| Full kernel | Matrix GFLOPs | Measured ms |
|---|---:|---:|
| Original | 68.921 | 0.8688 |
| Optimized | 18.128 | 0.2546 |
| Optimized, weak decay | 23.186 | 0.2776 |

Ordinary-decay matrix work is **73.7% lower**. Inverse arithmetic alone falls from
51.741 to 4.068 GFLOPs. At 4K there are 512 value-head chunks and 128 key-head
chunks. The original inverse executes 24 full 128³ products per chunk plus 48
setup products. The current diagonal solves/join execute 14 or 18 products of
size 64³ for 8 or 16 terms, respectively; the 64-term path uses 26. Recurrence
uses six 128³ products per chunk, or seven with finite FP16 decay, plus 24
identity setup products per launch. Each “product” counts `2*M*N*K` FLOPs.

## Interfaces and allocation

| Contract | Change |
|---|---|
| Public function signature and arguments | Unchanged |
| Input/output shapes and dtypes | Unchanged |
| Optional initial state and final-state output | Preserved |
| Required workspace sizes | Unchanged |
| Additional preallocation | None |
| Intermediate buffer contents after execution | Changed: opaque scratch |

**A, A_inv_f32, A_inv, w, u, s, and v_new must be treated as opaque scratch.**
Code inspecting their previous mathematical contents after execution is
incompatible. The cleanup corrects the host-side comments to reflect this.

At T=4,096/H=16/Hg=4, scratch/intermediate buffers remain **136.875 MiB**;
all unique supplied tensors total **178.406 MiB**, unchanged. Reduced traffic
has not yet translated into a smaller allocation interface. Reserved arguments
are retained for compatibility; callers can continue allocating exactly as
before. Outputs are written in place; the kernel performs no allocations.

Packed token count and sequence count are dynamic. Value heads are selected at
runtime from compiled specializations 8/12/16/24/32/48/64; key heads must divide
value heads. D=C=128 is fixed. Initial state remains `[N,H,128,128]` FP16 and
final state `[N*H,128,128]` FP16. See the
[source/interface map](../../csrc/kernel/gdn_mega/README.md) for buffer roles.

The optimized checkpoint passed all **32 tests**. Results are tolerance-checked,
not bitwise identical: finite values and relative RMSE below 0.005, with no
absolute-error bypass. At ordinary 4K, optimized output/state relative RMSE is
0.000429/0.000388; weak-decay output/state is 0.000537/0.000482.

## Production cleanup and long sequences

The production implementation has **one launch and two global phases**, not
one phase per source file. The original six conceptual stages were gate
preparation, KK, inverse, WY, state, and output; counting transpose separately
produced seven calls. In the current implementation scan/layout and shared KK/QK
run concurrently, followed by one global barrier and the persistent fused phase.
Per-chunk producer/consumer synchronization remains inside that phase.

The cleanup retains the validated production schedule, removes old staged
implementations and experimental inverse variants, removes `matmul_probe` and
timing/early-exit hooks, and removes the tuning options `GDN_MEGA_HEAD`,
`GDN_MEGA_DEFINITIONS`, and `GDN_MEGA_MANUAL_SYNC`. The useful build-scope option
`GDN_MEGA_ONLY` remains. Architecture guards and data-dependent correctness
paths remain; there are no algorithm tuning preprocessor switches.

Source is reduced from 19 implementation files to one `.cpp` entry point and
six included `.hpp` helpers, all used by the mega kernel. Separate public GDN
stage operators elsewhere in `csrc/kernel/` are independent APIs and are retained.

The cleaned production source contains **1,104 lines**, down from 7,126
(**84.5% fewer**). There are seven source files, all used. No new algorithm or
precision setting was introduced by this cleanup.

### Latency after cleanup

The same device, inputs, initial-state policy, and timing method are used as
above. “Original” is 10748eb; “Cleaned” is production cleanup commit **5141f56**.

| Tokens | Decay workload | Original ms | Cleaned ms | Speedup |
|---:|---|---:|---:|---:|
| 128 | Ordinary | 0.1112 | 0.0343 | 3.24x |
| 256 | Ordinary | 0.1474 | 0.0420 | 3.51x |
| 1,024 | Ordinary | 0.3123 | 0.0860 | 3.63x |
| 4,096 | Ordinary | 0.8688 | 0.2576 | 3.37x |
| 8,192 | Ordinary | 1.7197 | 0.5346 | 3.22x |
| 16,384 | Ordinary | 3.5022 | 1.0938 | 3.20x |
| 32,768 | Ordinary | 6.9650 | 2.2583 | 3.08x |
| 4,096 | Weak decay | 0.8663 | 0.2780 | 3.12x |
| 8,192 | Weak decay | 1.6848 | 0.5826 | 2.89x |
| 16,384 | Weak decay | 3.3755 | 1.1761 | 2.87x |
| 32,768 | Weak decay | 6.8061 | 2.4120 | 2.82x |

The 8K/16K/32K ordinary checkpoint timings before cleanup were
0.5334/1.1127/2.2994 ms. Across the ordinary sweep, cleanup changes latency by
−2.75% to +1.16%, with the same reported output and state error metrics. Treat
these small differences as measurement/compiler variation, not a new algorithmic
speedup. Weak-decay speedup at 32K is **2.82x**, below the 3x ordinary-workload
result; the long weak-decay rows are not claimed to meet a universal 3x target.

At 32K, cleaned output/state relative RMSE is **0.000430/0.000381** for
ordinary decay and **0.000538/0.000487** for weak decay. All are below the
unchanged 0.005 threshold.

### Allocation at longer lengths

Existing allocation shapes scale linearly for these aligned single-sequence
lengths. These are tensor-storage sizes, excluding allocator/runtime overhead;
the original and cleaned implementations require the same allocations.

| Tokens | Scratch/intermediates MiB | All supplied tensors MiB |
|---:|---:|---:|
| 4,096 | 136.875 | 178.406 |
| 8,192 | 265.500 | 347.406 |
| 16,384 | 522.750 | 685.406 |
| 32,768 | 1,037.250 | 1,361.406 |

### Validation and saved data

Final production build: **32 tests passed in 5.85 s**. All benchmark shapes
passed five checked launches against the CPU FP32 reference, checking both
output and final state. All seven head specializations are compiled; the tests
cover grouped/equal heads, tails, packed sequences, initial states, and
weak/no/mixed decay. Source checks found no unreferenced local header and no
remaining algorithm tuning guards. Python syntax and `git diff --check` pass.

Commands run from the repository root:

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh timeout 120 python -m pytest tests/test_gdn_mega.py -q
bash scripts/gdn_mega_docker.sh timeout 300 python benchmarking/gdn_mega.py \
  --initial-state --lengths 128 256 1024 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/branch_report_20261023/cleanup_initial.json
bash scripts/gdn_mega_docker.sh timeout 300 python benchmarking/gdn_mega.py \
  --initial-state --gate-scale 0.01 --lengths 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/branch_report_20261023/cleanup_weak.json
```

Original and pre-cleanup measurements used frozen uninstrumented libraries
from their respective checkpoints, selected per process through
`LD_LIBRARY_PATH`. Library SHA-256 fingerprints, precise measurements, and
error metrics are preserved in [measurements.json](measurements.json). Temporary
profiling builds were used only for the explicitly labeled phase table.

- [Original vs optimized checkpoint latency CSV](checkpoint_latency.csv)
- [Checkpoint phase FLOPs and timings CSV](checkpoint_stages.csv)
- [Original vs cleaned production latency CSV](production_latency.csv)
- [Raw measurements, environment, FLOP counts, and validation JSON](measurements.json)

The [head-count/sequence-length heatmap](heatmap.md) adds a full shape sweep and
explains executed FLOPs versus throughput and the persistent execution model.
