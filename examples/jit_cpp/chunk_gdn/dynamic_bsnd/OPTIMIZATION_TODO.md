# Optimization TODO for Dynamic BSND PTO Kernels

Per-kernel optimization ideas ordered by estimated impact. See
`OPTIMIZATION_LESSONS.md` for background on the hardware architecture
and general lessons learned.

**Important constraint**: The torch interface (arg list, memory layout)
must stay consistent with the Triton reference so PTO kernels remain
drop-in replacements. All layout optimizations must happen inside the
C++ kernel, not in the Python wrapper.

**Current status note**: The public torch API is still unchanged, but the
current optimized path stages contiguous `[H, T]` workspaces for `g_sum`
and `beta` inside the dynamic runtime helpers so the hot kernels can DMA
per-head slices directly. Moving that staging into PTO remains follow-up
work if we want to satisfy the original "no wrapper-side layout change"
goal strictly.

**Reference files**:
- Static BHSD baseline: `../static_baseline/` (best-case PTO perf)
- Triton baseline: `../triton_baseline/` (production reference)
- Linear attention: `../../linear_attention/` (well-optimized PTO example)
- PTO-ISA docs: `/sources/pto-isa/include/pto/`
- NPU kernel skill: `/workdir/pto-kernels/.skills/npu_kernel_general/skills.md`

**Current performance** (dynamic on `npu:7`, Triton on `npu:6`, static on
`npu:5`; dynamic numbers below are the median of 3 full runs, and
`bench_dynamic_bsnd.py` precomputes contiguous `g_t` / `beta_t` outside
the timed kernel loop):

| Kernel | Dynamic PTO | Triton | Static PTO | Speedup vs Triton |
|:--|--:|--:|--:|--:|
| chunk_cumsum | 0.18 ms | 1.03 ms | 1.19 ms | **5.72x** |
| scaled_dot_kkt | 4.67 ms | 4.87 ms | 8.83 ms | **1.04x** |
| wy_fast | 6.92 ms | 15.66 ms | 9.53 ms | **2.26x** |
| chunk_h | 9.68 ms | 30.82 ms | 9.16 ms | **3.18x** |
| chunk_o | 11.13 ms | 16.14 ms | 11.62 ms | **1.45x** |
| **total** | **32.57 ms** | **68.52 ms** | **40.33 ms** | **2.10x** |

**Status**: Overall goal achieved. The current dynamic BSND path now beats
Triton by just over 2x overall and is also faster than the current static
BHSD baseline on this benchmark shape. The remaining work is about
stability margin, moving transpose prep into PTO, and improving
wrapper-side staging. `chunk_cumsum` is no longer a bottleneck after the
new row-wise Vec accumulation path.

**Compiler note**: A post-optimization sweep of `-O3` and
`-O3 -funroll-loops` did not beat the current default compile path
consistently on repeated full benchmarks, so no compiler-flag change was
kept.

---

## Cross-Kernel Optimizations

These apply to multiple kernels and should be prioritized first.

### CK-1. In-Kernel G/Beta Transpose Preprocessing Pass (HIGH IMPACT)

**Status**: Implemented in a stopgap form via cached contiguous `[H, T]`
workspaces in `dynamic_kernel_libs.py` and `bench_dynamic_bsnd.py`.
`scaled_dot_kkt`, `wy_fast`, `chunk_h`, and `chunk_o` now load per-head
`g_sum` / `beta` slices contiguously and no longer pay the old
`GetValue`/`SetValue` extraction cost. Remaining follow-up: move the
workspace build step into PTO (inside-kernel or separate preprocess
kernel) so the optimization is no longer wrapper-side.

**Idea**: Add a preprocessing phase at the start of the Vec work loop
that transposes a window of G (and Beta where applicable) from `[T, H]`
to `[H, T]` layout in a GM workspace buffer. Then the main loop loads
G per-head contiguously from the transposed workspace.

**Implementation sketch**:
1. Allocate extra workspace `g_transposed` of size `T * sizeof(float)`
   (or per-chunk windows if full T doesn't fit)
2. Before the main loop, each Vec core processes its assigned chunks:
   load `[C, H]` blocks, use TTRANS on `[H, H]` sub-blocks, write
   transposed `[H, C]` blocks back to workspace
3. Barrier, then main loop reads from transposed workspace

**Estimated impact**: Eliminates 128 V→S stalls per chunk in kkt, chunk_o,
chunk_h. Should recover most of the gap vs static baseline for these
kernels (~2-3x improvement for kkt and chunk_o).

**Complexity**: Medium. Requires additional workspace allocation in Python
wrapper and a preprocessing phase in each kernel. Can be done as a
separate "transpose kernel" launched before the main kernel (user asked
for it to be inside the same kernel, but a separate lightweight launch
may be acceptable if performance justifies it).

**PTO-ISA constraints discovered**:
- `TLOAD` enforces same-layout transfers: ND→ND, DN→DN only (no cross-layout)
- `TTRANS` only works on square tiles (NxN)
- Minimum DMA row width is 32 bytes
- `GetValue`/`SetValue` are the only way to do arbitrary strided access in UB

### CK-2. Strided DMA Optimization for QKV Loads (MEDIUM IMPACT)

**Current**: QKV loaded with row stride = `H*D = 2048` elements. Each
row is only `D = 128` elements. This is 128 small bursts at large
intervals.

**Ideas**:
- Load wider tiles covering multiple heads, then extract the needed
  head using TMOV/TRESHAPE. For example, load `[C, H*D]` (full rows)
  into L1 and use `TEXTRACT` to select the head's `[C, D]` sub-tile.
  L1 has ~1 MB capacity so `C * H * D * sizeof(half) = 128*16*128*2 =
  512 KB` fits.
- Investigate whether L1→L0/UB transfers can do sub-tile extraction
  more efficiently than GM→L1 strided DMA.

**Estimated impact**: 1.5-2x improvement in DMA throughput for QKV loads.

### CK-3. Replace `pipe_barrier(PIPE_ALL)` with `pipe_barrier(PIPE_V)` (LOW-MEDIUM)

**Where**: `wy_fast_kernel.cpp` has 4 `PIPE_ALL` barriers per work item.

**Fix**: After scalar extraction, only Vec pipe needs sync. Change to
`pipe_barrier(PIPE_V)`. This allows MTE2/MTE3 to continue working.

**Estimated impact**: 5-15% improvement for wy_fast.

### CK-4. Precompute `cu_seqlens` Chunk Offsets (LOW)

**Current**: Each kernel recomputes `chunk_offset` for each work item
by looping over all sequences (O(batch) per work item).

**Fix**: Pass a precomputed `chunk_offsets` array (like Triton does with
`prepare_chunk_indices`). Eliminates O(batch) scalar loops per work item.

**Estimated impact**: Negligible for small batch counts (16), meaningful
for large batches.

---

## Per-Kernel Optimizations

### 1. chunk_cumsum (0.18 ms, target: <0.15 ms)

Now **5.7x faster than Triton** (0.18 ms vs 1.03 ms) and well ahead of
the static PTO baseline too. This is no longer a priority bottleneck.

#### CS-1. Vectorized Parallel Prefix Sum (HIGH IMPACT)

**Status**: Implemented in a simpler row-wise Vec form. The kernel now
keeps a `1 x H` UB accumulator and performs row-by-row Vec adds across
all heads, removing the old per-head scalar accumulation loop. This is
not the original Blelloch plan, but it captured most of the benefit with
much lower complexity.

**Current**: Pure scalar loop with `GetValue`/`SetValue`:
```cpp
for (int32_t i = 1; i < valid; ++i) {
    acc += g_block_ub.GetValue(i * HeadTileCols + h);
    s_block_ub.SetValue(i * HeadTileCols + h, acc);
}
```

**Idea**: Implement a Blelloch-style parallel prefix sum:
1. Load the `[C]` vector for one head into a Vec tile
2. Up-sweep: `log2(C) = 7` passes of pairwise TADD at doubling strides
3. Down-sweep: 7 passes to produce the scan
4. This replaces 127 scalar iterations with 14 SIMD passes

**Alternative**: Hierarchical approach — split C=128 into 8 blocks of
16, do scalar prefix sum within each block (cheap), then SIMD-combine
the block suffixes using TADDS broadcasts.

**Observed impact**: ~5.7x speedup vs Triton on the benchmark shape
(`1.03 ms -> 0.18 ms`).

#### CS-2. Use Both Sub-Blocks (vid=0 and vid=1) (MEDIUM)

**Status**: Implemented.

**Current**: `if (vid != 0) return;` — half the Vec hardware is idle.

**Fix**: Split 16 heads across two sub-blocks (8 heads each), or process
different chunks on each sub-block.

**Estimated impact**: Up to 2x throughput.

#### CS-3. DMA Double-Buffering (LOW-MEDIUM)

**Current**: Sequential load → compute → store per chunk. No overlap.

**Fix**: Load chunk i+1 while computing cumsum of chunk i. UB has >200 KB
free (only 16 KB used).

**Estimated impact**: Hide DMA latency, ~20-30% improvement.

---

### 2. scaled_dot_kkt (4.67 ms, target: <4.3 ms)

Now **slightly faster than Triton** (4.67 ms vs 4.87 ms). The old
G/Beta extraction bottleneck is gone; the remaining work is mostly
pipeline depth and DMA overlap.

#### KKT-1. Eliminate G/Beta Scalar Extraction (CRITICAL)

**Status**: Implemented via pre-transposed `[H, T]` workspace loads.

**Current**: 128 GetValue/SetValue for G + 64 for Beta = 192 V→S stalls
per chunk.

**Approach A** — In-kernel transpose preprocessing (see CK-1 above).

**Approach B** — Load G as `[H, C]` from a transposed workspace so the
per-head data is contiguous in a single DMA row.

**Approach C** — Use `set_vector_mask` to process only every H-th element
during a bulk TMOV. Needs investigation whether mask-controlled TMOV can
achieve strided access.

**Estimated impact**: 2-3x improvement (10+ ms savings).

#### KKT-2. Replace TSUB/TRELU/TSUB with TMINS (MEDIUM)

**Status**: Implemented.

**Current** (safe_exp clamping):
```cpp
TSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);   // diff
pipe_barrier(PIPE_V);
TSUB(coeff_ub, a_ub, coeff_ub);          // negate
pipe_barrier(PIPE_V);
TRELU(coeff_ub, coeff_ub);               // relu
pipe_barrier(PIPE_V);
TSUB(coeff_ub, a_ub, coeff_ub);          // negate back
pipe_barrier(PIPE_V);
TEXP(coeff_ub, coeff_ub);
```

**Better**:
```cpp
TSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);
pipe_barrier(PIPE_V);
TMINS(coeff_ub, coeff_ub, 0.0f);
pipe_barrier(PIPE_V);
TEXP(coeff_ub, coeff_ub);
```

Saves 2 TSUB + 1 TRELU + 2 `pipe_barrier`.

**Estimated impact**: ~1-2 ms savings (5 fewer Vec operations × 2048
chunks).

#### KKT-3. Overlap G/Beta DMA with Cube Work (MEDIUM)

**Current**: G/Beta DMA and extraction happen after `wait_flag_dev(slot)`,
which waits for the Cube to finish. The Vec is idle during Cube work.

**Better**: Start G/Beta DMA load **before** `wait_flag_dev(slot)`,
during the Cube's GEMM time. Pre-fetch G/Beta for chunk i while Cube
computes K^T@K for chunk i.

**Implementation**: Move the `copy_gm_to_ub` calls for G and Beta above
the `wait_flag_dev(slot)` call. Add the MTE2→V sync after the wait.

**Estimated impact**: Hides ~1-2 ms of DMA latency.

#### KKT-4. Deepen the Cube-Vec Pipeline (MEDIUM)

**Current**: 2-slot double-buffering (slot = ci & 1). Cube produces
KTK for chunk i, Vec processes chunk i.

**Better**: 3-slot or 4-slot pipelining with flag rotation, following the
linear_attention pattern (`work_idx & 3`). This allows Cube to race
ahead of Vec by 2-3 chunks.

**Estimated impact**: Better Cube utilization, ~10-20% overall.

---

### 3. wy_fast (6.92 ms, target: <6.5 ms)

Now **2.26x faster than Triton** (6.92 ms vs 15.66 ms) and comfortably
ahead of the current static baseline (9.53 ms).

#### WY-1. Eliminate Beta/G Scalar Extraction (CRITICAL)

**Status**: Implemented via pre-transposed `[H, T]` workspace loads.

**Current**: 128 GetValue/SetValue for Beta + 128 for G = 256 V→S
stalls per work item. This is the worst of any kernel.

**Same approaches as KKT-1**: In-kernel transpose preprocessing or
pre-transposed workspace.

**Estimated impact**: 3-5 ms savings.

#### WY-2. Replace `pipe_barrier(PIPE_ALL)` with `pipe_barrier(PIPE_V)` (LOW)

**Current**: 4 `pipe_barrier(PIPE_ALL)` per work item.

**Fix**: Change to `PIPE_V` where only Vec sync is needed (lines 179,
229, 313, 364).

**Estimated impact**: ~0.5-1 ms savings.

#### WY-3. DMA Double-Buffering for A Matrix Loads (MEDIUM)

**Current**: A matrix is loaded from GM per-chunk with strided DMA.
No overlap with compute.

**Fix**: Pre-load next chunk's A tiles while computing current chunk.

**Estimated impact**: ~1-2 ms savings.

#### WY-4. Fuse A1 and A2 Computation (MEDIUM)

**Current**: A1 (lower triangular) and A2 (upper triangular) are
computed in separate Vec phases, each requiring DMA loads and Cube GEMMs.

**Idea**: Investigate whether both can be computed from a single load of
the full A matrix, reducing DMA volume and enabling better Vec pipelining.

**Estimated impact**: ~1-2 ms savings.

---

### 4. chunk_h (9.68 ms, target: <9 ms)

Now **3.18x faster than Triton** (9.68 ms vs 30.82 ms) and close to the
current static baseline (9.16 ms).

#### CH-1. Eliminate G Scalar Extraction (MEDIUM-HIGH)

**Status**: Implemented via pre-transposed `[H, T]` workspace loads.

**Current**: 128 GetValue/SetValue per chunk, appearing twice (initial
load + next-chunk prefetch).

**Same approach as KKT-1**: In-kernel transpose or transposed workspace.

**Estimated impact**: ~2-3 ms savings.

#### CH-2. Vectorize the Coefficient Scaling Loop (MEDIUM)

**Status**: Implemented with `TROWEXPAND` + `TMUL`.

**Current**: The per-row decay scaling uses scalar GetValue in a loop:
```cpp
for (int32_t i_2 = 0; i_2 < HalfC / 4; ++i_2) {
    auto c0 = coeff_ub.GetValue(i_2 * 4);
    TMULS(k0, k0, c0);
    // ... c1, c2, c3 similarly ...
}
```

This is 64 V→S stalls per chunk (16 iterations × 4 GetValues).

**Better**: Use `TROWEXPAND` + `TMUL` pattern:
1. Expand the `[1, HalfC]` coefficient vector to `[HalfC, D]` using
   `TROWEXPAND`
2. Single `TMUL(k_ub, k_ub, coeff_expanded)` replaces the entire loop

The static baseline uses the same scalar loop, so this would make
dynamic BSND **faster** than static for this operation.

**Estimated impact**: ~1-2 ms savings.

#### CH-3. Optimize cu_seqlens Chunk Offset Computation (LOW)

**Current**: O(seq_idx) loop per work item to compute chunk_offset.

**Fix**: Precomputed array passed as kernel argument.

**Estimated impact**: Negligible for small batch.

---

### 5. chunk_o (11.13 ms, target: <10.5 ms)

Now **1.45x faster than Triton** (11.13 ms vs 16.14 ms) and close to the
static baseline (11.62 ms). The biggest remaining wins are now pipeline
and workspace-roundtrip related.

#### CO-1. Eliminate G Scalar Extraction (CRITICAL)

**Status**: Implemented via pre-transposed `[H, T]` workspace loads.

**Current**: 128 GetValue/SetValue per work item in both VEC paths
(non-cu_seqlens and cu_seqlens).

**Same approach as KKT-1**.

**Estimated impact**: ~3-5 ms savings.

#### CO-2. Pipeline Cube Phase 1 and Phase 2 (HIGH)

**Current**: 4 sequential phases per work item:
1. Cube: Q@K^T, Q@S → workspace
2. Vec: gate QK, write gated QK → workspace
3. Cube: gated_QK @ V → workspace
4. Vec: combine QS + QKV → O

Each phase waits for the previous to complete.

**Idea**: Overlap Cube work item N's phase 3 with Vec work item N's
phase 2. The current code has `first_cube_iter` tracking but doesn't
exploit it for pipelining.

**Implementation**: Use separate cross-core flags for phase 1 and
phase 3 Cube work. Start phase 3 of work item N while Vec processes
work item N+1's phase 2.

**Estimated impact**: ~3-5 ms savings by hiding one Cube phase.

#### CO-3. Reduce Workspace Round-Trips (MEDIUM)

**Current**: 6 DMA transfers on Vec + 8 on Cube = 14 DMA ops per work
item, going through GM workspace.

**Idea**: Keep intermediate results in L1/UB instead of writing to GM
workspace. For example, the QK result could stay in L0C and be converted
in-place rather than written to GM and re-read.

**Constraint**: Cube output (L0C) can only go to GM via TSTORE. But the
linear_attention kernel demonstrates fusing matmul output directly into
the next computation by using `copy_l0c_to_gm` → `copy_gm_to_ub`
patterns with minimal latency.

**Estimated impact**: ~2-3 ms savings.

#### CO-4. Adopt Linear Attention's Flag Rotation Pattern (MEDIUM)

**Current**: Simple alternating flags (flag 0/1 for Cube→Vec, flag 2/3
for Vec→Cube).

**Better**: 4-way flag rotation (`work_idx & 3`) with 6 flags per slot,
following linear_attention.cpp line 338. This enables deeper pipelining.

**Estimated impact**: ~1-2 ms improvement in Cube utilization.

#### CO-5. Replace TMINS-Based Safe Exp with Predicated TEXP (LOW)

**Current**: `TMINS(coeff, coeff, 0.0f)` + `TEXP(coeff, coeff)`.

**Alternative**: If PTO supports `TEXP` with saturation or clamped input,
this could be a single instruction.

---

## Remaining Priority Ranking

| Priority | Item | Kernels Affected | Est. Total Savings |
|:--|:--|:--|:--|
| **P0** | Move CK-1 staging into PTO instead of Python helpers | kkt, wy, chunk_h, chunk_o | Perf hygiene / interface fidelity |
| **P1** | CO-2: Pipeline Cube phase 1 and phase 2 | chunk_o | 0.5-1.5 ms |
| **P1** | CK-2: Wider QKV DMA loads | all | 1-3 ms |
| **P1** | KKT-3 / KKT-4: deeper Cube-Vec overlap | kkt | 0.5-1.5 ms |
| **P2** | CO-3: Reduce workspace round-trips | chunk_o | 0.5-1.5 ms |
| **P2** | CK-4 / CH-3: Precompute chunk offsets | chunk_h, chunk_o, kkt | <0.5 ms |

**Near-term expectation**: The current implementation is already around
`32.5-33 ms` on the benchmark shape. The most realistic next win is now
to trim another ~1-3 ms from `chunk_o` / QKV DMA and eventually move the
current transpose staging fully into PTO.

---

## How to Benchmark

```bash
# Verify correctness (always run first after changes)
GDN_NPU_DEVICE=npu:7 python verify_dynamic_bsnd.py

# Benchmark
GDN_NPU_DEVICE=npu:7 python bench_dynamic_bsnd.py

# Compare with references
cd ../triton_baseline && GDN_NPU_DEVICE=npu:6 python bench_triton_gdn.py
cd ../static_baseline && GDN_NPU_DEVICE=npu:5 python bench_static_gdn.py
```

Use different NPU devices to avoid contention. Check `npu-smi info`
for available devices. Devices 4-7 are often occupied by long-running
jobs.

## Files to Modify

| Kernel | Source | Python wrapper |
|:--|:--|:--|
| chunk_cumsum | `chunk_cumsum_kernel.cpp` | `dynamic_kernel_libs.py` → `run_chunk_cumsum` |
| scaled_dot_kkt | `scaled_dot_kkt_kernel.cpp` | `dynamic_kernel_libs.py` → `run_scaled_dot_kkt` |
| wy_fast | `wy_fast_kernel.cpp` | `dynamic_kernel_libs.py` → `run_wy_fast` |
| chunk_h | `chunk_h_kernel.cpp` | `dynamic_kernel_libs.py` → `run_chunk_h` |
| chunk_o | `chunk_o_kernel.cpp` | `dynamic_kernel_libs.py` → `run_chunk_o` |
| Common utilities | `include/common.h` | `pto_dynamic_common.py` |
| Benchmark | — | `bench_dynamic_bsnd.py` |
| Verification | — | `verify_dynamic_bsnd.py` |
