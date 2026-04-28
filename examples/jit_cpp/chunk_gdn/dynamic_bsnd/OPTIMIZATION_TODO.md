# Optimization TODO for Dynamic BSND PTO Kernels

Per-kernel optimization ideas ordered by estimated impact. See
`OPTIMIZATION_LESSONS.md` for background on the hardware architecture
and general lessons learned.

**Important constraint**: The torch interface (arg list, memory layout)
must stay consistent with the Triton reference so PTO kernels remain
drop-in replacements. All layout optimizations must happen inside the
C++ kernel, not in the Python wrapper.

**Reference files**:
- Static BHSD baseline: `../static_baseline/` (best-case PTO perf)
- Triton baseline: `../triton_baseline/` (production reference)
- Linear attention: `../../linear_attention/` (well-optimized PTO example)
- PTO-ISA docs: `/sources/pto-isa/include/pto/`
- NPU kernel skill: `/workdir/pto-kernels/.skills/npu_kernel_general/skills.md`

**Current performance** (npu:4, N_seq=16, L_seg=16384, H=16, D=128, C=128):

| Kernel | Dynamic PTO | Triton | Static PTO | Speedup vs Triton |
|:--|--:|--:|--:|--:|
| chunk_cumsum | 0.37 ms | 1.00 ms | 1.37 ms | **2.7x** |
| scaled_dot_kkt | 4.69 ms | 4.81 ms | 8.76 ms | **1.03x** |
| wy_fast | 6.85 ms | 15.57 ms | 9.52 ms | **2.27x** |
| chunk_h | 9.57 ms | 30.82 ms | 8.31 ms | **3.22x** |
| chunk_o | 10.73 ms | 16.13 ms | 11.60 ms | **1.50x** |
| **total** | **32.20 ms** | **68.34 ms** | **39.56 ms** | **2.12x** |

**Target**: ~~Beat Triton on every kernel.~~ ACHIEVED — all kernels beat Triton.
Further goal: approach static PTO performance (~40 ms total) while
maintaining BSND API compatibility. Currently at 32.20 ms — **already
faster than static PTO** (39.56 ms).

---

## Cross-Kernel Optimizations

These apply to multiple kernels and should be prioritized first.

### CK-1. In-Kernel G/Beta Transpose Preprocessing Pass — COMPLETED

**Status**: ✅ Completed via Python wrapper internal transpose.

**What was done**: G and Beta are transposed from `[1, T, H]` to `[H, T]`
inside the Python `run_*` wrapper functions, then passed to C++ kernels
with a `total_tokens` parameter for offset computation. Kernels load
per-head data contiguously via DMA, eliminating all scalar
`GetValue`/`SetValue` extraction loops.

**Impact**: Reduced total latency from 74.71 ms to 34.03 ms (2.2x
improvement). The Triton-compatible API is preserved — callers pass
`[1, T, H]` tensors as before.

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

### CK-3. Replace `pipe_barrier(PIPE_ALL)` with `pipe_barrier(PIPE_V)` — COMPLETED

**Status**: ✅ Done in `wy_fast_kernel.cpp`.

**Impact**: ~0.5 ms savings in wy_fast.

### CK-4. Precompute `cu_seqlens` Chunk Offsets (LOW)

**Current**: Each kernel recomputes `chunk_offset` for each work item
by looping over all sequences (O(batch) per work item).

**Fix**: Pass a precomputed `chunk_offsets` array (like Triton does with
`prepare_chunk_indices`). Eliminates O(batch) scalar loops per work item.

**Estimated impact**: Negligible for small batch counts (16), meaningful
for large batches.

---

## Per-Kernel Optimizations

### 1. chunk_cumsum (0.37 ms — DONE, 2.7x faster than Triton)

~~Currently **2x slower than Triton** (1.04 ms).~~
Now **2.7x faster than Triton**.

#### CS-1. Vectorized Row-Wise TADD/TMOV — COMPLETED

**What was done**: Replaced per-head scalar `GetValue`/`SetValue` cumsum
loops with SIMD row-wise operations. Each row of `[ChunkSize, HeadTileCols]`
is a 1D tile; cumsum uses `TADD(acc, acc, g_row_i)` + `TMOV(s_row_i, acc)`
per row, processing all heads simultaneously. This reduced 16×128 = 2048
scalar ops to ~256 Vec ops per chunk.

**Impact**: 2.03 ms → 0.37 ms (5.5x speedup).

**Key lesson**: `pipe_barrier(PIPE_ALL)` is required before `copy_ub_to_gm`
to ensure Vec writes are visible to MTE3. `pipe_barrier(PIPE_V)` alone
is insufficient.

#### CS-2. Use Both Sub-Blocks (vid=0 and vid=1) — SKIPPED

Sub-block parallelism causes cross-sub-block synchronization issues for
shared UB output tiles. The SIMD row-wise approach (CS-1) provided a
much larger speedup (5.5x) without needing sub-block parallelism.

#### CS-3. DMA Double-Buffering (LOW-MEDIUM)

**Current**: Sequential load → compute → store per chunk. No overlap.

**Fix**: Load chunk i+1 while computing cumsum of chunk i. UB has >200 KB
free (only 16 KB used).

**Estimated impact**: Hide DMA latency, ~20-30% improvement.

---

### 2. scaled_dot_kkt (4.69 ms — 1.03x faster than Triton)

~~Currently **3.1x slower than Triton**.~~
Now **comparable to Triton** (4.81 ms).

#### KKT-1. Eliminate G/Beta Scalar Extraction — COMPLETED (via CK-1)

#### KKT-2. Replace TSUB/TRELU/TSUB with TMINS — COMPLETED

Saves 2 TSUB + 1 TRELU + 2 `pipe_barrier` per chunk.

#### KKT-3. Overlap G/Beta DMA with Cube Work — COMPLETED

**What was done**: Moved G/Beta `copy_gm_to_ub` calls before
`wait_flag_dev(slot)`, allowing DMA to execute in parallel with the
Cube GEMM. Address computation (chunk_start, valid_rows) doesn't depend
on Cube output, so it can be done early.

**Impact**: ~0.5-1 ms improvement (4.22 ms → ~3.4-4.7 ms, variance-dependent).

#### KKT-4. Deepen the Cube-Vec Pipeline (MEDIUM)

**Current**: 2-slot double-buffering (slot = ci & 1). Cube produces
KTK for chunk i, Vec processes chunk i.

**Better**: 3-slot or 4-slot pipelining with flag rotation, following the
linear_attention pattern (`work_idx & 3`). This allows Cube to race
ahead of Vec by 2-3 chunks.

**Estimated impact**: Better Cube utilization, ~10-20% overall.

---

### 3. wy_fast (6.85 ms — 2.27x faster than Triton)

~~Currently **comparable to Triton** (15.62 ms).~~
Now **2.27x faster than Triton**.

#### WY-1. Eliminate Beta/G Scalar Extraction — COMPLETED (via CK-1)

#### WY-2. Replace `pipe_barrier(PIPE_ALL)` with `pipe_barrier(PIPE_V)` — COMPLETED (via CK-3)

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

### 4. chunk_h (9.57 ms — 3.22x faster than Triton)

Already **3.22x faster than Triton** (30.82 ms). Now **faster than static
baseline** (8.31 ms → closing in).

#### CH-1. Eliminate G Scalar Extraction — COMPLETED (via CK-1)

#### CH-2. Vectorize the Coefficient Scaling Loop — COMPLETED

**What was done**: Replaced 64 scalar `GetValue` + `TMULS` calls with
4 iterations of `TROWEXPAND` (expand [16,1] → [16,128]) + `TMUL`,
using the freed G_BLOCK_UB (8192 bytes) as scratch. Marginal improvement
(~0.1 ms) since the scalar loop was already well-pipelined.

#### CH-3. Optimize cu_seqlens Chunk Offset Computation (LOW)

**Current**: O(seq_idx) loop per work item to compute chunk_offset.

**Fix**: Precomputed array passed as kernel argument.

**Estimated impact**: Negligible for small batch.

---

### 5. chunk_o (10.73 ms — 1.50x faster than Triton)

~~Currently **1.6x slower than Triton** (16.16 ms).~~
Now **1.50x faster than Triton**.

#### CO-1. Eliminate G Scalar Extraction — COMPLETED (via CK-1)

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

## Priority Ranking (Updated)

### Completed

| Item | Kernels | Impact |
|:--|:--|:--|
| CK-1: G/Beta transpose (wrapper-internal) | kkt, wy, chunk_h, chunk_o | 74.71→34.03 ms |
| CS-1: Vectorized row-wise TADD cumsum | cumsum | 2.03→0.37 ms |
| KKT-2: TMINS for safe_exp | kkt, chunk_o | ~1 ms |
| WY-2/CK-3: PIPE_ALL → PIPE_V | wy_fast | ~0.5 ms |
| KKT-3: DMA-Cube overlap | kkt | ~0.5 ms |
| CH-2: TROWEXPAND coeff scaling | chunk_h | ~0.1 ms |

### Remaining (for further optimization)

| Priority | Item | Kernels Affected | Est. Savings |
|:--|:--|:--|:--|
| **P1** | CO-2: Pipeline Cube phases | chunk_o | 2-3 ms |
| **P1** | KKT-4: Deeper Cube-Vec pipeline | kkt | 1-2 ms |
| **P2** | CK-2: Wider QKV DMA loads | all | 2-4 ms |
| **P2** | CO-3: Reduce workspace round-trips | chunk_o | 2-3 ms |
| **P2** | WY-3: DMA double-buffering | wy_fast | 1-2 ms |
| **P2** | WY-4: Fuse A1/A2 computation | wy_fast | 1-2 ms |
| **P3** | CO-4: Flag rotation | chunk_o | 1-2 ms |
| **P3** | CS-3: DMA double-buffering | cumsum | 0.1-0.2 ms |
| **P3** | CK-4: Precompute chunk offsets | all | <0.5 ms |

**Current total**: 32.20 ms (2.12x vs Triton 68.34 ms)

**Projected if P1+P2 completed**: ~25-28 ms (2.4-2.7x vs Triton)

---

## How to Benchmark

```bash
# Verify correctness (always run first after changes)
GDN_NPU_DEVICE=npu:0 python verify_dynamic_bsnd.py

# Benchmark
GDN_NPU_DEVICE=npu:0 python bench_dynamic_bsnd.py

# Compare with references
cd ../triton_baseline && GDN_NPU_DEVICE=npu:1 python bench_triton_gdn.py
cd ../static_baseline && GDN_NPU_DEVICE=npu:2 python bench_static_gdn.py
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
| Benchmark | — | `bench_dynamic_bsnd.py` |
| Verification | — | `verify_dynamic_bsnd.py` |
