# PTO Kernel Performance Optimization Lessons

Lessons learned from optimizing the dynamic BSND chunkwise GatedDeltaNet
kernels on Ascend 910B2 using PTO-ISA C++.

## Hardware Architecture Essentials

The Ascend AI Core has **four independent processing pipes**:

| Pipe | Engine | Purpose |
|------|--------|---------|
| **Cube (M)** | Matrix multiply unit | GEMM operations (`TMATMUL`, `TMATMUL_ACC`) |
| **Vec (V)** | SIMD vector unit | Element-wise ops (`TADD`, `TMUL`, `TEXP`, etc.) |
| **MTE2** | DMA GM→L1/UB | Global memory loads (`TLOAD`, `copy_gm_to_ub`) |
| **MTE3** | DMA UB→GM | Global memory stores (`TSTORE`, `copy_ub_to_gm`) |

These pipes run **concurrently**. Performance comes from keeping all pipes
busy simultaneously.

### Memory Hierarchy

```
Global Memory (HBM, ~65 GB)
  └─ L1 Buffer (~1 MB, Cube input staging)
       └─ L0A / L0B (64 KB each, Cube operands)
       └─ L0C (256 KB, Cube accumulator)
  └─ Unified Buffer (UB, ~256 KB, Vec operands)
```

### Cross-Core Synchronization

- Cube and Vec are **separate cores** on the same AI Core
- They communicate through **cross-core flags** (`set_cross_flag` /
  `wait_flag_dev`) and shared GM workspace
- Flag-based synchronization is cheap but forces serialization at
  synchronization points

## Critical Performance Lessons

### 1. Scalar V→S Pipeline Stalls Were the Original #1 Bottleneck

**Problem**: `GetValue()` and `SetValue()` on UB tiles use the **Scalar
pipe (S)**, which requires explicit `set_flag(PIPE_V, PIPE_S)` /
`wait_flag(PIPE_V, PIPE_S)` transitions. Each transition stalls the
entire Vec pipe.

**Impact before the fix**: A loop of 128 `GetValue`+`SetValue` pairs costs
~5-10 μs per chunk. At 2048 chunks, that becomes 10-20 ms of pure
pipeline stalls, which was the dominant reason `scaled_dot_kkt`,
`wy_fast`, and `chunk_o` lagged badly in the original dynamic BSND path.

**Root cause in dynamic BSND**: The BSND layout `[B, S, H, D]` stores
heads interleaved. To extract per-head G values from `[C, H]` blocks,
we must gather every H-th element—requiring scalar loops since PTO-ISA
does not support:
- Cross-layout DMA (`TLOAD` only supports ND→ND, DN→DN, NZ→NZ)
- Strided single-element DMA (minimum row width = 32 bytes)
- Scatter/gather vector instructions

**What actually worked best**:
1. **Ensure data arrives in per-head-contiguous layout**. The current
   implementation stages `g_sum` / `beta` as contiguous `[H, T]`
   workspaces and then loads one per-head slice with a single-row DMA.
   This removed the dominant scalar extraction loops from
   `scaled_dot_kkt`, `wy_fast`, `chunk_h`, and `chunk_o`.
2. **Then revisit barriers and overlap**. Once the scalar path is gone,
   barrier narrowing and Cube/Vec overlap become second-order wins.
3. **Do not overinvest in clever scalar-loop reshaping first**. As long as
   the data is still fundamentally strided per head, scalar access tends
   to dominate anyway.

**Observed impact of contiguous per-head staging**:
- `scaled_dot_kkt`: ~15.5 ms -> ~4.7 ms
- `wy_fast`: ~16.8 ms -> ~6.9 ms
- `chunk_o`: ~26.2 ms -> ~11.1 ms
- `chunk_h`: ~14.2 ms -> ~9.7 ms

### 2. BSND Strided DMA Is 2-4x Slower Than Contiguous

**Problem**: Loading QKV tiles from BSND layout requires row stride =
`H * D = 2048` half-elements (4096 bytes) between rows, but each row is
only `D = 128` half-elements (256 bytes). The MTE2 engine issues one
burst per row, so 128 rows = 128 separate 256-byte bursts at 4096-byte
intervals.

**Comparison**: With BHSD layout (static baseline), the same data is
contiguous — one 32 KB burst DMA.

**Updated lesson**: Strided BSND QKV DMA is still real, but once
per-head-contiguous `g_sum` / `beta` staging is in place, it is no longer
the only story. The remaining QKV DMA penalty now shows up mostly in the
last few milliseconds of `chunk_o` and the small residual gap in
`scaled_dot_kkt`.

### 3. Cube-Vec Pipeline Balance Is Critical

**Problem**: If the Vec core takes much longer than the Cube core per
chunk iteration, the Cube sits idle waiting for the Vec cross-core signal.

**Original example**: In the pre-staging version of `scaled_dot_kkt`, the
Cube did a single GEMM (K^T@K) per chunk while the Vec side still paid
for G/Beta extraction, coefficient construction, workspace reload, and
store. That imbalance let the Cube sit idle waiting for the Vec signal.

**Current lesson**: After removing the scalar extraction path, the worst
remaining balance problem is no longer `scaled_dot_kkt`; it is the
multi-phase `chunk_o` pipeline, where one work item still contains two
Vec phases and two Cube phases with several GM workspace round-trips.

### 4. `pipe_barrier(PIPE_ALL)` Is Expensive, but Narrow It Carefully

**Problem**: `pipe_barrier(PIPE_ALL)` stalls **all** pipes until
completion. Use `pipe_barrier(PIPE_V)` when only Vec synchronization is
needed (most cases after SIMD operations).

**Practical lesson**:
- In the original `wy_fast`, once the scalar extraction path was removed,
  the old full barriers were no longer justified and could be narrowed to
  `PIPE_V`.
- In other places, a barrier that looks removable may provide little or no
  benefit once the rest of the pipeline is considered.

**Rule**: Replace `PIPE_ALL` only when the dependency is truly Vec-only and
after re-running correctness repeatedly. This is not a mechanical search
and replace.

### 5. TTRANS Has Significant Per-Call Overhead

**Attempted optimization**: Replace scalar GetValue/SetValue loops with
`pto::TTRANS` on `[H, H]` sub-blocks to transpose data in UB.

**Result**: 8 TTRANS + 8 TMOV operations (with `pipe_barrier(PIPE_V)`
between each) cost roughly the same as 128 scalar operations. Each
TTRANS + barrier costs ~0.6 μs, so 8 iterations = ~5 μs per chunk.

**Lesson**: TTRANS is useful for large square matrices, but for small
tiles (16×16) the per-operation overhead dominates. For the dynamic BSND
G/Beta problem, staging a contiguous per-head workspace was much more
effective than trying to synthesize the transpose with many small TTRANS
calls inside the hot loop.

### 6. DMA Double-Buffering Hides Latency

**Pattern from linear_attention**: Pre-load chunk i+1's data while
computing chunk i, using ping-pong buffers.

**Application**: `chunk_h` already pre-fetches K and G for the next
chunk (lines 336-351). `scaled_dot_kkt` uses workspace double-buffering
(slot = ci & 1). But `chunk_o` and `wy_fast` do not pipeline their
DMA loads.

### 7. UB Address Aliasing Enables Tight Memory Packing

**Pattern**: Reuse UB regions that are dead at different phases:
```cpp
constexpr int32_t GBlockUbAddr = AUbAddr;      // G block reuses A's space
constexpr int32_t BetaBlockUbAddr = CoeffUbAddr; // Beta reuses coeff space
constexpr int32_t AUbHalfAddr = GR2dUbAddr;     // Half-A reuses expanded-g space
```

**Rule**: Only alias buffers whose live ranges don't overlap. Document
the aliasing with comments.

### 8. Cross-Core Flag Rotation Prevents Stalls

**Pattern from linear_attention**:
```cpp
const int32_t flag_base = static_cast<int32_t>((work_idx & 3) * 6);
```

Rotating through 4 sets of flags prevents cross-iteration conflicts.
The GDN kernels use simpler 2-way rotation which is adequate for their
current pipeline depth but limits deeper pipelining.

### 9. Numerical Stability Has Performance Cost

**Example**: `scaled_dot_kkt` adds `min(0, g_row - g_col)` clamping
before `exp()` to prevent `Inf * 0 = NaN`. This requires:
```
TSUB → TSUB(negate) → TRELU → TSUB(negate) → TEXP
```
instead of the static baseline's:
```
TSUB → TEXP
```

**Better alternative**: `TMINS(coeff, coeff, 0.0f)` replaces
TSUB+TRELU+TSUB with a single instruction.

### 10. Simple Vecization Across Heads Can Beat Fancy Scan Designs

**Original expectation**: `chunk_cumsum` looked like it needed a full
Blelloch-style parallel prefix scan to become competitive.

**What worked instead**: Keep a `1 x H` running accumulator in UB and
process the chunk row-by-row with Vec ops:
```cpp
TMOV(acc_ub, g_row_0);
TMOV(s_row_0, g_row_0);
for (int32_t i = 1; i < valid; ++i) {
  TADD(s_row_i, acc_ub, g_row_i);
  TMOV(acc_ub, s_row_i);
}
```

This vectorizes across the already contiguous head dimension, which is
exactly where the scalar work used to be concentrated.

**Observed impact**: `chunk_cumsum` dropped from ~1.03 ms to ~0.18 ms,
which is about **5.3x faster** than the current Triton baseline
(`0.96 ms`) on the benchmark shape.

**Lesson**: For short scans where one dimension is already small and
contiguous, "vectorize across the contiguous dimension and keep a tiny
state tile" can be far more cost-effective than implementing a textbook
parallel scan.

### 11. Repeated Measurements Matter More After the Big Wins

Once the large bottlenecks are gone, run-to-run variance becomes more
visible:
- tiny kernels like `chunk_cumsum` can quantize strangely on single runs
- different kernels may move in opposite directions by a few tenths of a
  millisecond
- compiler flag sweeps that look good once may lose on the median

**Practical rule**: use repeated full benchmark runs and compare medians,
not one-off best cases, before keeping an optimization.

## Performance Reference Points

| Configuration | Total Latency | Total TFLOPS |
|:--|--:|--:|
| Triton baseline (BT=64, bf16) | 68.3 ms | 10.6 |
| **Dynamic BSND PTO (C=128, fp16)** | **32.6 ms** | **25.3** |
| Static BHSD PTO (C=128, fp16) | 40.7 ms | 20.2 |
| Linear attention PTO (peak) | — | 77.3 |

Per-kernel comparison (dynamic PTO vs Triton vs static PTO):

| Kernel | Dynamic PTO (ms) | Triton (ms) | Static PTO (ms) |
|:--|--:|--:|--:|
| chunk_cumsum | 0.18 | 0.96 | 1.28 |
| scaled_dot_kkt | 4.67 | 4.79 | 9.07 |
| wy_fast | 6.92 | 15.59 | 9.61 |
| chunk_h | 9.68 | 30.83 | 9.14 |
| chunk_o | 11.13 | 16.12 | 11.63 |

Current state: the dynamic PTO path now beats Triton on **every** kernel,
and also beats the current static PTO baseline overall on this benchmark
shape. The main remaining performance headroom is concentrated in
`chunk_o`, plus smaller pipeline/overlap opportunities in
`scaled_dot_kkt`.

## Public API Compatibility Constraint

PTO kernels must be **drop-in replacements** for Triton kernels:
- Accept `[B, S, H, D]` (BSND) layout tensors
- Accept `cu_seqlens` (int32) for variable-length sequences
- Same Python function signatures in `dynamic_kernel_libs.py`

**Current practical compromise**:
- The **public API is unchanged**, so callers still pass BSND-layout
  tensors and the same Python entry points.
- The current optimized implementation materializes temporary contiguous
  `[H, T]` workspaces for `g_sum` / `beta` inside the runtime helpers
  before launching the hot kernels.

This preserves drop-in usability, but it is not the same as a pure
"all layout work happens inside PTO" design. Long term, the remaining
cleanup item is to move that staging fully into PTO (either in-kernel or
as an explicit preprocess kernel) while keeping the same public API.
