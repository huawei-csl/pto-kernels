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

### 1. Scalar V→S Pipeline Stalls Are the #1 Bottleneck

**Problem**: `GetValue()` and `SetValue()` on UB tiles use the **Scalar
pipe (S)**, which requires explicit `set_flag(PIPE_V, PIPE_S)` /
`wait_flag(PIPE_V, PIPE_S)` transitions. Each transition stalls the
entire Vec pipe.

**Impact**: A loop of 128 `GetValue`+`SetValue` pairs costs ~5-10 μs per
chunk. At 2048 chunks, that's 10-20 ms of pure pipeline stalls—dominating
the total kernel time for `scaled_dot_kkt` (15.5 ms → 4.7 ms after fix)
and `chunk_o` (26.2 ms → 10.7 ms after fix).

**Root cause in dynamic BSND**: The BSND layout `[B, S, H, D]` stores
heads interleaved. To extract per-head G values from `[C, H]` blocks,
we must gather every H-th element—requiring scalar loops since PTO-ISA
does not support:
- Cross-layout DMA (`TLOAD` only supports ND→ND, DN→DN, NZ→NZ)
- Strided single-element DMA (minimum row width = 32 bytes)
- Scatter/gather vector instructions

**Solution applied**: Transpose G/Beta from `[1, T, H]` to `[H, T]`
inside the Python `run_*` wrapper functions. C++ kernels then load
per-head data contiguously from the transposed layout using a
`total_tokens` offset parameter. This eliminated all scalar extraction
loops while preserving the Triton-compatible API (callers still pass
`[1, T, H]` tensors).

**Overall impact**: 74.71 ms → 34.03 ms (2.2x improvement).

### 2. Vectorize Scalar Loops with SIMD Row Operations

**Problem**: Even after eliminating strided G/Beta extraction, some
kernels still used scalar `GetValue`/`SetValue` for element-wise
operations (e.g., cumsum, coefficient scaling).

**Solution for cumsum**: Replace per-head sequential scalar cumsum with
row-wise SIMD operations. Create 1D tile views (`TileUbDataND<float,
1, H, 1, H>`) for each row of the `[C, H]` UB tile using `TASSIGN`
with runtime-computed addresses (`GUbAddr + i * RowBytes`). Then use
`TADD(acc, acc, g_row_i)` and `TMOV(s_row_i, acc)` to process all
heads simultaneously per row.

**Impact**: 2.03 ms → 0.37 ms (5.5x speedup). Replaced 16×128 = 2048
scalar ops with ~256 Vec ops per chunk.

**Solution for coefficient scaling (chunk_h)**: Replace 64 scalar
`GetValue` + `TMULS` calls with 4 iterations of `TROWEXPAND` (expand
`[16, 1]` DN → `[16, 128]` ND) + `TMUL`. Reused the freed G_BLOCK_UB
region (8192 bytes) as scratch for the expansion tile. Impact was
marginal (~0.1 ms) since the scalar loop was already well-pipelined
with unrolling.

**Key lesson**: `TASSIGN` works with runtime-computed addresses in loops.
The compiler treats it as metadata assignment, not an instruction. This
enables creating tile views at arbitrary row offsets within larger tiles.

### 3. Proper Vec→MTE3 Synchronization Before Output DMA

**Problem**: After Vec writes to UB via `TMOV`/`TADD`, issuing
`copy_ub_to_gm` (MTE3) to read from the same UB requires that Vec
writes are committed and visible to MTE3.

**Incorrect approach**: `pipe_barrier(PIPE_V)` only synchronizes the
Vec pipe internally. It does **not** establish a happens-before
relationship with MTE3.

**Correct approaches** (from lightweight to heavy):
1. `set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)` +
   `wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)` — places a flag on the
   Vec pipe that fires after all pending Vec ops complete; MTE3 waits
   for this flag before starting the DMA. This is the standard pattern
   used throughout the codebase.
2. `pipe_barrier(PIPE_ALL)` — waits for all pipes. Works but
   unnecessarily stalls MTE2 and other pipes.

**Impact**: Without proper Vec→MTE3 sync, cumsum produced completely
wrong results (max abs diff = 125). Adding the correct sync fixed it.

**Rule**: Before `copy_ub_to_gm` that reads Vec-written UB data, use
`set_flag(PIPE_V, PIPE_MTE3)` / `wait_flag(PIPE_V, PIPE_MTE3)`.
Reserve `pipe_barrier(PIPE_ALL)` for cases that genuinely need
all-pipe synchronization (e.g., before cross-core flag signals).

### 4. DMA-Cube Overlap Hides Load Latency

**Problem**: In kernels with Cube-Vec pipelines (e.g., `scaled_dot_kkt`),
the Vec core waits for the Cube to finish (`wait_flag_dev(slot)`) before
loading auxiliary data (G, Beta) from GM. This leaves the MTE2 pipe
idle during the Cube's GEMM.

**Solution**: Move DMA loads for data that doesn't depend on the Cube
output (G, Beta addresses depend only on chunk index, not Cube result)
to **before** `wait_flag_dev(slot)`. The DMA executes on MTE2 in
parallel with the Cube GEMM. After `wait_flag_dev` returns,
`pipe_barrier(PIPE_ALL)` ensures the DMA is complete.

**Implementation in scaled_dot_kkt**:
```cpp
// Before: DMA after Cube wait
wait_flag_dev(slot);
pipe_barrier(PIPE_ALL);
copy_gm_to_ub G;   // MTE2 idle during Cube work
copy_gm_to_ub Beta; // MTE2 idle during Cube work

// After: DMA before Cube wait (overlaps with Cube GEMM)
copy_gm_to_ub G;   // MTE2 runs in parallel with Cube
copy_gm_to_ub Beta; // MTE2 runs in parallel with Cube
wait_flag_dev(slot);
pipe_barrier(PIPE_ALL); // ensures both DMA and Cube are done
```

**Impact**: ~0.5-1 ms improvement for `scaled_dot_kkt` (4.22 ms → ~3.4-4.7 ms,
variance-dependent).

**Prerequisite**: The DMA source addresses must not depend on the Cube
output. Verify this by checking that address computations use only loop
indices and precomputed offsets.

### 5. BSND Strided DMA Is 2-4x Slower Than Contiguous

**Problem**: Loading QKV tiles from BSND layout requires row stride =
`H * D = 2048` half-elements (4096 bytes) between rows, but each row is
only `D = 128` half-elements (256 bytes). The MTE2 engine issues one
burst per row, so 128 rows = 128 separate 256-byte bursts at 4096-byte
intervals.

**Comparison**: With BHSD layout (static baseline), the same data is
contiguous — one 32 KB burst DMA.

**Measured impact**: Static baseline total = 39.6 ms vs initial dynamic
BSND total = 74.7 ms. Roughly half the gap came from strided DMA and
scalar extraction overhead.

### 6. Cube-Vec Pipeline Balance Is Critical

**Problem**: If the Vec core takes much longer than the Cube core per
chunk iteration, the Cube sits idle waiting for the Vec cross-core signal.

**Example**: In `scaled_dot_kkt`, the Cube does a single GEMM (K^T@K)
per chunk, but the Vec must do: DMA load G/Beta → compute gating → DMA
load KTK → SIMD gating → DMA store. After optimization, Vec work is
still longer than Cube work but the gap is much smaller.

**Good example**: `chunk_h` achieves better balance because its two GEMMs
(W@S, K^T@V) are large enough to dominate, making the Vec work a smaller
fraction. This is why chunk_h is 3.2x faster than Triton.

### 7. `pipe_barrier(PIPE_ALL)` vs `pipe_barrier(PIPE_V)`

**Problem**: `pipe_barrier(PIPE_ALL)` stalls **all** pipes until
completion. Use `pipe_barrier(PIPE_V)` when only Vec synchronization is
needed (most cases between consecutive SIMD operations).

**When to use `PIPE_ALL`**:
- Before `copy_ub_to_gm` when UB was written by Vec (lesson 3)
- When synchronizing multiple pipes (e.g., Vec + MTE2 + MTE3)

**When to use `PIPE_V`**:
- Between consecutive Vec operations (`TADD` → `TMUL` → `TEXP`)
- After `TMOV`/`TCVT` when the next operation is also Vec

**Impact**: Replacing 4 `pipe_barrier(PIPE_ALL)` with `PIPE_V` in
`wy_fast` saved ~0.5 ms.

### 8. TTRANS Has Significant Per-Call Overhead

**Attempted optimization**: Replace scalar GetValue/SetValue loops with
`pto::TTRANS` on `[H, H]` sub-blocks to transpose data in UB.

**Result**: 8 TTRANS + 8 TMOV operations (with `pipe_barrier(PIPE_V)`
between each) cost roughly the same as 128 scalar operations. Each
TTRANS + barrier costs ~0.6 μs, so 8 iterations = ~5 μs per chunk.

**Lesson**: TTRANS is useful for large square matrices, but for small
tiles (16×16) the per-operation overhead dominates. The `pipe_barrier`
after each TTRANS is the real cost.

### 9. TROWEXPAND + TMUL Replaces Scalar Coefficient Broadcasting

**Pattern**: To multiply each row of a `[R, C]` tile by a per-row scalar
coefficient, the naive approach uses `GetValue` + `TMULS` per row. The
vectorized approach:

1. Reinterpret the `[1, R]` ND coefficient tile as `[R, 1]` DN at the
   same UB address (both are R contiguous floats)
2. `TROWEXPAND(expanded_2d, coeff_dn)` broadcasts to `[R, C]`
3. `TMUL(tile, tile, expanded_2d)` applies all coefficients at once

**Constraint**: TROWEXPAND output (`[R, C]` floats) needs `R * C * 4`
bytes of UB scratch. For large tiles (e.g., `[64, 128]` = 32 KB), this
may not fit. Split into blocks (e.g., 4 iterations of `[16, 128]` = 8 KB
each).

**Impact**: Replaces `R` V→S stalls with `ceil(R/block)` TROWEXPAND+TMUL
iterations. Marginal gain when the scalar loop is already well-unrolled.

### 10. Sub-Block Parallelism Requires Careful Synchronization

**Attempted**: Use both Vec sub-blocks (vid=0, vid=1) in `chunk_cumsum`
to parallelize across heads.

**Problem**: Both sub-blocks sharing the same UB input address causes
race conditions — one sub-block's DMA can overwrite data while the other
is reading. Cross-sub-block synchronization is limited: `pipe_barrier`
only waits for THIS sub-block's operations, and event flags can have
ordering issues when both sub-blocks issue to shared pipes (MTE2).

**Lesson**: Sub-block parallelism works well when each sub-block has
**independent UB buffers** and **independent output regions** (as in
`scaled_dot_kkt` and `chunk_o` where vid splits rows). It fails when
sub-blocks need to share input data or synchronize on a shared output.

For the cumsum case, the SIMD row-wise approach (processing all heads
per row with single sub-block) was 5.5x faster than scalar—far better
than the 2x theoretical gain from dual sub-blocks.

### 11. DMA Double-Buffering Hides Latency

**Pattern from linear_attention**: Pre-load chunk i+1's data while
computing chunk i, using ping-pong buffers.

**Application**: `chunk_h` pre-fetches K and G for the next chunk at
the end of each iteration. `scaled_dot_kkt` uses workspace
double-buffering (slot = ci & 1). `wy_fast` naturally overlaps MTE2
loads with MTE3 stores across iterations since they use independent
pipes.

### 12. UB Address Aliasing Enables Tight Memory Packing

**Pattern**: Reuse UB regions that are dead at different phases:
```cpp
constexpr int32_t KV_UB = U_UB_HALF;  // KV reuses U's space after U is consumed
constexpr int32_t EXPAND_UB = 0;       // Expansion scratch reuses freed G_BLOCK region
```

**Rule**: Only alias buffers whose live ranges don't overlap. Document
the aliasing with comments. Verify with the UB allocation map.

### 13. Numerical Stability Has Performance Cost

**Example**: `scaled_dot_kkt` adds `min(0, g_row - g_col)` clamping
before `exp()` to prevent `Inf * 0 = NaN`.

**Better alternative**: `TMINS(coeff, coeff, 0.0f)` replaces the
original 4-instruction sequence (`TSUB` → `TSUB(negate)` → `TRELU` →
`TSUB(negate)`) with a single instruction. Always prefer `TMINS`/`TMAXS`
over multi-instruction clamp sequences.

## Performance Reference Points

| Configuration | Total Latency | Speedup vs Triton |
|:--|--:|--:|
| Triton baseline (BT=64, bf16) | 68.3 ms | 1.00x |
| Static BHSD PTO (C=128, fp16) | 39.6 ms | 1.73x |
| **Dynamic BSND PTO (C=128, fp16)** | **32.2 ms** | **2.12x** |

Per-kernel comparison:

| Kernel | Dynamic PTO (ms) | Triton (ms) | Speedup |
|:--|--:|--:|--:|
| chunk_cumsum | 0.37 | 1.00 | **2.7x** |
| scaled_dot_kkt | 4.69 | 4.81 | **1.03x** |
| wy_fast | 6.85 | 15.57 | **2.27x** |
| chunk_h | 9.57 | 30.82 | **3.22x** |
| chunk_o | 10.73 | 16.13 | **1.50x** |

All 5 PTO kernels now beat Triton. Dynamic BSND PTO is also faster than
the static BHSD PTO baseline (32.2 ms vs 39.6 ms) despite supporting
variable-length sequences.

## API Compatibility Constraint

PTO kernels must be **drop-in replacements** for Triton kernels:
- Accept `[B, S, H, D]` (BSND) layout tensors
- Accept `cu_seqlens` (int32) for variable-length sequences
- Same Python function signatures in `dynamic_kernel_libs.py`
- G/Beta transposition (`[1, T, H]` → `[H, T]`) happens inside the
  Python `run_*` wrappers, invisible to callers

Any additional layout optimization must happen **inside** the C++ kernel
or within the Python wrapper's `run_*` functions, not in the caller.
