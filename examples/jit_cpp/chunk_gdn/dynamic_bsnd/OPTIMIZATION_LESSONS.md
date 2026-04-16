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
the total kernel time for `scaled_dot_kkt` (15.5 ms) and `chunk_o`
(26.2 ms).

**Root cause in dynamic BSND**: The BSND layout `[B, S, H, D]` stores
heads interleaved. To extract per-head G values from `[C, H]` blocks,
we must gather every H-th element—requiring scalar loops since PTO-ISA
does not support:
- Cross-layout DMA (`TLOAD` only supports ND→ND, DN→DN, NZ→NZ)
- Strided single-element DMA (minimum row width = 32 bytes)
- Scatter/gather vector instructions

**Mitigation strategies** (in order of effectiveness):
1. **Ensure data arrives in per-head-contiguous layout** — eliminates
   scalar loops entirely (the static BHSD baseline does this)
2. **Minimize the number of scalar accesses** — batch multiple heads
   per load, or reduce chunk size
3. **Overlap scalar work with DMA/Cube** — pre-fetch next chunk's data
   while current chunk's scalar extraction runs

### 2. BSND Strided DMA Is 2-4x Slower Than Contiguous

**Problem**: Loading QKV tiles from BSND layout requires row stride =
`H * D = 2048` half-elements (4096 bytes) between rows, but each row is
only `D = 128` half-elements (256 bytes). The MTE2 engine issues one
burst per row, so 128 rows = 128 separate 256-byte bursts at 4096-byte
intervals.

**Comparison**: With BHSD layout (static baseline), the same data is
contiguous — one 32 KB burst DMA.

**Measured impact**: Static baseline total = 39.6 ms vs dynamic BSND
total = 74.7 ms. Roughly half the gap comes from strided DMA overhead.

### 3. Cube-Vec Pipeline Balance Is Critical

**Problem**: If the Vec core takes much longer than the Cube core per
chunk iteration, the Cube sits idle waiting for the Vec cross-core signal.

**Example**: In `scaled_dot_kkt`, the Cube does a single GEMM (K^T@K)
per chunk (~2 ms total), but the Vec must do: DMA load G/Beta → scalar
extract → 10+ SIMD ops → DMA load KTK → SIMD gating → DMA store. This
Vec work is ~3x longer than the Cube work.

**Good example**: `chunk_h` achieves better balance because its two GEMMs
(W@S, K^T@V) are large enough to dominate, making the Vec scalar
extraction a smaller fraction.

### 4. `pipe_barrier(PIPE_ALL)` Is Expensive

**Problem**: `pipe_barrier(PIPE_ALL)` stalls **all** pipes until
completion. Use `pipe_barrier(PIPE_V)` when only Vec synchronization is
needed (most cases after SIMD operations).

**Example**: `wy_fast_kernel.cpp` uses 4 `pipe_barrier(PIPE_ALL)` calls
per work item. The static baseline uses only `pipe_barrier(PIPE_V)`.

### 5. TTRANS Has Significant Per-Call Overhead

**Attempted optimization**: Replace scalar GetValue/SetValue loops with
`pto::TTRANS` on `[H, H]` sub-blocks to transpose data in UB.

**Result**: 8 TTRANS + 8 TMOV operations (with `pipe_barrier(PIPE_V)`
between each) cost roughly the same as 128 scalar operations. Each
TTRANS + barrier costs ~0.6 μs, so 8 iterations = ~5 μs per chunk.

**Lesson**: TTRANS is useful for large square matrices, but for small
tiles (16×16) the per-operation overhead dominates. The `pipe_barrier`
after each TTRANS is the real cost.

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

## Performance Reference Points

| Configuration | Total Latency | Total TFLOPS |
|:--|--:|--:|
| Triton baseline (BT=64, bf16) | 68.6 ms | 10.5 |
| **Dynamic BSND PTO (C=128, fp16)** | **74.7 ms** | **11.0** |
| Static BHSD PTO (C=128, fp16) | 39.6 ms | 20.8 |
| Linear attention PTO (peak) | — | 77.3 |

Per-kernel comparison (dynamic PTO vs Triton vs static PTO):

| Kernel | Dynamic PTO (ms) | Triton (ms) | Static PTO (ms) |
|:--|--:|--:|--:|
| chunk_cumsum | 2.03 | 1.04 | 1.37 |
| scaled_dot_kkt | 15.52 | 4.93 | 8.76 |
| wy_fast | 16.78 | 15.62 | 9.52 |
| chunk_h | 14.18 | 30.83 | 8.31 |
| chunk_o | 26.20 | 16.16 | 11.60 |

Kernels where PTO already beats Triton: **chunk_h** (2.2x faster),
**wy_fast** (comparable). Kernels where PTO lags: **scaled_dot_kkt**
(3.1x slower), **chunk_o** (1.6x slower), **chunk_cumsum** (2x slower).

## API Compatibility Constraint

PTO kernels must be **drop-in replacements** for Triton kernels:
- Accept `[B, S, H, D]` (BSND) layout tensors
- Accept `cu_seqlens` (int32) for variable-length sequences
- Same Python function signatures in `dynamic_kernel_libs.py`
- No Python-side transposes or layout conversions

Any layout optimization must happen **inside** the C++ kernel, not in
the Python wrapper.
