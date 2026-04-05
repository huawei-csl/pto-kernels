# Linear Attention Optimization Lessons

This note records the optimization practices learned while improving the standalone PTO-ISA `linear_attention` kernel, and lists the next promising directions inspired by higher-performance kernels such as `flash_atten`-style examples and `mla_prefill_cce`.

The file name intentionally matches the requested spelling: `optimization_lession.md`.

## Current Status

- The current kernel is a minimal direct-`pto::` implementation in `linear_attention.cpp`.
- It keeps dynamic `B` and `L`, while `H`, `D`, and `C` stay compile-time parameters.
- It passes the full correctness sweep in `run_linear_attention.py`.
- The current measured default-table performance is about `29-31 TFLOP/s`, with best measured throughput around `30.53 TFLOP/s`.

## Practices That Already Helped

### 1. Remove unnecessary abstraction on the hot path

- Rewriting the kernel to use direct `pto::` APIs instead of many wrapper helpers improved both readability and performance.
- The main hot-path operations are now easy to see: `TLOAD`, `TEXTRACT`, `TRESHAPE`, `TMATMUL`, `TMATMUL_ACC`, `TSTORE`, `TMUL`, and `TADD`.
- Keeping only the used full-tile path helped the compiler generate better code and reduced control-flow noise.

### 2. Keep the kernel compile-time specialized where it matters

- Fixing `H`, `D`, and `C` as compile-time parameters is important.
- This removes dynamic branches from the inner loops.
- It also lets the compiler better optimize tile shapes, on-chip allocation, and instruction scheduling.

### 3. Use a fixed launch shape and loop over dynamic work in-kernel

- Using `block_dim = num_cores` and mapping dynamic work items with:
  `pid = work_idx * block_num + cid`
  is the right persistent-kernel structure.
- This avoids host-side launch variability and keeps work distribution simple.

### 4. Respect on-chip memory limits aggressively

- Explicit byte-level budgeting for L1, L0C, and UB is critical.
- A compile-time `static_assert` on the L0C footprint catches invalid tile choices early.
- This is especially important when trying larger `C` or `D`.

### 5. Avoid scalar loops on the vector path

- The scalar causal-mask loop on `acc_ub` was a major bottleneck.
- Precomputing the triangular mask in PyTorch once, passing it as an extra tensor, and applying it with UB `TMUL` gave a large speedup.
- This follows the same general advice from flash-attention docs: replace scalar loops with tile/vector operations.

### 6. Match the compiler flags used by stronger PTO builds

- Using the stronger Bisheng/AICore flags mattered a lot:
  `-cce-aicore-addr-transform`
  stack sizing flags
  overflow-record flags
  `-cce-aicore-dcci-insert-for-scalar=false`
  `-DL2_CACHE_HINT`
- These recovered a meaningful amount of performance even before deeper kernel changes.

### 7. Separate cube and vector responsibilities clearly

- The kernel follows a clean cube/vector split:
  cube side computes `QK^T`, `K^T V`, `acc @ V`, and `Q @ H`
  vector side applies the causal mask and accumulates the running `H`
- This maps well onto the underlying machine model.

### 8. Keep correctness handshakes explicit

- Cross-core synchronization around workspace handoff is essential.
- The vector side must initialize or update workspace state before cube consumes it.
- Reducing barriers is tempting, but correctness breaks easily if producer/consumer edges are not exact.

## Main Lessons From High-Performance Examples

From `flash_attention` optimization docs and `mla_prefill_cce`, the repeated themes are:

- use larger-granularity compute and data movement
- keep data resident in L1/L0 when reuse is high
- overlap MTE, cube, vector, and FIX as much as possible
- double buffer aggressively
- reduce scalar dispatch and synchronization frequency
- tune toward a single dominant bound instead of many small bubbles

The current kernel has improved vectorization, but it is still structurally much more serial than those high-performance kernels.

## Next Potential Optimizations To Try

Below is a broad list of experiments worth trying next.

## A. Cube-Side Data Reuse And Tiling

- Keep `Q` resident in L1 across more work instead of reloading it every time.
- Explore whether `V` can also stay in L1 longer for the `acc @ V` stage.
- Introduce explicit K-splitting inside the `ChunkSize x HiddenSize` matmuls instead of relying on a single full-tile extraction path.
- Try alternate valid `(C, D)` pairs that still satisfy the L0C budget but improve arithmetic intensity.
- Consider splitting `D` into smaller inner tiles if that enables better overlap and higher cube utilization.
- Explore 2D tile decomposition for the `H` update and output path rather than treating each chunk as one monolithic tile.

Why this might help:
- `mla_prefill_cce` heavily exploits tiled inner loops and explicit fragment-level movement.
- High-performance FA kernels usually increase reuse per GM byte rather than only optimizing single-tile execution.

Risks:
- More tiling increases indexing complexity.
- Accumulation order and workspace semantics can change subtly.

## B. L1 <-> L0 Double Buffering

- Add ping-pong buffering for L1-to-L0A and L1-to-L0B movement.
- While one tile is being consumed by cube, preload the next tile into the alternate buffer.
- Apply the same idea separately to the two cube phases:
  `QK^T`
  `K^T V`
  `acc @ V`
  `Q @ H`

Why this might help:
- Flash-attention docs explicitly call out L1->L0 double buffering as a key intra-core optimization.
- The current cube path still has visible serialization between load, compute, and store.

Risks:
- Easy to introduce hazards by reusing a buffer before the previous pipe finishes.

## C. Vector-Side Double Buffering

- Double buffer the vector path for:
  loading masked-acc tiles
  loading `H` tiles
  applying `TMUL`
  running `TADD`
  storing updated workspace
- Overlap `MTE2 -> VEC -> MTE3` between successive chunks.

Why this might help:
- Flash-attention tuning docs explicitly recommend vector-path double buffering.
- The vector side currently became much faster after mask vectorization, but it still executes in a mostly serial chunk loop.

Risks:
- More buffers increase UB pressure.
- Cross-core synchronization logic becomes more delicate.

## D. Inter-Core Pipeline / Multi-Stage Workspace

- Replace the single workspace slot per core with 2-stage or 3-stage workspace ping-pong.
- Let cube produce chunk `i + 1` while vector is still consuming chunk `i`.
- Tune a `num_stages` style pipeline similar to the inter-core pipeline ideas in flash-attention docs.
- Try separate workspace slots per stage for `workspace_1` and `workspace_2`.

Why this might help:
- Current cube/vector handoff is still mostly chunk-by-chunk serial.
- High-performance CV-fusion style kernels reduce bubbles by letting stages run ahead.

Risks:
- This is one of the easiest places to deadlock or consume stale data.
- Memory footprint grows with the number of stages.

## E. Reduce Synchronization Frequency

- Check whether multiple chunks can be processed before one cross-core sync, instead of syncing every chunk.
- Batch some vector-side updates before signaling back to cube.
- Replace broad `pipe_barrier(PIPE_ALL)` use with narrower event dependencies where safe.

Why this might help:
- Flash-attention docs explicitly call out excessive synchronization as a scalar-bound problem.
- `mla_prefill_cce` uses very fine-grained flags rather than global drain-like barriers everywhere.

Risks:
- Wrong dependency pruning can silently corrupt results.
- Some per-chunk dependencies are real because `H` is recurrent.

## F. Fuse Or Simplify Vector Operations Further

- Check whether mask application and running-sum update can be fused more tightly.
- Try using fewer vector instructions per chunk.
- Explore whether the mask can be stored in a more compact or more hardware-friendly layout.
- Consider generating the triangular mask in-kernel with vector primitives if that removes GM reads without reintroducing scalar overhead.
- Try a permanent UB-resident mask if the same tile can be reused safely across all work on a core.

Why this might help:
- The current precomputed-mask change already proved scalar dispatch was very expensive.
- Further reduction in vector instruction count may still be available.

Risks:
- Fused math may change rounding behavior.

## G. Improve Workspace Layout And Memory Format

- Revisit `workspace_1` and `workspace_2` layout for better GM burst efficiency.
- Check whether ND is the best format for all transfers or whether some paths want NZ-like organization.
- Add explicit alignment or padding between workspace regions if that reduces bank conflicts or improves bursts.
- Explore storing only the needed half-tiles per vector sub-block instead of full dense tiles in some stages.

Why this might help:
- `mla_prefill_cce` uses more specialized GM->L1 and L0C->GM movement patterns.
- The current kernel is still using a very straightforward layout.

Risks:
- Layout changes are correctness-sensitive and easy to mismatch with host tensor assumptions.

## H. Make The Cube Path More Like A Real GEMM Pipeline

- Introduce a dedicated GEMM micro-kernel structure closer to optimized `matmul` examples.
- Explicitly separate:
  TLOAD panels
  TEXTRACT fragments
  TMATMUL / TMATMUL_ACC
  TSTORE
- Let each stage overlap with the next through ping-pong and event ordering.
- Treat the cube path as a proper steady-state pipeline instead of repeating a simple serial sequence.

Why this might help:
- High-performance examples do not leave cube work as isolated, barrier-heavy single-shot matmuls.
- A more pipeline-friendly cube path is likely necessary to move toward ~100 TFLOP/s.

Risks:
- Code size and complexity increase significantly.

## I. Tune Toward A Single Dominant Bound

- Use profiling to determine whether the kernel is now:
  vector bound
  cube bound
  MTE bound
  FIX bound
  sync/scalar bound
- Then optimize so the longest useful stage hides the others.
- Prefer one clean dominant pipeline over many partially idle stages.

Why this might help:
- The flash-attention doc explicitly recommends optimizing toward a single bound.

Risks:
- Optimizing blindly without stage-level timing can waste time.

## J. Try Larger Benchmark Shapes That Improve Steady-State Utilization

- Benchmark more shapes with larger `B * H` and longer `L`.
- Try shape sets that improve persistent-kernel occupancy and reduce warm-up/drain distortion.
- Add a second benchmark preset specifically for throughput hunting.

Why this might help:
- Larger shapes often hide fixed overhead better.
- The current best result already improved at larger total work.

Risks:
- Bigger shapes do not fix structural inefficiency by themselves.

## K. Compiler / Codegen Experiments

- Sweep a small set of compiler flags around the currently good Bisheng configuration.
- Compare whether any flag changes help the vectorized-mask kernel differently from the scalar-mask version.
- Recheck newer PTO headers once the API mismatch is resolved, because codegen behavior may improve.

Why this might help:
- Compiler/codegen choices already showed real performance impact in this project.

Risks:
- Hard to generalize across toolchain versions.

## L. Borrow More Structure From `mla_prefill_cce`

Specific ideas to copy conceptually from `mla_prefill_cce`:

- ping-pong flags for multiple stages
- separate ping-pong control for KV-style loads
- launch delay / staggered pipeline structure
- multiple L0C buffers to overlap compute and store
- more deliberate event choreography between MTE1, MTE2, M, and FIX

Why this might help:
- `mla_prefill_cce` is much closer to the class of kernels that actually reach the ~100 TFLOP/s range.

Risks:
- It is much more complex and may require a partial redesign rather than a local patch.

## Suggested Optimization Order

If continuing from the current ~30 TFLOP/s kernel, a practical order is:

1. Add vector-side double buffering.
2. Add cube-side L1/L0 ping-pong.
3. Introduce 2-stage workspace pipeline between cube and vector.
4. Reduce synchronization frequency where dependencies allow.
5. Revisit `(C, D)` and K-splitting under the explicit pipeline.
6. Profile and tune toward one dominant bound.

## Experiment Tracking Plan

Use the following lightweight tracking format for each optimization attempt:

- `ID`: short experiment ID
- `Goal`: what performance bottleneck it is trying to reduce
- `Hypothesis`: why it should help
- `Change`: what code or benchmark setup to modify
- `Check`: how to validate correctness and performance
- `Status`: `todo`, `doing`, `done`, `dropped`
- `Result`: measured outcome and brief notes

Suggested workflow:

1. Pick one experiment only.
2. Record the exact benchmark shape set before changing code.
3. Run correctness first.
4. Run the same benchmark table before and after.
5. Keep or drop the change based on measured evidence.

### Backlog

#### `exp01` Vector Double Buffer

- `Goal`: overlap vector `MTE2 -> VEC -> MTE3` across chunks
- `Hypothesis`: the vector side still has serial bubbles after the mask-vectorization improvement
- `Change`: ping-pong `acc_ub`, `h_ub`, and masked output UB tiles
- `Check`: full `run_linear_attention.py` plus default benchmark table
- `Status`: `done (reverted)`
- `Result`: first implementation compiled and passed a smoke test but deadlocked during the full sweep; reverted to the known-good baseline and noted that future vector double-buffer attempts need a stricter event plan and careful cleanup of orphaned hung processes before re-measuring

#### `exp02` Cube L1/L0 Ping-Pong

- `Goal`: overlap cube-side load, extract, and matmul work
- `Hypothesis`: current cube stages still serialize `TLOAD`, `TEXTRACT`, `TMATMUL`, and `TSTORE`
- `Change`: introduce double buffering for L1/L0A/L0B tiles in the cube path
- `Check`: correctness sweep plus benchmark table, compare against current ~30 TFLOP/s baseline
- `Status`: `todo`
- `Result`: not started

#### `exp03` Two-Stage Workspace Pipeline

- `Goal`: let cube and vector work on adjacent chunks concurrently
- `Hypothesis`: chunk-by-chunk cube/vector handoff is causing avoidable inter-core bubbles
- `Change`: add 2 workspace slots per core and a stage-aware cross-core handshake
- `Check`: correctness sweep, deadlock check, benchmark table
- `Status`: `todo`
- `Result`: not started

#### `exp04` Reduced Sync Frequency

- `Goal`: lower scalar/synchronization overhead
- `Hypothesis`: syncing every chunk may be too expensive
- `Change`: batch some work before cross-core signaling where data dependencies permit
- `Check`: correctness sweep with long `L` and large `B*H`, benchmark table
- `Status`: `todo`
- `Result`: not started

#### `exp05` Narrower Pipe Dependencies

- `Goal`: reduce over-barriering
- `Hypothesis`: some `pipe_barrier(PIPE_ALL)` calls can be replaced by narrower event dependencies
- `Change`: replace selected full barriers with more precise waits/flags
- `Check`: correctness sweep and repeated benchmark runs to catch unstable behavior
- `Status`: `todo`
- `Result`: not started

#### `exp06` K-Split Cube Microkernel

- `Goal`: improve overlap and on-chip utilization inside large matmuls
- `Hypothesis`: explicit K-splitting can outperform the current single full-tile extract path
- `Change`: rewrite `MatmulL1` into a K-part microkernel with accumulation across parts
- `Check`: correctness sweep and benchmark table
- `Status`: `todo`
- `Result`: not started

#### `exp07` Workspace Layout / Padding

- `Goal`: improve memory-system efficiency for workspace traffic
- `Hypothesis`: padding or alternate workspace layout may improve burst behavior or reduce conflicts
- `Change`: try aligned/padded workspace layout and matching indexing changes
- `Check`: correctness sweep and benchmark table
- `Status`: `todo`
- `Result`: not started

#### `exp08` Alternate `(C, D)` Search

- `Goal`: find a better arithmetic-intensity point within L0C budget
- `Hypothesis`: some valid `(C, D)` combinations may use the machine better than the current one
- `Change`: benchmark additional compile-time shape families that still satisfy the current memory budget
- `Check`: compile success, correctness, benchmark table
- `Status`: `done`
- `Result`: reworked the minimum kernel to reuse one shared L0C accumulator region across the serialized cube stages and kept the vector mask path adaptive to the UB budget; this enabled `C=128, D=128` to compile and pass correctness, and the benchmark improved from the `~30 TFLOP/s` class at `C=64` to `47.79-53.07 TFLOP/s` on the default large-shape table, with the current best at `(32, 20, 1024, 128, 128)`

#### `exp09` Larger Throughput Shapes

- `Goal`: identify shapes that push utilization higher without changing the kernel
- `Hypothesis`: larger `B * H` and longer `L` may reduce fixed overhead impact further
- `Change`: add a second benchmark preset specifically for throughput hunting
- `Check`: benchmark-only experiment, keep same correctness-tested kernel
- `Status`: `todo`
- `Result`: not started

#### `exp10` Compiler Flag Sweep

- `Goal`: extract more performance from codegen without changing semantics
- `Hypothesis`: some Bisheng/AICore flags may interact differently with the new vectorized mask path
- `Change`: sweep a small set of compile flags around the current known-good configuration
- `Check`: correctness on at least one small and one large shape, benchmark table on one reference shape
- `Status`: `done`
- `Result`: on the reference shape `(16, 20, 2048, 128, 64)`, `baseline` measured `30.77 TFLOP/s`, dropping `L2_CACHE_HINT` measured `31.22 TFLOP/s`, and dropping both `L2_CACHE_HINT` plus `addr-transform` measured `31.70 TFLOP/s`; the default JIT flags were updated accordingly, the full correctness sweep still passed, and the default benchmark table improved to `31.17 TFLOP/s` / `1062.92 GiB/s`

#### `exp11` Mask Residency Experiment

- `Goal`: reduce even the small remaining cost of mask loading
- `Hypothesis`: keeping the mask resident or regenerating it cheaply in-core may beat GM mask load
- `Change`: test persistent UB-resident mask or vector-generated mask forms
- `Check`: correctness sweep and benchmark table
- `Status`: `todo`
- `Result`: not started

#### `exp12` MLA-Style Pipeline Borrowing

- `Goal`: move toward the structural style used by ~100 TFLOP/s kernels
- `Hypothesis`: MLA-style staged ping-pong plus delayed launch and multi-buffer orchestration is needed for the next major jump
- `Change`: selectively port one structural idea from `mla_prefill_cce`, starting with ping-pong L0C or staggered stage scheduling
- `Check`: correctness sweep, deadlock check, benchmark table
- `Status`: `todo`
- `Result`: not started

## Closing Thought

The big lesson so far is that the largest gains did not come from changing the math. They came from:

- reducing scalar work
- improving specialization
- simplifying the hot path
- matching good codegen flags
- using vector instructions for masking

To go from ~30 TFLOP/s to the ~100 TFLOP/s class, the next leap will likely require deeper pipeline overlap and buffering, not just another small local micro-optimization.
