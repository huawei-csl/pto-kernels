# Porting Guide: Static BNSD -> Dynamic BSND Varlen

This note summarizes the lessons learned while porting the original static GatedDeltaNet PTO kernels into the `dynamic_bsnd` directory.

The goal of the port is not only to accept runtime `batch` and `seq_len`, but also to:

- accept native BSND tensors (`[batch, seq, head, hidden]`) without a Torch-side transpose
- support packed varlen execution through `cu_seqlens`
- keep the main math in PTO cube/vector code instead of shifting work back to the host

## Current outcome

- `chunk_cumsum` is native dynamic BSND PTO code.
- `scaled_dot_kkt` is a fused cube+vector PTO kernel and passes fixed plus packed-varlen checks.
- `wy_fast` is a fused cube+vector PTO kernel and passes fixed plus packed-varlen checks.
- `chunk_h` is a fused cube+vector PTO kernel with cross-core synchronized recurrence and passes fixed plus packed-varlen checks.
- `chunk_o` is a fused cube+vector PTO kernel and passes fixed plus packed-varlen checks.

All five stages are now fully native PTO kernels with no Torch fallback or host-side orchestration.

## Porting principles that worked

### 1. Keep the static math, change the indexing and launch contract

The working static kernels are the best reference for the math and synchronization pattern. Most dynamic-BSND work should be:

- change tensor addressing from static contiguous BNSD to dynamic strided BSND
- replace compile-time `L` assumptions with runtime `fixed_seq_len` and `cu_seqlens`
- add dynamic tail handling for short chunks

Avoid rewriting the math unless the layout change truly requires it.

### 2. Introduce shared sequence metadata helpers early

The most useful early step was centralizing sequence/chunk metadata in:

- `gdn_seq_info.h`
- `gdn_pto_shared.h`

These helpers let each kernel answer the same questions consistently:

- where a sequence begins in packed storage
- how many valid tokens are in the current chunk
- what global BSND stride to use
- what packed chunk index corresponds to a `(sequence, chunk, head)` tuple

Without this layer, every kernel ends up re-solving packed-varlen indexing differently and bugs multiply quickly.

### 3. Separate "logical shape" from "physical storage"

Dynamic BSND ports repeatedly hit bugs where the logical valid rows differed from the tile's physical size.

Be explicit about:

- `valid_rows` for the whole chunk
- `local_rows` for one vector half-chunk
- the physical tile size still being `ChunkSize` or `HalfChunk`

This matters for:

- GM load/store shapes
- zero padding rules
- final stores for varlen tail chunks
- synchronization participation for empty subblocks

### 4. Use dynamic global tensors for varlen tail stores

One recurring correctness issue was writing padded rows back to GM for short chunks.

The fix pattern was:

- use a dynamic-shape GM tensor for the final store
- set its row count to the actual `valid_rows` or `local_rows`

Do not rely on a fixed `ChunkSize` store when the last chunk is short.

### 5. Mirror working cube/vector fusion patterns exactly

For fused kernels, the most reliable references were:

- `linear_attention.cpp`
- static `chunk_o`
- static `scaled_dot_kkt`

The successful pattern is:

- cube computes the heavy matmul into a workspace or direct output tile
- vector waits on a cross-core flag before consuming cube results
- vector performs coefficient, gating, masking, or add/store epilogue
- vector signals cube when the next stage can proceed

In practice, the reliable building blocks were:

- `GdnWaitCrossFlag(...)`
- `GdnSetCrossFlag<...>(...)`
- `GdnSetFlag<Src, Dst>(...)`
- `GdnWaitFlag<Src, Dst>(...)`

Cross-core sync alone is not enough. In-kernel pipe ordering often also needs explicit pipeline flags around:

- GM -> UB loads before vector math
- vector convert/transform before GM stores
- UB -> GM stores before another core reads the result

### 6. Empty tail participants must still join the handshake

Packed-varlen deadlocks appeared when a vector subblock had `local_rows == 0` and simply skipped work.

For fused cube/vector kernels, even empty tail participants often still need to:

- wait on the same cross-flag
- set the next cross-flag

Otherwise one side advances and the other side stalls forever.

### 7. UB layout bugs are easy to mistake for math bugs

Several "numerical" failures were really UB overlap or aliasing problems.

Common symptoms:

- `inf` or `nan` appearing only on some rows
- correct values at the beginning of a tile and garbage near the end
- row tails or half-chunk boundaries failing while the rest looks fine

When debugging:

- write down every UB region and its exact byte size
- check alignment boundaries
- check whether padded tile widths differ from logical widths
- verify whether a later scratch allocation overlaps a prior temporary

For dynamic kernels, this mattered especially for:

- `beta` scratch tiles
- coefficient workspaces
- tail row broadcast temporaries

### 8. Packed beta and g extraction are subtle in BSND

For BSND varlen kernels, `beta` and `g` handling is easy to get wrong because the mathematical role can be row-wise or column-wise depending on the stage.

Lessons:

- verify whether the coefficient should be attached to source rows, destination rows, or columns in the packed matrix
- do not assume the extraction pattern from one stage transfers unchanged to another
- when a tile API behaves unexpectedly, reduce the load path to the simplest possible contiguous block and rebuild the intended vector in UB manually

This was crucial for the `scaled_dot_kkt` fusion effort and was also important for the `wy_fast` native port.

### 9. Probe kernels are worth it for hard vector bugs

When a fused kernel is failing and the failing stage is unclear, a tiny debug kernel is often faster than guessing.

Useful probe categories:

- load/store a suspicious GM slice into UB and back out
- isolate beta extraction
- isolate g extraction
- isolate coefficient construction
- isolate workspace copy paths

The `dynamic_bsnd/debug/` directory was created for exactly this reason during `scaled_dot_kkt` debugging.

### 10. Validate stage-by-stage before chaining

The staged approach was the right one.

Recommended order:

1. port one stage
2. get fixed-length correctness
3. get packed-varlen correctness
4. fuse cube/vector if applicable
5. benchmark that stage
6. move to the next stage

Trying to debug the full GDN chain before each stage is stable makes failures much harder to localize.

### 11. Prefer tensor operations over scalar loops for row-wise scaling

The `wy_fast` port hit a persistent bug where scalar `TMULS` loops corrupted the last two rows of each half-chunk (rows 62, 63 and 126, 127). The root cause was pipeline synchronization between the scalar pipe (`GetValue`) and the vector pipe (`TMULS`). Explicit `set_flag(PIPE_V, PIPE_S)` / `wait_flag` partially helped but did not fully resolve the issue across both sub-blocks.

The fix was to replace the scalar loop entirely with `TROWEXPANDMUL`, which performs row-wise scaling as a single tensor operation without any scalar-vector pipe interaction. This pattern should be preferred wherever a 2D tile needs per-row scaling by a 1D coefficient vector.

The `TROWEXPANDMUL` approach requires:

- a `[Rows, Cols]` RowMajor source tile
- a `[Rows, 1]` ColMajor coefficient tile (aliased at the same UB address as a `[1, Rows]` RowMajor tile)

### 12. Cross-core flag management across work items requires care

For kernels that process multiple work items per block (e.g., `chunk_h` iterating over `(seq, head)` pairs), cross-core flags can leak between work items if not managed carefully.

The safe pattern is:

- only signal a flag when the other side is guaranteed to wait for it
- do not signal the final handshake flag after the last iteration of an inner loop
- let the initialization phase of the next work item provide the first signal

In `chunk_h`, flag 3 (vector-to-cube state ready) is signaled before the chunk loop starts and after each non-final chunk, but NOT after the final chunk. This ensures the cube sees exactly `chunk_num` flag-3 signals per work item.

## Kernel-specific lessons

### `chunk_cumsum`

- good first target because it is mostly vector logic
- useful for validating packed-varlen BSND indexing helpers

### `scaled_dot_kkt`

- the static kernel's math and sync pattern transferred well once the dynamic indexing was correct
- key bugs were beta extraction, UB overlap, and tail stores
- the successful end state is one fused cube+vector kernel

### `chunk_o`

- this stage maps naturally onto the `linear_attention.cpp` fused design
- the biggest dynamic-only issues were tail handling and explicit pipeline ordering around vector epilogues
- the current fused result is a good reference for future fusion work

### `wy_fast`

- the fused kernel mirrors the static version's math and sync pattern
- the key breakthrough was replacing scalar `TMULS` loops for row-wise coefficient scaling with `TROWEXPANDMUL`, which avoids pipeline stall issues that corrupted half-chunk boundary rows
- the `A1 = A * (exp(g) * beta)` and `A2 = A * beta` coefficient builds are fully kernel-side
- earlier debugging showed that the scalar `TMULS` loop had systematic corruption at rows 62, 63, 126, 127 (last two rows of each half-chunk), caused by pipeline synchronization issues between the scalar and vector pipes
- `TROWEXPANDMUL` performs the entire row-wise scaling in a single tensor operation, eliminating the pipeline sync problem
- `TEXP` on the full-chunk `g_ub` buffer works correctly when the packed `g` tensor is pre-padded with zeros
- the successful end state is one fused cube+vector kernel with no Torch fallback

### `chunk_h`

- the fused kernel uses a 4-point cross-core handshake per chunk iteration (flags 0, 1, 2, 3)
- cube computes `ws = W @ state` (flag 0) and `kv = k_scaled^T @ new_v` (flag 2)
- vector computes coefficients, `k_scaled`, `new_v` (flag 1) and updates `state = state * exp(g_last) + kv` (flag 3)
- each block processes one `(sequence, head)` work item and iterates sequentially over its chunks
- state is carried between chunks via a per-block half-precision GM workspace
- the vector side handles both sub-blocks' state portions (64 rows each of the 128x128 state matrix) even when `local_rows == 0` for K/U/new_v
- cross-core flag 3 is only signaled when there is a subsequent chunk to process, preventing stale flags across work items
- dynamic L1 tiles with `PadValue::Zero` handle partial chunks: the cube loads only `valid_rows` from k_scaled and new_v workspaces
- K is loaded from BSND layout with dynamic zero-padded UB tiles; new_v is stored to `nv_out` with dynamic stores to preserve zero-padding for invalid rows
- the successful end state is one fused cube+vector kernel with no host-side recurrence loop

## Recommended debugging workflow

1. Start from the static kernel or another known-good fused reference.
2. Port indexing and GM tensor shapes first.
3. Keep math identical until the first correctness failure.
4. If failure is localized, compare intermediate packed tensors against Torch reference.
5. If failure is not localized, write a minimal debug kernel.
6. Once correctness is stable, benchmark on a small case and on at least one large underfill-resistant case.

## Performance lessons

- Small-shape timings can be misleading because launch overhead and underfill dominate.
- A kernel can be "correct and fused" while still being far slower than the static reference.
- The main performance gap is not only launch count; it also comes from dynamic indexing overhead, extra vector work, and conservative workspace usage.
- After correctness is stable, the next optimization pass should focus on:
  - reducing extra GM traffic
  - shrinking temporary workspace
  - improving vector-side coefficient generation
  - tuning synchronization granularity

## Practical advice for future work

- Treat `scaled_dot_kkt`, `wy_fast`, `chunk_h`, and `chunk_o` as working fused cube+vector references in this directory.
- Treat `linear_attention.cpp` as the best cross-core fusion reference.
- Keep new experiments local to one stage at a time.
- All five stages are now fully native. Future work should focus on performance optimization and large-shape benchmarking.
