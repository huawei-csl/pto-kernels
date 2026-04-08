# Porting Guide: Static BNSD -> Dynamic BSND Varlen

This note summarizes the lessons learned while porting the original static GatedDeltaNet PTO kernels into the `dynamic_bsnd` directory.

The goal of the port is not only to accept runtime `batch` and `seq_len`, but also to:

- accept native BSND tensors (`[batch, seq, head, hidden]`) without a Torch-side transpose
- support packed varlen execution through `cu_seqlens`
- keep the main math in PTO cube/vector code instead of shifting work back to the host

## Current outcome

- `chunk_cumsum` is native dynamic BSND PTO code.
- `scaled_dot_kkt` is a fused cube+vector PTO kernel and passes fixed plus packed-varlen checks.
- `chunk_o` is a fused cube+vector PTO kernel and passes fixed plus packed-varlen checks.
- `wy_fast` and `chunk_h` still pass correctness today, but still rely on host-side fallback/orchestration for part of the algorithm.

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

This was crucial for the `scaled_dot_kkt` fusion effort and remains the key issue in the unfinished native `wy_fast` port.

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

- the fused kernel structure exists and mostly mirrors the static version
- the remaining native bug is in the dynamic BSND vector-side coefficient build for `A1/A2`, especially around half-chunk boundaries and row-wise scaling semantics
- the current correctness path still uses exact Torch helpers for packed `A1/A2`
- the latest native debugging narrowed the failure further:
  - row-wise `beta` scaling for `A2` is much closer to correct than the older column-scaling attempt
  - the most suspicious remaining issue is now the native `g` load plus `TEXP` path for `A1`
  - identity probes (`beta = 1`, `g = 0`) showed that the native `A1` path can still corrupt leading rows of a half-chunk even when `A2` is otherwise correct
  - additional scratch-row `TEXP` experiments did not eliminate that leading-row corruption, so the bug is likely deeper than a simple scalar-exp patch
  - future work should debug native `g` extraction and exponentiation first, before changing the cube matmul path again

### `chunk_h`

- the cube matmuls are straightforward
- the hard part is the recurrence: state carry, `new_v`, `K^T @ new_v`, and final-state updates must all be made native while preserving varlen correctness
- this stage likely needs a more deliberate kernel design rather than only translating the existing host loop line by line

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
  - removing remaining host fallback/orchestration

## Practical advice for future work

- Treat `scaled_dot_kkt` and `chunk_o` as the best current native references in this directory.
- Treat `linear_attention.cpp` as the best cross-core fusion reference.
- Keep new experiments local to one stage at a time.
- Do not discard the host-backed path for `wy_fast` or `chunk_h` until the native replacement fully passes both fixed and packed-varlen checks.
