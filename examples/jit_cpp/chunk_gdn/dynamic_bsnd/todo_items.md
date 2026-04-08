# Dynamic BSND GDN Todo Items

This file is a handoff note for the remaining work in `dynamic_bsnd`.

It summarizes:

- what currently passes
- what is still hybrid
- what the known performance gap is
- which next debugging and optimization actions are most promising

## What is passing today

As of the latest verification run, the stage-validation driver `run_gated_delta_dynamic_bsnd.py` passes all currently implemented stage checks:

- `chunk_cumsum`
- `scaled_dot_kkt`
- `wy_fast`
- `chunk_h`
- `chunk_o`

Verified commands:

```bash
export PTO_LIB_PATH=/sources/pto-isa
python run_gated_delta_dynamic_bsnd.py
```

Latest reported outputs:

- `chunk_cumsum`: fixed `0.074 ms`, packed-varlen `0.072 ms`
- `scaled_dot_kkt`: fixed `0.064 ms, 0.52 TFLOP/s`, packed-varlen `0.062 ms, 0.41 TFLOP/s`
- `wy_fast`: fixed `1.934 ms, 0.03 TFLOP/s`, packed-varlen `1.645 ms, 0.03 TFLOP/s`
- `chunk_h`: fixed `4.611 ms`, packed-varlen `3.620 ms`
- `chunk_o`: fixed `0.167 ms, 0.40 TFLOP/s`, packed-varlen `0.172 ms, 0.29 TFLOP/s`

## Remaining high-level problems

### 1. `wy_fast` is still hybrid

Current state:

- PTO cube kernels are used for the packed `A1 @ K` and `A2 @ V` matmuls.
- Torch/NPU helper code still builds the dynamic BSND packed `A1` and `A2` tensors for correctness.

Why this matters:

- this stage is not yet a fully native dynamic BSND PTO kernel
- the fallback keeps extra host-side logic in the execution path
- performance remains far below the static reference

### 2. `chunk_h` is still hybrid

Current state:

- PTO cube kernels are used for `W @ S` and `K^T @ new_v`
- the recurrent state update and chunk-by-chunk sequencing are still driven on the host

Why this matters:

- the recurrence is not yet a native dynamic BSND kernel
- host orchestration makes the stage much harder to optimize
- it prevents the chain from becoming a fully kernel-side GDN implementation

### 3. Dynamic kernels are still much slower than static references

Even the stages that are now native and fused still trail the original static kernels by a large margin.

Known examples:

- `scaled_dot_kkt` dynamic fused performance is still far below the static reference on large benchmark shapes
- `chunk_o` is correct and fused, but current throughput is still far below the expected static-baseline neighborhood
- `wy_fast` and `chunk_h` are particularly slow because they still retain host-side work

Why this matters:

- correctness is no longer the only blocker
- the project still needs a real optimization pass after the remaining hybrid stages are removed

## Kernel-specific leftover issues

### `wy_fast`

Status:

- correctness currently comes from the fallback path in `dynamic_kernel_libs.py`
- the native fused kernel attempt in `wy_fast_kernel.cpp` is not yet correct enough to replace it

Most useful findings from the latest native debugging:

- the fused structure itself is plausible and close to the static version
- the biggest remaining issue is in the vector-side dynamic BSND coefficient build for `A1` and `A2`
- the earlier native attempt showed half-chunk and tail-row corruption patterns
- `A2` was brought much closer to correct after fixing row-wise scaling semantics
- the remaining drift is concentrated in the `A1 = A * (exp(g) * beta)` side
- the bug appears near half-chunk boundaries and row/tail handling, not in the cube GEMM itself
- the most recent probe narrowed this further:
  - native `A2` can be made close to correct with local row-wise `beta` scaling
  - the most suspicious remaining native issue is the `g` vector load / `TEXP` path used to build `A1`
  - identity-style probes (`A=1`, `beta=1`, `g=0`) showed that `A1` can still corrupt leading rows of a half-chunk even when `A2` is much healthier
  - attempts to patch this with scalar exp or alternate contiguous `g` loads either failed to link or regressed the wider kernel, so the current committed path keeps the host-backed correctness wrapper
  - a scratch-row `TEXP` patch was also tried and still did not remove the leading-row corruption, so the unresolved bug is not yet reduced to a trivial scalar-exp replacement

Practical consequence:

- the best next work item is to continue debugging the native `wy_fast` vector-side coefficient construction, not the matmul stage

### `chunk_h`

Status:

- the stage passes today with host-side recurrence/orchestration
- no native in-kernel recurrence replacement exists yet

Main missing pieces:

- persistent chunk-to-chunk state propagation in-kernel
- native computation and storage of `new_v`
- native update of `state = state * exp(g_last) + kv`
- packed-varlen-safe final state writeback

Practical consequence:

- this stage likely needs a dedicated redesign instead of incremental tweaks to the current host loop

## Promising next-step action items

### For `wy_fast`

1. Resume from the fused `wy_fast_kernel.cpp` attempt rather than starting over.
2. Compare native intermediate tensors against Torch reference in this exact order:
   - packed local beta vector
   - packed local `exp(g) * beta` vector
   - `workspace_a2`
   - `workspace_a1`
3. Keep the cube GEMM path unchanged while debugging vector-side coefficient generation.
4. Reuse the debug-kernel approach that worked for `scaled_dot_kkt`:
   - one probe for beta extraction
   - one probe for local `g` extraction
   - one probe for `A2` row scaling
   - one probe for `A1` row scaling
5. Focus especially on:
   - half-chunk boundary rows
   - the last rows in each local vector slice
   - the first row of each half-chunk on the native `g` / `TEXP` path
   - whether row-wise versus column-wise scaling semantics are correct for packed BSND `A`
6. Only replace the fallback path in `dynamic_kernel_libs.py` after both fixed and packed-varlen stage checks pass.

### For `chunk_h`

1. Write down the exact native kernel contract first:
   - inputs
   - packed workspaces
   - state handoff
   - final outputs
2. Decide whether `chunk_h` should be:
   - one fused recurrent kernel, or
   - a small native kernel chain with explicit workspaces and ordering
3. Prototype the recurrence on fixed-length BSND first.
4. Add packed-varlen only after fixed-length recurrence is correct.
5. Reuse the same sequence/chunk metadata helpers already used by `chunk_o` and `scaled_dot_kkt`.
6. Pay special attention to:
   - cross-chunk state carry
   - final-state writeback shape
   - empty-tail behavior for short varlen chunks

### For performance

1. Re-benchmark native stages on large shapes after every substantial kernel change.
2. Use the static kernels as the throughput target, not just the small-stage smoke tests.
3. After correctness is stable, inspect:
   - unnecessary GM round-trips
   - oversized temporary workspaces
   - expensive vector-side scalar loops or repeated `GetValue` paths
   - synchronization points that may be over-conservative
4. Prioritize optimizing already-native fused stages first:
   - `scaled_dot_kkt`
   - `chunk_o`
5. Only then try to close the remaining gap on `wy_fast` and `chunk_h`.

## Recommended execution order for future agents

1. Keep the repository in a passing state at all times.
2. Continue native `wy_fast` debugging until the fallback can be removed safely.
3. Design and implement a native `chunk_h` recurrence path.
4. Re-run the full stage driver after each step.
5. Once all stages are native, do a dedicated performance pass.

## Files to use as primary references

- `dynamic_bsnd/scaled_dot_kkt_kernel.cpp`
- `dynamic_bsnd/chunk_o_kernel.cpp`
- `dynamic_bsnd/gdn_seq_info.h`
- `dynamic_bsnd/gdn_pto_shared.h`
- `linear_attention/linear_attention.cpp`
- `chunk_gdn/static_baseline/*.cpp`

## Important guardrail

Do not remove the current `wy_fast` or `chunk_h` fallback/orchestration paths until the native replacements pass:

- fixed-length BSND checks
- packed-varlen BSND checks
- the combined stage-validation driver

The current codebase is in a useful state because correctness is passing today, even though the port is not yet fully native.
