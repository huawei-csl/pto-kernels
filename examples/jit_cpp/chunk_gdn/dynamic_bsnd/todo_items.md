# Dynamic BSND GDN Todo Items

This file is a handoff note for the `dynamic_bsnd` port.

It summarizes:

- what currently passes
- what was completed
- what the remaining optimization opportunities are

## What is passing today

All five stage kernels are fully native PTO kernels with no Torch fallback or host-side orchestration. The stage-validation driver `run_gated_delta_dynamic_bsnd.py` passes all checks:

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

- `chunk_cumsum`: fixed `0.064 ms`, packed-varlen `0.063 ms`
- `scaled_dot_kkt`: fixed `0.066 ms, 0.51 TFLOP/s`, packed-varlen `0.065 ms, 0.39 TFLOP/s`
- `wy_fast`: fixed `0.167 ms, 0.40 TFLOP/s`, packed-varlen `0.167 ms, 0.30 TFLOP/s`
- `chunk_h`: fixed `0.144 ms`, packed-varlen `0.146 ms`
- `chunk_o`: fixed `0.197 ms, 0.34 TFLOP/s`, packed-varlen `0.199 ms, 0.25 TFLOP/s`

## Completed milestones

### `wy_fast` — fully native (was hybrid)

Previous state:

- PTO cube kernels handled `A1 @ K` and `A2 @ V` matmuls.
- Torch/NPU helper code still built the packed `A1` and `A2` coefficient tensors on the host.
- Performance was ~1.9 ms (0.03 TFLOP/s).

What was done:

- Replaced the scalar `TMULS` loops for row-wise coefficient scaling with `TROWEXPANDMUL` tensor operations.
- The scalar loops had systematic corruption at rows 62, 63, 126, 127 (last two rows of each half-chunk) caused by pipeline synchronization issues between the scalar and vector pipes.
- `TROWEXPANDMUL` performs the entire row-wise scaling in one tensor operation, eliminating the pipeline sync problem.
- Both `A1 = A * (exp(g) * beta)` and `A2 = A * beta` coefficient builds are now fully kernel-side.
- The Torch fallback in `dynamic_kernel_libs.py` was removed; the fused `call_kernel` entry point handles everything.

Result:

- Performance improved from ~1.9 ms to ~0.17 ms (over 10x speedup).
- Both fixed-BSND and packed-varlen checks pass.

### `chunk_h` — fully native (was hybrid)

Previous state:

- PTO cube kernels handled `W @ S` and `K^T @ new_v` matmuls.
- The chunk-by-chunk recurrence, `new_v` computation, coefficient calculation, and final-state propagation were all driven on the host with Python loops and `torch.npu.synchronize()` calls.
- Performance was ~4.6 ms.

What was done:

- Designed and implemented a single fused PTO cube+vector kernel with a 4-point cross-core handshake per chunk iteration.
- Cube computes `ws = W @ state` (flag 0) and `kv = k_scaled^T @ new_v` (flag 2).
- Vector computes coefficients via `TROWEXPANDMUL`, `new_v = U - ws`, and updates `state = state * exp(g_last) + kv` (flags 1, 3).
- Each block processes one `(sequence, head)` work item and iterates sequentially over all chunks in the sequence.
- State is carried between chunks via a per-block half-precision GM workspace (3 slots: ws/kv, k_scaled, state).
- Both vector sub-blocks always process their 64-row portion of the 128x128 state, even when `local_rows == 0` for K/U/new_v data.
- Cross-core flag 3 is only signaled when there is a next chunk, preventing stale flags across work items.
- K is loaded from BSND layout with dynamic zero-padded UB tiles; new_v is stored with dynamic stores to preserve zero-padding.
- The entire host-side loop and per-chunk `synchronize()` calls were removed from `dynamic_kernel_libs.py`.

Result:

- Performance improved from ~4.6 ms to ~0.14 ms (over 30x speedup).
- Both fixed-BSND and packed-varlen checks pass.

## Remaining work: performance optimization

All five stages are now correct and fully native. The remaining opportunity is closing the performance gap with the static baseline kernels.

### Known optimization targets

1. **Large-shape benchmarking**: Current timings are from small test shapes. Re-benchmark on production-size inputs to measure the real gap against static baselines.

2. **GM traffic reduction**: Several stages still round-trip intermediate data through GM workspaces where on-chip reuse might be possible.

3. **Workspace sizing**: `chunk_h` allocates `block_dim * 3 * D * D` half elements of workspace. This could potentially be reduced by overlapping slots that are not live at the same time.

4. **Synchronization granularity**: Some `pipe_barrier(PIPE_ALL)` calls could be replaced with more targeted pipeline flags to reduce stall time.

5. **Vector-side efficiency**: Coefficient construction paths in `wy_fast` and `chunk_h` could potentially be further streamlined (e.g., precomputing shared values once across sub-blocks).

6. **Dynamic indexing overhead**: The `GdnBsndSeqInfo` helper and per-chunk `valid_rows` / `local_rows` calculations add scalar overhead that doesn't exist in the static kernels.

### Recommended approach

1. Profile each stage individually on large shapes.
2. Identify whether the bottleneck is compute, memory bandwidth, or launch/sync overhead.
3. Optimize the highest-impact stage first.
4. Re-run the full stage driver after each change to guard against regressions.

## Files to use as primary references

- `dynamic_bsnd/wy_fast_kernel.cpp` — fused cube+vector with `TROWEXPANDMUL` coefficient build
- `dynamic_bsnd/chunk_h_kernel.cpp` — fused cube+vector with cross-core recurrence
- `dynamic_bsnd/chunk_o_kernel.cpp` — fused cube+vector with BSND output store
- `dynamic_bsnd/scaled_dot_kkt_kernel.cpp` — fused cube+vector with coefficient masking
- `dynamic_bsnd/gdn_seq_info.h` — sequence/chunk metadata helpers
- `dynamic_bsnd/gdn_pto_shared.h` — cross-core sync and tile helpers
- `linear_attention/linear_attention.cpp` — cross-core fusion reference
- `chunk_gdn/static_baseline/*.cpp` — static performance targets
