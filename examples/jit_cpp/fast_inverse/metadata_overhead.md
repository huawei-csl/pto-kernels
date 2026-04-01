# Metadata Overhead Comparison

This note compares two varlen BSND metadata strategies for the fast-inverse PTO kernel.

## Strategies

### 1. Device-side metadata from `cu_seqlens`

Files:
- `kernel_tri_inv_rec_unroll.cpp`
- `fast_inverse.cpp`
- `jit_util_fast_inverse.py`

Behavior:
- Python passes only `cu_seqlens` for the varlen path.
- The NPU kernel derives each chunk's row offset and `valid_size` by scanning `cu_seqlens` inside `GetBSNDVarlenTileInfoFromCuSeqlens()`.

Pros:
- Matches the Triton-style deployment API.
- No host-side metadata buffers to build or upload.
- Best end-to-end latency in the current measurements.

Cons:
- Adds a small amount of device-side work per tile.

### 2. Host-side C++ metadata precompute

Files:
- `host_chunk_metadata.cpp`
- `host_metadata_util.py`
- `kernel_tri_inv_rec_unroll.cpp`
- `fast_inverse.cpp`
- `jit_util_fast_inverse.py`

Behavior:
- A small host C++ helper builds `chunk_indices` and `chunk_valid_sizes` from `cu_seqlens`.
- Python uploads those buffers to NPU memory.
- The NPU kernel uses the precomputed metadata directly and skips the in-kernel `cu_seqlens` scan.

Pros:
- Simpler varlen metadata lookup inside the kernel.
- Kernel-only time is slightly lower or roughly equal to the device-side scan path.

Cons:
- Host metadata build plus host-to-device upload dominates the savings.
- Worse end-to-end latency in the current measurements.

## Quick Perf Summary

Benchmark setup:
- script: `benchmark_bsnd_fast_inverse.py`
- input style: Triton-unit-test-like `k` / `beta` generation
- config: `B=32`, `H=4`, `feature_dim=64`
- seqlens: `2048,8192`
- repeats: `10`
- warmup: `3`
- true-varlen samples: `0`

### `chunk_size=64`

| T | Device metadata total | Host metadata total | Host kernel only | Host metadata only |
|---|---:|---:|---:|---:|
| 2048 | 556 us | 862 us | 553 us | 309 us |
| 8192 | 2075 us | 2377 us | 2048 us | 329 us |

Takeaway:
- Device-side metadata cost is only about `3-27 us` relative to the host-precomputed kernel-only time.
- Host-side metadata costs about `309-329 us`, so it loses badly end to end.

### `chunk_size=128`

| T | Device metadata total | Host metadata total | Host kernel only | Host metadata only |
|---|---:|---:|---:|---:|
| 2048 | 1088 us | 1378 us | 1089 us | 289 us |
| 8192 | 4074 us | 4372 us | 4058 us | 314 us |

Takeaway:
- Device-side metadata overhead is effectively negligible here.
- Host-side metadata still adds about `289-314 us`, so end-to-end performance is worse.

## Conclusion

For the current implementation and tested shapes, the device-side `cu_seqlens` scan is the better overall strategy.

Reason:
- The host-C++ path does reduce or nearly eliminate kernel-side metadata overhead.
- But the saved kernel time is much smaller than the cost of building and uploading host metadata.

## How To Reproduce

From `examples/jit_cpp/fast_inverse/`:

```bash
export PTO_LIB_PATH=/sources/pto-isa

python benchmark_bsnd_fast_inverse.py \
  --chunk-size 64 \
  --seqlens 2048,8192 \
  --repeats 10 \
  --warmup 3 \
  --true-varlen-samples 0

python benchmark_bsnd_fast_inverse.py \
  --chunk-size 128 \
  --seqlens 2048,8192 \
  --repeats 10 \
  --warmup 3 \
  --true-varlen-samples 0
```

The benchmark writes:
- `benchmark_results/bench_results_bsnd_fast_inverse_64.csv`
- `benchmark_results/bench_results_bsnd_fast_inverse_128.csv`
- `benchmark_results/bench_results_bsnd_fast_inverse_bw_64.png`
- `benchmark_results/bench_results_bsnd_fast_inverse_bw_128.png`

Relevant CSV fields:
- `metadata_strategy`
- `time_us`
- `kernel_time_us`
- `metadata_time_us`
- `bw_gbs`
