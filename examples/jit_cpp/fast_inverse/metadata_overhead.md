# Metadata Overhead Comparison

This note compares three varlen BSND metadata strategies for the fast-inverse PTO kernel.

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

### 2. Device-side compact chunk-prefix metadata

Files:
- `host_chunk_metadata.cpp`
- `host_metadata_util.py`
- `kernel_tri_inv_rec_unroll.cpp`
- `fast_inverse.cpp`
- `jit_util_fast_inverse.py`

Behavior:
- A small host C++ helper builds a compact per-sequence cumulative chunk-count prefix.
- Python uploads that prefix together with `cu_seqlens`.
- The NPU kernel uses the prefix to binary-search the owning sequence for each chunk, instead of walking all prior sequences.

Pros:
- Reduces in-kernel metadata work compared with full `cu_seqlens` walking.
- Metadata payload is much smaller than full per-chunk host metadata.
- Better end-to-end than the full host metadata path.

Cons:
- Still requires host preprocessing and one extra metadata upload.
- Still slower end-to-end than pure device-side `cu_seqlens` scanning in the current measurements.

### 3. Host-side C++ metadata precompute

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

| T | Device scan total | Prefix total | Prefix kernel | Prefix metadata | Host total | Host kernel | Host metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 564 us | 746 us | 559 us | 187 us | 836 us | 559 us | 277 us |
| 8192 | 2071 us | 2235 us | 2049 us | 186 us | 2340 us | 2048 us | 292 us |

Takeaway:
- Prefix metadata cuts host metadata overhead from about `277-292 us` down to about `186-187 us`.
- Kernel-only time for prefix is slightly better than full device-side scanning, by about `5-22 us`.
- End to end, plain device-side `cu_seqlens` scanning is still best.

### `chunk_size=128`

| T | Device scan total | Prefix total | Prefix kernel | Prefix metadata | Host total | Host kernel | Host metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 1085 us | 1298 us | 1084 us | 214 us | 1363 us | 1080 us | 283 us |
| 8192 | 4065 us | 4253 us | 4056 us | 197 us | 4351 us | 4063 us | 288 us |

Takeaway:
- Prefix metadata cuts host metadata overhead from about `283-288 us` down to about `197-214 us`.
- Kernel-only improvement versus device scan is tiny, around `1-9 us`.
- End to end, the pure device-side scan still wins.

## Conclusion

For the current implementation and tested shapes, the device-side `cu_seqlens` scan is still the best overall strategy.

Reason:
- The compact prefix path does reduce kernel-side metadata work and is clearly better than full host per-chunk metadata.
- But the saved kernel time is still much smaller than the cost of building and uploading the prefix.
- The full host per-chunk metadata path remains the slowest end-to-end option.

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
