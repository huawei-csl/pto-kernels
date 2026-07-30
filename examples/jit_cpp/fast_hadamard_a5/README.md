# fast_hadamard_256_a5 — Walsh–Hadamard on Ascend A5

A register/DMA-fused fast Walsh–Hadamard transform (WHT) on the Ascend 950 / A5
(`dav-c310`) vector core, exposed via JIT `bisheng` compilation + `ctypes`.
Block size is a build-time macro: **N = 32…2048**, default 256.

Each of the `log2(N)` butterfly stages does the even/odd split on the
deinterleave **load** (`vlds DINTLV_B16`) and the concat-halves recombine on the
**store** (`vsts` to the group's two halves), so only `vadd`/`vsub` ever touch
the vector-execute pipe. The transform is therefore **memory-bound** and runs at
essentially HBM copy speed — the natural yardstick is a plain `GM→UB→GM` copy of
the same tiling, benchmarked alongside it.

## Files

- `fast_hadamard_256_a5.cpp` — the kernel (`hadamard256`), standalone.
- `jit_util_hadamard256_a5.py` — `bisheng` build + `ctypes` load. The returned
  callable pads the batch up to a multiple of `ROWS_PER_TILE` and slices back,
  so **any batch size works** (following the `matmul_swizzle` padding convention).
- `copy_ref_256_a5.cpp` — the copy-floor reference (`copy256`): a plain
  `GM → UB → GM` round trip over the same tiling, with no vector-execute work, so
  it measures the DMA ceiling for the shape. Its own translation unit, so the
  transform above builds and runs independently of it.
- `jit_util_copy256_a5.py` — build + load for the copy reference.
- `test_copy256_a5.py` — asserts the copy reference is **bit-exact** and covers every
  tile, so the floor the transform is judged against is known to be a real copy.
- `test_hadamard256_a5.py` — correctness vs a torch reference across batch sizes,
  including non-power-of-2 and non-tile-multiple ones.
- `test_block_sizes_a5.py` — correctness at every supported block size, covering both
  the packed N<256 and chunked N>256 paths, plus the padding check (see below).
- `benchmark.py` — sweeps batch × `ROWS_PER_TILE`, or block size `N` with `--nsweep`,
  reporting `hadamard / copy` in both cases.
- `plot_hadamard256_a5.py` — heatmap + bandwidth-vs-batch plot for the batch sweep.
- `plot_hadamard_nsweep_a5.py` — the block-size sweep from `benchmark.py --nsweep`.

## Build & run

Requires a real A5 device with `torch`/`torch_npu` and the CANN toolkit
(`bisheng`); set `ASCEND_HOME_PATH` (or `ASCEND_TOOLKIT_HOME`).

```bash
bash run_benchmark.sh 64                 # block_dim = number of AI cores
python benchmark.py 64 --nsweep          # block-size sweep -> build/nsweep256.csv
pytest test_hadamard256_a5.py            # correctness (incl. non-power-of-2 batches)
python plot_hadamard256_a5.py            # -> build/hadamard256_grid.png (needs matplotlib)
```

## Block size

`N` (`HAD_N`) is a build-time macro, default **256**, supported over **N = 32…2048**.
A butterfly stage splits its unit of work into two half-rows; one 128-element fp16 vector
holds 128 of them, and the kernel keeps that vector full at every `N` in two directions:

- **N < 256 — packing.** `R = 256/N` rows share one 256-element window, so every stage
  drives all 128 lanes. Rows never mix: the split is on the low bit of the *within-row*
  index and `N` is even, so row `r`'s evens always land contiguously in group `r`. A packed
  window emerges with its index rotated right by `log2(N)`, which `HAD_ROT = 8 − log2(N)`
  `vdintlv` finish — one register-to-register op per rotation, no UB traffic, fused into
  the final stage using the `(e,o)` registers that are dead by then.
- **N > 256 — chunking.** A row spans `CHUNKS = (N/2)/128` windows, processed as
  independent pieces. (N=4096 would need 16 chunks, more than one unroll set, and is
  rejected at compile time.)

The two are mutually exclusive by construction — `R > 1` implies a 256-element group implies
`CHUNKS == 1` — which is asserted, and is what keeps the inner loop readable.

Measured on A5, tile size and total bytes held constant, each `N` against its own rebuilt
copy floor (`python benchmark.py 64 --nsweep`):

| N | 32 | 64 | 128 | **256** | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| rows packed (R) | 8 | 4 | 2 | 1 | 1 | 1 | 1 |
| GB/s | 2712 | 2700 | 2691 | **2664** | 2624 | 2590 | 2554 |
| fraction of copy floor | 0.96 | 0.94 | 0.94 | **0.94** | 0.92 | 0.91 | 0.90 |

The transform is now memory-bound at every supported `N`. Vector-op cost per element is
`(5·log2(N) + log2(R)) / 256`, lowest at N=32 — packing removes the lane waste that
previously left N=32 at 0.30 of the floor and N=128 at 0.76.

**Two ordering rules, if you extend either path.**

1. *Keep all loads ahead of all stores.* A stage's sums compact into the lower half of the
   group, exactly where a lower-numbered chunk still has to read from, so a per-chunk
   load/store loop aliases in place. It also *appears* to work at N=512 while the stores
   have not landed yet, then corrupts silently at larger N. The unroll slots span
   `(group, chunk)` pairs for this reason.
2. *Keep the rotation in registers.* A rotation pass through UB would re-read the window it
   just wrote and would need a `mem_bar(VST_VLD)`; omitting it is the same silent-corruption
   trap as (1). The fused `vdintlv` tail cannot hit it.

`test_block_sizes_a5.py` pins both, and additionally asserts that batch padding sharing a
packed window with real rows cannot contaminate them — checked against `inf`/`nan` padding,
bit-exactly, since that is the one hazard packing introduces.

## Notes

- The kernel computes the **unnormalized** transform `x @ H` (H the ±1 Hadamard
  matrix); multiply by `1/sqrt(N)` for the orthonormal WHT.
- At the kernel level, `batch` must be a multiple of `ROWS_PER_TILE` (which
  defaults to `16384/N`, i.e. 64 at N=256, so that a tile is 32 KB at every `N`);
  the Python wrapper pads to satisfy this, so callers may pass any batch.
- At large batch the kernel reaches **2.55–2.71 TB/s depending on `N`, which is
  0.90–0.96 of the measured copy floor for that `N`**. Generated plots live in the
  companion `pto-kernels-plots` repo.
