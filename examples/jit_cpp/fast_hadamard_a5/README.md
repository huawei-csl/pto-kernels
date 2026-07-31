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

- `fast_hadamard_256_a5.cpp` — the kernel, standalone. Shape is the template
  parameter set `<N, Rows, Buffers, Prefetch>`; every derived constant and its
  `static_assert` lives in one `KernelShape` struct, checked per instantiation.
- `copy_ref_256_a5.cpp` — the copy-floor reference: a plain `GM → UB → GM` round
  trip over the same tiling with no vector-execute work, so it measures the DMA
  ceiling for the shape. Its own translation unit, so the transform builds and
  runs independently of it.
- `jit_a5.py` — the shared `bisheng` invocation everything builds through, so the
  flag list exists once.
- `jit_util_a5.py` — build + load for both kernels, a `Kernel` descriptor carrying
  what differs. The transform's callable pads the batch to a multiple of
  `ROWS_PER_TILE` and slices back, so **any batch size works** (the
  `matmul_swizzle` convention).
- `test_hadamard_a5.py` — correctness vs a torch reference over batch sizes
  (including non-power-of-2 and non-tile-multiple) and over every supported `N`,
  plus the packed-row padding check described below.
- `test_copy256_a5.py` — asserts the copy reference is **bit-exact** and covers
  every tile, so the floor the transform is judged against is a real copy.
- `benchmark.py` — sweeps batch × `ROWS_PER_TILE`, or block size `N` with
  `--nsweep`, reporting `hadamard / copy` in both cases.
- Plotting lives in a separate repo,
  [`pto-kernels-plots`](https://github.com/Mocchibird/pto-kernels-plots/tree/main/fast_hadamard_a5),
  alongside the generated figures. The CSV below is the contract between them.

## Build & run

Requires a real A5 device with `torch`/`torch_npu` and the CANN toolkit
(`bisheng`); set `ASCEND_HOME_PATH` (or `ASCEND_TOOLKIT_HOME`).

```bash
bash run_benchmark.sh 64                 # block_dim = number of AI cores
python benchmark.py 64 --nsweep          # block-size sweep -> build/nsweep256.csv
pytest test_hadamard_a5.py               # correctness over batch sizes and N
```

## Block size

`N` is a template parameter, default **256**, supported over **N = 32…2048**;
`-DHAD_N` only selects which instantiation the `extern "C"` entry point exposes.
A butterfly stage splits its unit of work into two half-rows; one 128-element fp16 vector
holds 128 of them, and the kernel keeps that vector full at every `N` in two directions:

- **N < 256 — packing.** `R = 256/N` rows share one 256-element window, so every stage
  drives all 128 lanes. Rows never mix: the split is on the low bit of the *within-row*
  index and `N` is even, so row `r`'s evens always land contiguously in group `r`. A packed
  window emerges with its index rotated right by `log2(N)`, which `ROT = log2(WIN) − log2(N)`
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
| rows packed (R) | 8 | 4 | 2 | **1** | 1 | 1 | 1 |
| GB/s | 2886 | 3027 | 2879 | **2897** | 2722 | 2813 | 2714 |
| copy floor GB/s | 3248 | 3136 | 3243 | **3254** | 3243 | 3258 | 3246 |
| fraction of floor | 0.89 | 0.97 | 0.89 | **0.89** | 0.84 | 0.86 | 0.84 |

Bandwidth counts read + write traffic, and every number above is from the same
`build/nsweep256.csv` the plots are generated from. Repeating the sweep three
times gives median fractions of 0.94/0.93/0.93/0.88/0.89/0.88/0.85 with spread up
to 0.06 at some `N`, so read these as +/-0.03 rather than exact.

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

`test_hadamard_a5.py` pins both, and additionally asserts that batch padding sharing a
packed window with real rows cannot contaminate them — checked against `inf`/`nan` padding,
bit-exactly, since that is the one hazard packing introduces.

## Notes

- The kernel computes the **unnormalized** transform `x @ H` (H the ±1 Hadamard
  matrix); multiply by `1/sqrt(N)` for the orthonormal WHT.
- At the kernel level, `batch` must be a multiple of `ROWS_PER_TILE` (which
  defaults to `16384/N`, i.e. 64 at N=256, so that a tile is 32 KB at every `N`);
  the Python wrapper pads to satisfy this, so callers may pass any batch.
- At large batch the kernel reaches **2.71–3.03 TB/s depending on `N`, which is
  0.84–0.97 of the measured copy floor for that `N`** (0.85–0.94 as three-run
  medians). Generated plots live in the
  companion `pto-kernels-plots` repo.
- The copy floor is ~3.25 TB/s and is a fair ceiling: a torch device-to-device
  copy of the same size measures 3.22–3.28 TB/s, so the reference kernel is
  memory-limited rather than limited by its own 2-buffer ping/pong.
- Sizing the benchmark's buffer pool matters more than it looks. A pool-size sweep
  at batch 16384 (16 MiB buffers) gave 3532/3569/3577/3415 GB/s for working sets of
  8/16/32/64 MiB and 2595/2547/2547 for 128/256/512 MiB — a cache knee between 64
  and 128 MiB. An earlier fixed 8-buffer pool sat at 64 MiB for that batch, so the
  "floor" it reported was partly cache bandwidth and every ratio measured against
  it was flattering. `WORKING_SET_BYTES` now derives the buffer count per batch.

## Implementation notes

Three constraints shape the kernel, none of them obvious from reading it.

**Loads must all precede stores within an unroll set.** A stage's sums compact into
the lower half of its group — exactly where a lower-numbered chunk still has to read
from — so a per-chunk load/store loop aliases in place. It *passes* at
`group = 512`, where the stores happen not to have landed yet, and then corrupts
silently at larger `N`. Slot `i` therefore covers `(group, chunk) = (i / chunks,
i % chunks)` so the addresses are fixed at compile time, and the comma fold's
guaranteed left-to-right evaluation is what holds the order.

**The unroll must be indexed by a parameter pack, not a loop.** An array of vector
registers indexed by a `#pragma unroll` loop variable fails to compile —
`fatal error: error in backend: Unsupported Inst must be hoisted` — because the
index is not a compile-time constant. `std::index_sequence` gives indices that are,
which is what lets the sweep be ordinary C++ instead of token-pasting macros.
Related toolchain limits: a `constexpr` *function* cannot be called from `[aicore]`
code (hence `Log2` and `RowsFor` are value templates), and `set_flag`/`wait_flag`
do not resolve inside a lambda though they do inside a plain function.

**The loop step must be a literal.** The vector loop analyser only verifies a
tripcount when the step is a literal token; a template parameter warns exactly like
a `constexpr`. Stepping by `1` and folding the stride into `base` sidesteps this
entirely, because 1 divides any bound — which is what allows `iters` to be
template-dependent.

The packing tail is fused into the last stage rather than run separately: each
`vdintlv` rotates the 256-element window right by one, ping-ponging into the
register pair `vadd`/`vsub` has just freed, so it costs no extra registers, no UB
traffic and no barrier. A rotation through UB would have to re-read the window it
just wrote. Device-verified for 1–3 rounds against `ror(index, k)`; an odd count
leaves the result in `(even, odd)`, which is why the store source is selected on
parity.
