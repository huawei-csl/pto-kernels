# fast_hadamard_a5 — Walsh–Hadamard on Ascend A5

A register/DMA-fused fast Walsh–Hadamard transform (WHT) on the Ascend 950 / A5
(`dav-c310`) vector core, exposed via JIT `bisheng` compilation + `ctypes`.
Block size is a template parameter: **N = 32…2048**, default 256. One .so
holds an instantiation per N and the launcher dispatches on it, so there is no
rebuild per size.

Each of the `log2(N)` butterfly stages does the even/odd split on the
deinterleave **load** (`vlds DINTLV_B16`) and the concat-halves recombine on the
**store** (`vsts` to the group's two halves), so only `vadd`/`vsub` ever touch
the vector-execute pipe. The transform is therefore **memory-bound** and runs at
**0.87–0.95 of a torch device-to-device copy of the same bytes**, which is the
yardstick throughout. What is left is the `log2(N)` UB round trips the butterfly
needs, so the fraction widens with `log2(N)`.

## Files

- `fast_hadamard_a5.cpp` — the kernel, standalone. Shape is the template
  parameter set `<N, Rows, Buffers, Prefetch>`; every derived constant and its
  `static_assert` lives in one `KernelShape` struct, checked per instantiation.
- `jit_a5.py` — the shared `bisheng` invocation everything builds through, so the
  flag list exists once.
- `jit_util_a5.py` — build + load. The callable pads the batch to a multiple of
  `ROWS_PER_TILE` and slices back, so **any batch size works** (the
  `matmul_swizzle` convention).
- `test_hadamard_a5.py` — correctness vs a torch reference over batch sizes
  (including non-power-of-2 and non-tile-multiple) and over every supported `N`,
  plus the packed-row padding check described below.
- `benchmark.py` — sweeps `ROWS_PER_TILE` × batch; `--tiles` sweeps tile size ×
  `N`, `--nsweep` sweeps block size. Every row carries the pool depth, rep count
  and microseconds per launch behind it, and a `status` column that says when a
  row is not a usable bandwidth ratio.
- Plotting lives in a separate repo,
  [`pto-kernels-plots`](https://github.com/Mocchibird/pto-kernels-plots/tree/main/fast_hadamard_a5),
  alongside the generated figures. The CSV below is the contract between them.

## Build & run

Requires a real A5 device with `torch`/`torch_npu` and the CANN toolkit
(`bisheng`); set `ASCEND_HOME_PATH` (or `ASCEND_TOOLKIT_HOME`).

```bash
bash run_benchmark.sh 64                        # block_dim = number of AI cores
python benchmark.py 64 --nsweep --repeat 3      # block size -> build/nsweep.csv
python benchmark.py 64 --repeat 3               # ROWS x batch -> build/grid.csv
python benchmark.py 64 --tiles --repeat 3       # tile x N -> build/tiles.csv
pytest test_hadamard_a5.py                      # correctness over batch sizes, N
```

`--repeat` takes the median of that many full measurements. It is worth using: a
single measurement of the transform and its reference can disagree with the next
by more than any effect being looked for.

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

Measured on A5, tile size and total bytes held constant, each `N` against a torch
copy of the same bytes (`python benchmark.py 64 --nsweep --repeat 3`):

| N | Packed | Chunked | Throughput (GB/s) | Copy (GB/s) | Ratio |
|---|---|---|---|---|---|
| 32 | 8 | 1 | 2854 | 3024 | 0.943 |
| 64 | 4 | 1 | 2865 | 3025 | 0.947 |
| 128 | 2 | 1 | 2845 | 3026 | 0.941 |
| **256** | **1** | **1** | **2829** | **3025** | **0.935** |
| 512 | 1 | 2 | 2755 | 3026 | 0.910 |
| 1024 | 1 | 4 | 2754 | 3025 | 0.910 |
| 2048 | 1 | 8 | 2640 | 3025 | 0.872 |

The GM<->UB tile is 16 KB, which `--tiles` shows is a genuine optimum and not
just a good guess: at N=256 the ratio is 0.79 / **0.92** / 0.90 / 0.86 for
8 / 16 / 32 / 64 KB tiles, and 16 KB wins at every `N`. Smaller tiles mean
fewer bytes per UB pass, which is what a UB-round-trip-bound kernel wants,
until per-tile overhead takes over below 16 KB.

Bandwidth counts read + write traffic. The reference measures **3024..3026 GB/s
across every `N` — a 0.08% spread** — so the ratio reflects the kernel and not the
reference. Every `N` moves the same 16.7 M elements through a derived pool depth,
so the working set is an identical ~256 MiB in every row, past the cache knee.

The ratio falls with `log2(N)`: each stage is one more UB round trip, and
vector-op cost per element is `(5·log2(N) + log2(R)) / 256`. Packing is what makes
small `N` cheap rather than expensive — before it, N=32 sat at 0.30 of the floor
and N=128 at 0.76.

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

## Tiling: 16 KB, measured

`python benchmark.py 64 --tiles --repeat 3` sweeps GM↔UB tile size against block
size. Both are shape knobs a caller chooses, and every cell moves the same 16.7 M
elements, so the reference depends only on `N` and the ratios are comparable:

| tile | N=32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| 8 KB | 0.821 | 0.804 | 0.792 | 0.785 | 0.768 | 0.753 | 0.734 |
| **16 KB** | **0.945** | **0.929** | **0.924** | **0.916** | **0.899** | **0.895** | **0.858** |
| 32 KB | 0.930 | 0.907 | 0.902 | 0.896 | 0.884 | 0.883 | 0.851 |
| 64 KB | 0.907 | 0.885 | 0.879 | 0.859 | 0.843 | 0.813 | 0.790 |

**16 KB is fastest at every supported `N`.** Every figure here is the median
of four independent sweeps (each itself `--repeat 3`, so twelve measurements a
cell), which matters: a single sweep moves a cell by up to 3.5 points, wider
than the 0.8–2.2 point margins being compared, and reads systematically high.
Averaging brings the residual under ~0.9 points, and only then is the result
resolved rather than merely observed — which is why
`TILE_BYTES` is 16 KB and `ROWS_PER_TILE` defaults to `8192/N`. An earlier sweep
compared 32 / 64 / 128 KB, found 32 KB best, and never tried smaller. 8 KB is much worse
because at `NBUF=4` it leaves too little in flight to hide the DMA; a 16 KB tile at
`NBUF=4` uses only 64 KB of the 256 KB UB, so what binds there is pipeline depth,
not space.

`cfg()` caps `NBUF` at 4 and `PREFETCH` at 2, and `PREFETCH` must stay below
`NBUF` or the pipeline deadlocks. Going past that cap needs a patched build — the
harness exposes no flag for it — so no `NBUF` figures are quoted here; every
number in this file comes from a sweep `benchmark.py` can reproduce.

## Notes

- The kernel computes the **unnormalized** transform `x @ H` (H the ±1 Hadamard
  matrix); multiply by `1/sqrt(N)` for the orthonormal WHT.
- At the kernel level, `batch` must be a multiple of `ROWS_PER_TILE` (which
  defaults to `8192/N`, i.e. 32 at N=256, so that a tile is 16 KB at every `N`);
  the Python wrapper pads to satisfy this, so callers may pass any batch.
- At large batch the kernel reaches **2.64–2.87 TB/s depending on `N`, which is
  0.87–0.95 of a torch device-to-device copy of the same bytes**. Generated plots
  live in the companion `pto-kernels-plots` repo.
- Sizing the benchmark's buffer pool matters more than it looks. A pool-size sweep
  at batch 16384 (16 MiB buffers) gave 3532/3569/3577/3415 GB/s for working sets of
  8/16/32/64 MiB and 2595/2547/2547 for 128/256/512 MiB — a cache knee between 64
  and 128 MiB. A fixed 8-buffer pool sat at 64 MiB for that batch, so the "floor"
  it reported was partly cache bandwidth and every ratio measured against it was
  flattering. `WORKING_SET_BYTES` now derives the pool depth per batch, holding
  the working set at ~256 MiB. `POOL_MAX` has to stay loose enough for the
  smallest batch to reach it — 512 buffers of 0.5 MiB at batch 1024 — or the
  footprint grows with batch again and the sweep varies two things at once. A cap
  of 16 did exactly that: it pinned the footprint only from batch 32768 up, and
  bandwidth-vs-batch kinked where the smaller batches crossed the knee (batch
  16384 read 2542 GB/s cache-fed against 2131 measured at a full footprint).
- **Three measurement rules the harness now enforces rather than assumes**, each
  because it went wrong first: a batch whose device time is under 20 µs per launch
  is refused (host dispatch outruns the device — batch 1024 read 216 GB/s for the
  transform against 200 for the copy, a meaningless "ratio" of 1.08); rep counts
  are derived so every trial moves 8 GiB (at a fixed 50 reps the trial spread was
  7–9%, wider than anything being measured); and a ratio above 1.0 or a copy above
  the HBM bound is reported as broken rather than plotted. The batch × ROWS grid
  this replaces had **24 of its 36 cells above 1.0**, drawn as the greenest on the
  map because the colour scale was clamped at 1.0.

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
