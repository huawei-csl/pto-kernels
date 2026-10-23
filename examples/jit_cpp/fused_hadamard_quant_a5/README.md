# fused_hadamard_quant_a5 - a Hadamard and MXFP4 quantization in one launch

`x -> Hadamard -> E2M1 nibbles + one E8M0 scale per 32`, as a single kernel on
the Ascend 950 / A5 (`dav-c310-vec`) vector core, JIT-compiled with `bisheng`
and loaded through `ctypes`. Two kernels share this directory and
`fused_hadamard_quant_common.hpp`; they differ only in the rotation.

| | rotation | K | widths |
|---|---|---|---|
| `fused_hadamard_quant_a5.cpp` | the whole row, order K | power of two | 10, from 32 to 16384 |
| `fused_hadamard_quant_b32_a5.cpp` | independent 32-blocks | need not be a power of two | 28, from 32 to 16384 |

Pick the full-row kernel when the method calls for a rotation across the whole
row. Pick block-32 when `K` is not a power of two, or when the widest rotation
is not wanted: MXFP4's scale covers 32 elements, so a 32-wide rotation already
matches the quantizer's granularity, and on heavy-tailed data it measured 4-5%
lower quantization error at K=4096, because spreading an outlier across the
whole row lifts every block's shared scale instead of just one block's.

`K` is a template parameter; one `.so` per kernel holds an instantiation per
width and the launcher dispatches on it, so there is no rebuild per size. The
block-32 rotation frees `K` from being a power of two but not from the tile:
`RowsFor` still needs `Rows * K` to be a whole 1024-element grain, which at the
default `FUSED_TILE_ELEMS` 76 of the 512 multiples of 32 up to 16384 satisfy.
A width that does not -- 11008, for instance -- fails a `static_assert` at
compile time rather than misbehaving at run time.

## Fusing the pair is 2.45-2.54x the two separate launches

Unfused, this is two passes over HBM: the butterfly writes the rotated tile out
and the quantizer reads it straight back. Fused, that tile never leaves UB and
only the nibbles and scales are written. Bytes per element tell the whole story:
6.53 unfused against 2.53 fused.

| K | 2 launches | fused | vs 2 | rel err | spread | | 2 launches | fused | vs 2 | spread |
|---|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|
| | *full-row* | | | | | | *block-32* | | | |
| 32 | | | | | | | 29.1 | 13.6 | 2.14x | 9.2% |
| 1024 | 38.8 | 28.4 | 1.37x | 0.0 | 17.7% | | 37.5 | 27.2 | 1.38x | 14.5% |
| 4096 | 297.7 | 121.7 | **2.45x** | 0.0 | 2.8% | | 293.4 | 119.7 | **2.45x** | 3.7% |
| 8192 | 612.3 | 247.4 | **2.47x** | 0.0 | 2.2% | | | | | |
| 16384 | 1209.9 | 493.8 | **2.45x** | 0.0 | 1.0% | | 1206.1 | 474.6 | **2.54x** | 1.3% |

M = 16384, microseconds per launch, what `benchmark.py` prints. Both arms agree
to a relative error of 0.0, checked before either is timed. Byte traffic
predicts 6.53 / 2.53 = 2.58x, and the clean widths measure 2.45-2.54x.

Two rows are not traffic results, for different reasons, and are in the sweep
because the effects are worth seeing rather than because they are the headline.
At K=1024 the unfused intermediate is `2*M*k` = 32 MB against a 128 MiB L2, so
the unfused arm reads much of it from cache rather than HBM, which flatters the
arm fusing is measured against; the bracket spread on those rows is the same
thing showing up as noise. At K=32 a row is 0.5M elements at M=16384 and the
fused arm's 13.6 us is the dispatch floor, so its 2.14x is two launches against
one rather than anything about bytes.

## It runs at about the speed of a copy of its input

| K | fused | d2d copy | vs copy | fused GB/s | copy GB/s | | fused | d2d copy | vs copy | fused GB/s | copy GB/s |
|---|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|
| | *full-row* | | | | | | *block-32* | | | | |
| 32 | | | | | | | 122.1 | 191.1 | 1.57x | 1391 | 1404 |
| 1024 | 121.5 | 191.5 | 1.58x | 1398 | 1402 | | 122.4 | 192.4 | 1.57x | 1388 | 1395 |
| 4096 | 122.9 | 191.7 | 1.56x | 1382 | 1400 | | 117.1 | 192.8 | **1.65x** | 1450 | 1392 |
| 8192 | 122.5 | 192.3 | 1.57x | 1386 | 1396 | | | | | | |
| 16384 | 122.5 | 189.3 | 1.55x | 1387 | 1418 | | 117.3 | 192.4 | **1.64x** | 1448 | 1395 |

67 million elements per launch, whatever `K` is. The fused column is flat across
a 16x range of row width, because the transform is entirely hidden under the
DMA at every width. Both arms reach much the same bandwidth -- 1382-1450 GB/s
against the copy's 1392-1420 -- so neither kernel is moving bytes faster than a
copy; both are moving 1.58x fewer of them, 2.53 B/element against 4.0.

Measured on an `Ascend950PR_9589`: 64 vector cores, 128 MiB L2, 1.65 GHz, HBM
peak 1.6 TB/s, so the kernels reach 86-91% of peak. The copy is a reference for
what moving the bytes costs, not a proven lower bound -- it is a vendor kernel
doing a simpler job. HBM peak is the closer thing to a real ceiling. Other A5
parts have different HBM, so absolute GB/s should not be compared across parts;
the ratios are the portable numbers.

Two changes got the full-row kernel there, and their sizes are worth recording.
The cross-window stages were originally addressed by a shift-and-OR index
computed per register slot inside the unrolled fold, which cost 266 us of a 388
us kernel at K=16384; nested loops over `base + m*step`, same memory pattern and
same number of passes, cut that phase to about 23 us. Fusing the passes on top
(`FUSED_CROSS_FUSE`) added a further 1.16x. For block-32 the tile size was what
mattered, and a `vsts` to `vscatter` swap with bit-identical output cost +29 us.

## Correctness

Neither kernel can be bit-exact against a torch expression: they rotate in bf16
with a specific operand order and no torch formulation reproduces that tree. So
`test_fused_hadamard_quant_a5.py` establishes correctness three ways, strongest
first, for both.

1. **Scale bytes** must match a reference that rotates in fp32 and quantizes
   with `torch_npu`. A scale is a power of two derived from a block maximum, so
   bf16 rounding inside the butterfly almost never moves it -- disagreeing
   scales mean a wrong rotation, not different rounding. Threshold 98%.
2. **Dequantized values** must track that reference to within MXFP4's own
   resolution. This catches a correct-looking permutation, which a check on the
   packed bytes would not. Threshold 5%.
3. **The output must be non-trivial.** A kernel that writes nothing, or writes
   its input back, is the characteristic silent failure on this hardware and
   would pass a loose tolerance. Separate tests assert the nibbles are neither
   all-zero nor degenerate, and that the result differs from quantizing without
   the rotation.

A constant row separates the two rotations sharply: the full-row kernel must
produce one delta for the whole row, block-32 one delta per 32-block. Each is
the other's characteristic bug.

Both width lists span both of their kernel's **addressing classes**, because
both have hidden real bugs here. Full-row splits on `chunks` -- whether a group
is a whole window of several rows or one row spread over 2 to 16 chunks -- and
block-32 on the unroll, by 8 or by 4 depending on `rows_for(k) * k / 256`. Class
membership is derived from the built `.so` rather than hardcoded, because
raising the tile size once moved four widths between classes and left a matrix
single-class. One test also uses a batch deep enough that every core walks
several tiles, since a shallow batch leaves the buffer rotation, the prefetch
and the drain unexercised.

```bash
python3 -m pytest -q test_fused_hadamard_quant_a5.py   # 91 tests, both kernels
```

## Running the benchmark

```bash
./run_benchmark.sh                       # both kernels
./run_benchmark.sh --kernel b32          # or full
```

Needs a CANN whose PTO carries MXFP4: the kernels pack through
`vector_f4e2m1x2`, declared in `pto/npu/a5/datatype.hpp`. 9.1.0 and 9.2.0 have
it; 9.0.0 does not.

## Tunables

All compile-time, with the shipped defaults, and shared by both kernels. Every
combination is checked by `static_assert`, so a tile that will not fit UB or a
prefetch depth that would deadlock fails to compile rather than misbehaving.

| flag | default | what it is |
|---|---|---|
| `FUSED_TILE_ELEMS` | 24576 | elements per UB tile, 48 KB in bf16 |
| `FUSED_BUFFERS` | 3 | UB pipeline buffers |
| `FUSED_PREFETCH` | 2 | tiles in flight ahead |
| `FUSED_CROSS_FUSE` | 3 | full-row only: cross-window stages fused per pass |

The same source builds the reduced kernels the ladder needs:
`FUSED_ROTATE_ONLY` drops the quantizer and `FUSED_NO_ROTATE` drops the
butterfly, so the two arms differ in what they fuse and in nothing else.
`FUSED_BUFFERS` above 4 does not build at K=4096 -- five slots need 311,040
bytes of UB against 253,952 available.

## The unnormalised factor

The butterfly is the unnormalised Sylvester matrix, so its output is `sqrt(K)`
larger than an orthogonal Hadamard's for the full-row kernel and `sqrt(32)`
larger for block-32. That factor is left to the caller: scale `x` by its inverse
going in if orthogonal semantics are wanted.

`E8M0` cannot absorb it in general. It is a power-of-two scale, and `sqrt(32)`
is not a power of two; for the full-row kernel the factor is a power of two only
at even `log2(K)` -- 8, 16, 32, 64 at K = 64, 256, 1024, 4096 -- and not at 32,
128, 512 or 2048. Leaving it out is the one behaviour that holds at every
supported width.
