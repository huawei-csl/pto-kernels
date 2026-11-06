# mxfp4_matmul_a5 — MXFP4 x MXFP4 matmul on the A5 cube

A full `y = A @ B` over the whole `(M, K) x (K, N)`, with **both** operands
MXFP4 -- E2M1 nibbles carrying one E8M0 scale per 32 elements along K --
accumulated in fp32 and stored bf16, on the Ascend 950 / A5 (`dav-c310`)
cube's native microscaled path. JIT-compiled with `bisheng` and driven
through `ctypes`.

The 32 is MXFP4's scale granularity and nothing else: the matmul itself is
not blocked. Do not read it as the block-32 rotation that
`fused_hadamard_quant_b32_a5` names.

```
A: (M, K)  E2M1 nibbles, two per byte, + one E8M0 scale per 32 along K
B: (K, N)  the same, stored DN
y: (M, N)  bf16
```

A5 has a microscaled matmul in hardware, `TMATMUL_MX`, whose semantics are
MXFP4's exactly. From PTO's own CPU reference (`pto/cpu/TMatmul.hpp`):

```
acc += a(i, k) * b(k, j) * aScale(i, k / 32) * bScale(k / 32, j)
```

The scaling is in the datapath, so this is not a dequantize-and-matmul. And
`aScale` is `(M, K/32)`, which is what `mxfp4_quant_a5` already writes, so the
two kernels meet with no repacking.

## Running it

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 9.1.0 or newer
python -m pytest test_mxfp4_matmul_a5.py -q          # 20 tests
./run_benchmark.sh --out results.csv                 # the full grid, ~4 min
./run_benchmark.sh --kn 4096 --m 16 1024 16384       # one column
```

K and N are compile-time template arguments, because the GM strides are
per-layer constants and static addressing is what keeps the inner loop tight.
One `.so` therefore serves one `(K, N)` pair and the build cache is keyed on
both. M is a runtime argument.

**K must be a whole L1 slab**, so a multiple of 512 at the default `MXMM_K_L1`,
and N a multiple of 64 for fp4. The default build shape is 512x512 for that
reason. `jit_util_mxfp4_matmul_a5.py` rejects anything else before launching,
which matters more here than usual: the kernel cannot report an error from the
device, so handed a shape it was not built for its launcher returns having
written nothing and the caller gets an untouched buffer back.

M is rounded **up** to whole tiles. The result is `m_round(m)` rows tall, not
`m`, and rows past `m` hold whatever the padded operands produced. Read the
height off the returned tensor.

## Measured against torch_npu

`torch_npu`'s `npu_quant_matmul` reaches the same hardware path, so it is the
arm that matters. Ratios below are **ours / vendor**, above 1 meaning this
kernel is faster. Each figure is the median of six separate processes, each
of which took the median of up to 100 interleaved reps.

| K=N | ours / vendor | ours wins | ours peak | vendor peak | ours as % of vendor peak |
|---|--:|--:|--:|--:|--:|
| 512 | **1.77x** | 16/16 | 378 TF/s | 280 TF/s | 135% |
| 1024 | **1.71x** | 16/16 | 808 | 761 | 106% |
| 2048 | **1.67x** | 14/16 | 1236 | 1316 | 94% |
| 3072 | **1.48x** | 14/16 | 1399 | 1514 | 92% |
| 4096 | **1.42x** | 13/16 | 1488 | 1590 | 94% |
| 6144 | **1.19x** | 13/16 | 1536 | 1662 | 92% |
| 8192 | **1.05x** | 12/16 | 1555 | 1685 | 92% |
| 12288 | 0.89x | 1/16 | 1590 | 1698 | 94% |
| 16384 | 0.80x | 1/16 | 1607 | 1712 | 94% |

The shape of it is one mechanism. The vendor kernel reaches 1685–1712 TF/s,
which is the MXFP4 cube ceiling (1692 TF/s from the L0 format ratio), so at
large K=N there is nothing left to win and this kernel's 92–94% of that peak
is the whole gap. At the other end the vendor is on a dispatch floor of about
38 us — at K=N=512 it measures 39.3 us at M=1 and 37.7 us at M=1024, flat
across three decades of work — and there it loses to a plain bf16 GEMM:

| K=N | ours vs bf16 | vendor vs bf16 |
|---|--:|--:|
| 512 | 1.29x | 0.71x |
| 1024 | 1.30x | 0.75x |
| 2048 | 1.58x | 0.94x |
| 4096 | 2.19x | 1.47x |
| 8192 | 2.15x | 2.02x |
| 16384 | 2.28x | 2.90x |

So this kernel beats bf16 at every width measured, and the vendor does not
below K=N=3072.

M matters as much as K=N, and in the same direction: more work per launch helps
the vendor. At K=N=4096 this kernel runs 1.44x the vendor at M=16 and 0.94x at
M=32768. The M at which it first falls behind drops sharply as K=N grows:

| K=N | 512 | 1024 | 2048 | 3072 | 4096 | 6144 | 8192 | 12288 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| first losing M | never | never | 16384 | 16384 | 8192 | 8192 | 4096 | 1 |

`ours_vs_vendor` for all 144 cells is in the CSV.

### What is not in these numbers

Both MXFP4 arms are timed on **pre-quantized operands**. Quantization is a
separate kernel (`mxfp4_quant_a5`) and is in neither figure, so this is matmul
against matmul.

Two asymmetries, both stated rather than corrected:

* `npu_quant_matmul` has no `out=` and allocates its result on every call.
  Measured separately, that allocation is 2.2 us against a caching allocator,
  flat in size. At K=N=512 it is a few percent of the vendor arm and it
  flatters this kernel; there is no way to remove it through the public API.
* Both arms have their dtype views and transposes hoisted out of the timed
  region. Leaving the vendor's six `view`/`t`/`transpose` calls inside the loop,
  which an earlier harness did, costs that arm a flat **8.5–10.1 us per call**
  — measured by A/B of the two constructions in one process, and flat across
  shapes because it is host-side work. That is 10–35% at small shapes and moved
  `ours_vs_vendor` at M=16 K=N=1024 from 1.81x to 2.29x. Neither arm should be
  charged for per-call metadata bookkeeping.

This kernel's own wrapper has the identical trap in reverse: `load_matmul`
re-derives the tile plan per call, three ctypes round-trips into the `.so`,
which read 1.30x at that same shape. Use `matmul.prepare(...)` for anything
timed or hot, as `benchmark.py` does.

### Method

Every part of this is load-bearing on a shared box.

* **L2 is evicted before every timed launch**, outside the timer, by copying a
  buffer twice the size of L2. That models a forward pass, where a layer's
  weights are touched once per token and the model dwarfs the 128 MB L2. A
  back-to-back loop leaves the weights resident and favours whichever format is
  smaller.
* **Arms are interleaved in one measurement window**, order rotated per rep, so
  a drift in machine load cannot land on one arm.
* **Wall clock over a synchronised launch.** The NPU event timer has reported
  82, 28, 7.6 and 24 us for one and the same launch.
* **A device-to-device copy reference per run**, printed and stored in the CSV.
  It held at 1424–1435 GB/s across the six runs behind the table above, which
  is this part's normal figure; a collapse means another tenant and the ratios
  from that run are not comparable.
* **Six processes, not one.** The vendor op is bimodal per process — it picks
  a kernel at process start — so a single process can settle a near-tie the
  wrong way, and it is the largest source of disagreement here by some margin.
  Across the six, 14 of 144 shapes disagree on the sign. Eleven of those are one
  process in which the vendor took its faster kernel at K=N=12288 and at
  M=16, K=N=8192 — that is the bimodality itself, not noise — and the other
  three sit at a median of 0.99–1.00 and are true near-ties. Read as two
  independent triples, the medians agree to 1.8% per shape and 11.2% at worst,
  and 2 of 144 cells change sign, both of them on 1.00x.

Absolute microseconds and TFLOP/s here are from one box, an `Ascend950PR_9589`
whose HBM runs at about 1.6 TB/s. Ratios between arms measured in the same run
carry over; absolute throughput does not, and a 950DT part will not match these
figures.

## Structure

The layout is asymmetric and none of it is guessable from the type names.
Getting any one of these wrong produces a plausible-looking wrong answer rather
than a failure, which is why each is called out at its declaration in the
source:

| | data | scale |
|---|---|---|
| A | `Layout::ND`, row-major `(M, K)` | `MX_A_ND`, row-major `(M, K/32)` |
| B | `Layout::DN`, **not** ND | `MX_B_DN`, **not** `MX_B_ND` |

B being DN costs nothing, since weights are prepared offline. Three more traps
in the same class: GM strides come from a `BaseShape2D` of the **whole matrix**,
not of the tile; fp4 pointer arithmetic is in **bytes**, so a K-element step is
`K >> 1`; and scale tensors are **paired** along K, which is the real reason K
must be a multiple of 64 rather than 32. The scale tiles also bind by address —
the hardware reads them at `GetScaleAddr(operand.data())`, a `>> 4` of the
operand's L0 address — so `TASSIGN`ing them anywhere else compiles and
multiplies by whatever happens to be there.

The performance structure is the **L1 K slab held separate from the cube's
`BASE_K`**. A GM->L1 copy of a `(BASE_M, K_L1)` fp4 tile moves `K_L1/2`
contiguous bytes per row, so tying the slab to `BASE_K=256` made every row copy
128 bytes, a quarter of a 512-byte line. Widening the slab to 512 while the
cube still consumes 256 from an inner sub-tile loop moves identical bytes and
was worth 1.35–1.57x.

L1 is 512 KB, and that bounds the slab. Two `K_L1=512` sets fit in 272 KB; two
`K_L1=1024` sets would need 544 KB, and `-DMXMM_K_L1=1024` is refused at
**compile** time rather than faulting on device:

```
error: static assertion failed due to requirement
'L1_SETS * kSlabBytes + kScaleBytes <= L1_BYTES': The data sets plus the
whole-K scale pair must fit L1; reduce L1_SETS or K.
```

`-DMXMM_K_L1=1024 -DMXMM_L1_SETS=1` does build, at the cost of the load/extract
overlap.

Tunable through `-D`: `MXMM_BASE_M`, `MXMM_BASE_N`, `MXMM_K_L1`,
`MXMM_L1_SETS` (1 or 2), `MXMM_SWIZZLE`, the build shape `MXMM_TEST_K` /
`MXMM_TEST_N`, and `MXMM_TARGET_BLOCKS`, which `jit_util` sets from the
device's `cube_core_num` and which is part of the build cache key. Every one produces correct output; the timing-only ablation
switches used to attribute the bottleneck have been removed.

The derived sizes and their `static_assert`s live in one
`TileShape<M_MAX, K, N, TILE_M, TILE_N>` struct, so an unsupported combination
fails to instantiate rather than faulting on device. The launcher instantiates
it three times, once per output tile.

`MXMM_TINY_M` is gone. Its value was `M_ALIGN`, the 16-row floor TMATMUL_MX
imposes, so it could not go lower and above it was simply another tile size.

### One block per core, and a tile rule that knows it

`cube_core_num` is **32** on an Ascend950PR_9589, and two constants assumed
64: the launcher capped `block_dim` at 64, so every core ran two blocks, and
`TARGET_BLOCKS` — the count `pickMTile` wants filled before it will use the
256-row tile — was 64, so the tall tile was withheld from shapes that already
filled the machine.

Three arms, interleaved with a cold L2, medians of three processes:

| K=N, M | 64 blocks, target 64 | 32 blocks, target 64 | 32 blocks, target 32 | tile, last arm |
|---|--:|--:|--:|---|
| 2048, 1024 | 1.00x | 1.03x | **1.11x** | 128x256 -> 256x256 |
| 4096, 512 | 1.00x | 1.03x | **1.11x** | 128x256 -> 256x256 |
| 8192, 256 | 1.00x | 1.01x | **1.06x** | 128x256 -> 256x256 |
| 4096, 1024 | 1.00x | 1.05x | 1.05x | 256x256 throughout |
| 8192, 512 | 1.00x | 1.03x | 1.03x | 256x256 throughout |
| 8192, 8192 | 1.00x | 1.00x | 0.99x | 256x256 throughout |

Oversubscribing by 2:1 cost 1-5%, and the tile threshold cost a further 5-6%
in the band where it changed the branch. Both are fixed by reading the core
count off the device. Note the large-M control is flat, so this is not the
large-M decay below.

## What would close the gap

This kernel is at 88–95% of the cube ceiling where the vendor is at it, and the
remaining distance is known rather than mysterious:

* **`K_L1=1024`, for full 512-byte bursts — measured, and it does not pay.**
  The wider slab is real: inside a configuration narrow enough to hold it, it
  is worth 1.25x at K=N=8192 (555.9 us against 693.5). But two `K_L1=1024`
  sets at the 256x256 output tile need 544 KB against a 512 KB L1, so
  something has to give, and everything that gives costs more than 1.25x.
  Dropping the 256-wide N tile costs 1.35x on its own, which nets out below
  the default at every shape tried:

  | M, K=N | default, `K_L1`=512 N=256 | `K_L1`=1024 N=128 | `K_L1`=512 N=128 |
  |---|--:|--:|--:|
  | 4096, 4096 | 122.8 us | 164.2 (0.75x) | 166.1 (0.74x) |
  | 4096, 8192 | 477.3 us | 555.9 (0.86x) | 693.5 (0.69x) |
  | 16384, 8192 | 1781.7 us | 2127.3 (0.84x) | 2623.0 (0.68x) |

  So the lever is not the slab width itself but **L1 capacity**: reach
  `K_L1=1024` while keeping a 256x256 tile, by single-buffering one operand or
  otherwise cutting a set, and the 1.25x is available. Note also that the slab
  gain shows only at K=N=8192 and is 1.01x at 4096, so at 4096 the kernel is
  bound by something else.
* **Double-buffering the accumulator**, so the cube does not idle while a tile
  drains — available at the 128-row tile and impossible at the 256-row one.
  A5 L0C is 256 KB (`PTO_L0C_SIZE_BYTES` under `PTO_NPU_ARCH_A5`), and PTO
  enforces it directly:

  ```
  constexpr size_t accBytes = TileRes::Rows * TileRes::Cols * sizeof(CType);
  static_assert(accBytes <= PTO_L0C_SIZE_BYTES,
                "TMatmulMX:accumulator (Rows*Cols*sizeof(out)) exceeds L0C capacity.");
  ```

  The BIG tile's accumulator is 256x256 fp32 = 256 KB, which is the entire
  L0C, so a second one cannot exist there. The SMALL tile's is 128x256 fp32 =
  128 KB, where two fit exactly. An `Acc<float, 256, 512>` needs 512 KB and is
  refused at compile time by that assert.
* **The large-M column.** The ratio decays monotonically with M at every width,
  from 1.87x to 1.29x even at K=N=512, which points at the tile-selection rule
  rather than at the inner loop.

A third L1 set is not one of them, and is no longer in the source. It built to
a genuinely different object and measured 1.005x, 0.999x and 0.989x at
(M=4096, K=N=4096), (4096, 8192) and (1024, 2048) — 1.00x against spreads of
2–23%. Deeper buffering cannot help a stage bound by per-descriptor issue cost
rather than by bandwidth, so the set was removed and `MXMM_L1_SETS` now accepts
1 or 2.
