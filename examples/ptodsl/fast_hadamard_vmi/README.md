# fast_hadamard_vmi — parametric-N fast Walsh–Hadamard (ptodsl / VMI)

A memory-bound fast Walsh–Hadamard transform (WHT) written in **ptodsl (VMI)**
raw `.pto`. It computes, in place, the unnormalized `y = x @ H` in fp16, where
`H` is the natural-order (Sylvester) ±1 Hadamard matrix of order `N`.

The kernel, **`fulltp`**, is **parametric on the transform width** `N ∈ {32, 64,
128, 256, 512, 1024, 2048}`. Each row of the `(batch, N)` fp16 input is
transformed by a `log2(N)`-stage deinterleave butterfly. It is multi-core (the
batch is split into `grid=G` contiguous row bands) with a per-block 4-buffer /
32 KB UB software pipeline (prefetch depth 2); the per-chunk tile loop is unrolled
at generate-time so every UB offset is an immediate.

`fulltp` is **flexible and fast**: it uses contiguous per-core DMA (which sustains
higher HBM bandwidth than a grid-stride layout on device) yet accepts **any
per-band chunk count** — `batch % (G·CR) == 0`, `nchunk ≥ 1`, where
`CR = 16384/N` rows/chunk — by running the whole 4-chunk quads through the
pipeline and the remaining `nchunk % 4` chunks through a partial-quad **tail** (see
Flexibility). On device it **meets or beats the CCE reference at every N** for both
aligned and remainder batches.

This is an `examples/ptodsl` example (the `examples/jit_cpp` ones are C++/bisheng).
Its distinguishing feature: **the same `--check` code path runs both under
`cannsim` and on a real A5 device** — correctness is proved through pyACL
(torch-free), so no device is required to validate it.

## Files

| File | Purpose |
| --- | --- |
| `run_hadamard_vmi.py`      | Runner. `--check` (pyACL correctness, cannsim + device), `--nsweep` (all N), `--bench` (device-only bandwidth). |
| `fulltp_n_mlir.py`         | `make_fulltp(N)` — the `@pto.jit` wrapper selecting the per-N `.pto`. |
| `gen_had_fulltp.py`        | Parametric `.pto` generator: `gen_had_fulltp.py [N] [BATCH_HINT] [OUT.pto]`. |
| `fast_hadamard_vmi_fulltp_n{32,64,128,256,512,1024,2048}.pto` | Committed per-N kernels (`@fast_hadamard_vmi_fulltp_n{N}`). |
| `golden.py`                | Numpy reference `ref_hadamard` (`x @ H`). |
| `test/check_cannsim.py`    | cannsim entry: sets argv and calls the runner's `--check`. |
| `scripts/`                 | Minimal cannsim harness: `run_sim.sh`, `run_sim_entry.sh`, `camodel_entry.sh`, `env.sh`, `cannsim_metrics.py`. |

The `.pto` sit beside `fulltp_n_mlir.py` because `@pto.jit(source=...)` resolves the
relative path against the declaring module's directory. Regenerate any of them with,
e.g., `python3 gen_had_fulltp.py 512` (writes `fast_hadamard_vmi_fulltp_n512.pto`).

## Run correctness on cannsim

```bash
export PTOAS_ROOT=<your ptoas checkout> PTOAS_HOST_TARGET_CPU=tsv110 \
       PTOAS_ENV_SKIP_SMOKE_TEST=1
source "${PTOAS_ROOT}/scripts/ptoas_env.sh"

cd examples/ptodsl/fast_hadamard_vmi
# N and grid via env; batch defaults to the smallest valid = grid*65536/N
HAD_N=512 HAD_GRID=1 bash scripts/run_sim.sh test/check_cannsim.py sim_outputs/n512
```

Expect `[check] PASS: N=512 ... rel_err≈1e-3`. Verified under `cannsim -s
Ascend950` across N=32…2048 (rel_err ~1e-3).

## Run on device

Same runner, no cannsim wrapper (needs CANN + pyACL; `--bench` also needs `torch`
+ `torch_npu` and a real A5):

```bash
python run_hadamard_vmi.py --check --n 512            # pyACL correctness
python run_hadamard_vmi.py --nsweep                   # correctness over all N
python run_hadamard_vmi.py --bench --n 512 --grid 64  # GM bandwidth vs a torch D2D copy floor
```

`--check` is torch-free and identical to the path cannsim runs. `--bench` reports
effective GM bandwidth (`4*batch*N` bytes/launch) next to the device-to-device
copy ceiling.

## Supported N and performance

- **N = 32 … 2048** are correctness-verified (rel_err ~1e-3). N=32 uses ROT=3
  `pto.vdintlv` window rotations and works (there is no net-rotation cap). **N=16
  (ROT=4) is untested** and left out; N > 2048 needs ≥8 chunks/row and is rejected.

- **Flexibility (the tail):** the per-band chunk count is `nchunk = batch/(G·CR)`
  with `CR = 16384/N` rows/chunk. The 4-buffer pipeline consumes chunks in quads,
  so `fulltp` runs `nquad = nchunk // 4` quads through it (guarded by
  `scf.if(nquad>0)`) and the remaining `rem = nchunk % 4` chunks through a
  single-buffer **tail** — giving `batch % (G·CR) == 0`, any `nchunk ≥ 1`. A naive
  tail (a `PIPE_ALL` drain before it) loses ~4 chunks of pipeline overlap and
  trails CCE at small remainder batches; instead the tail uses an **overlapped ring
  free-flag handshake**: the buffer frees are unconditional (so `nchunk<4` also
  starts clean), the epilogue re-frees `buf0`, and the tail does `wait free[0] →
  load → compute → store → set free[0]`, so its load overlaps the epilogue on the
  MTE2 pipe. This holds CCE cycle-parity on the remainder (cannsim rem=1 cycles:
  `fulltp/cce ≈ 0.995` at nchunk=5/17/33; a serial tail was 1.03–1.20).

- **Compute (cannsim cycles):** VMI reaches **cycle parity with the CCE reference
  across all N** (VMI/CCE ≈ 1.0 for N=32…2048), including the remainder chunks.

- **Device (real A5), `fulltp` vs CCE** (950DT, grid=64, ~270 MB working set,
  `had_GBs = 4·batch·N/time`): the transform is memory-bound and contiguous per-core
  DMA sustains higher HBM bandwidth than CCE's grid-stride, so `fulltp` **beats CCE
  at every N for both aligned and flexible batches**, +4…+13% (sole exception
  N=2048 flexible at −0.2%, parity). Flexible = `nchunk=65` (rem=1), a batch an
  aligned-only kernel would reject:

  | N | aligned fulltp/cce | flexible (rem=1) fulltp/cce |
  | --- | --- | --- |
  | 32 | 3268/2897 **+12.8%** | 3187/2985 **+6.8%** |
  | 64 | 3137/2937 **+6.8%** | 3192/3029 **+5.4%** |
  | 128 | 3184/2938 **+8.4%** | 3185/3035 **+4.9%** |
  | 256 | 3203/2962 **+8.1%** | 3114/2895 **+7.6%** |
  | 512 | 3024/2878 **+5.1%** | 3017/2876 **+4.9%** |
  | 1024 | 2941/2785 **+5.6%** | 2924/2804 **+4.3%** |
  | 2048 | 2733/2703 **+1.1%** | 2695/2699 −0.2% |

  The margin is largest at small N and narrows toward N=2048, where all approach
  the device-to-device copy floor.
