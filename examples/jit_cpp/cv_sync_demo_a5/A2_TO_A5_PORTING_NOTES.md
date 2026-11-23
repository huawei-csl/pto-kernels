# Porting A2/A3 Cube-Vector Kernels to A5

This note summarizes practical lessons from porting the old DAV_2201
Cube-Vector sync demos to A5 / DAV_3510.

## 1. Change the Mental Model

On A2/A3, Cube-Vector exchange commonly used GM as a rendezvous buffer:

```cpp
// A2/A3 style: Cube -> GM workspace -> Vec
TSTORE(workspace_tile, acc_l0c);
pipe_barrier(PIPE_ALL);
ffts_cross_core_sync(PIPE_FIX, ready_flag);

wait_flag_dev(ready_flag);
TLOAD(vec_ub, workspace_tile);
```

On A5, prefer the direct local path:

```cpp
// A5 style: Cube L0C -> Vec UB
TMOV<VecTileFloat, AccTile, AccToVecMode::DualModeSplitM>(vec_ub, acc_l0c);
pipe_barrier(PIPE_ALL);
set_intra_block(PIPE_FIX, ready_flag);
set_intra_block(PIPE_FIX, ready_flag + 16);
```

For Vector-to-Cube, use `TINSERT`, which lowers to the fast
`copy_ubuf_to_cbuf` instruction:

```cpp
// A5 style: Vec UB -> Cube L1
TMOV(vec_nz, vec_nd);  // Convert ND Vec tile to NZ layout for Cube.
TINSERT(l1_tile, vec_nz, vid * HALF_TILE, 0);
pipe_barrier(PIPE_ALL);
set_intra_block(PIPE_MTE3, ready_flag);
```

## 2. Use A5 Compile Guards

Old kernels often use DAV_2201 guards:

```cpp
#if defined(__DAV_C220_CUBE__)
// Cube code
#endif

#if defined(__DAV_C220_VEC__)
// Vec code
#endif
```

Use A5-generic guards instead:

```cpp
#if defined(__DAV_CUBE__)
// Cube code
#endif

#if defined(__DAV_VEC__)
// Vec code
#endif
```

Compile mixed Cube/Vec kernels with:

```bash
--cce-aicore-arch=dav-c310
```

## 3. Count the Right Bytes

The old GM round-trip path transferred two copies per iteration:

```text
Vec -> GM workspace
GM workspace -> Cube
```

The A5 direct path transfers one physical copy:

```text
Vec UB -> Cube L1
```

For performance reporting, be explicit:

```python
actual_a5_bytes = cores * rows * cols * element_size * iters
old_formula_bytes = 2 * cores * rows * cols * sizeof_fp16 * iters
```

The first number measures the actual A5 data movement. The second is useful only
when comparing against old reports that counted GM write plus GM read.

## 4. Convert Vec ND to NZ Before Feeding Cube

Cube-side matrix tiles expect an NZ-style layout. A common V2C mistake is
inserting a row-major Vec tile directly into L1 and then using it as a Cube
operand.

Incorrect:

```cpp
TADD(sum_nd, a_ub, b_ub);
TINSERT(l1_tile, sum_nd, vid * HALF_TILE, 0);  // Wrong layout for Cube.
```

Correct:

```cpp
TADD(sum_nd, a_ub, b_ub);
SetFlag<PIPE_V, PIPE_MTE3>(0);
WaitFlag<PIPE_V, PIPE_MTE3>(0);

TMOV(sum_nz, sum_nd);
SetFlag<PIPE_V, PIPE_MTE3>(0);
WaitFlag<PIPE_V, PIPE_MTE3>(0);

TINSERT(l1_tile, sum_nz, vid * HALF_TILE, 0);
```

## 5. Separate Data Readiness From Slot Lifetime

For persistent V2C kernels, the fast path is not just `TINSERT`. You must also
ensure the Vec side does not overwrite the L1 handoff slot before Cube has
copied it into L0 and no longer needs it.

A conservative single-slot protocol:

```cpp
constexpr uint16_t READY = 10;
constexpr uint16_t FREE = 11;

// Vec producer
if (round > 0) {
    wait_intra_block(PIPE_MTE3, FREE);
    pipe_barrier(PIPE_ALL);
}
TINSERT(l1_tile, vec_nz, vid * HALF_TILE, 0);
pipe_barrier(PIPE_ALL);
set_intra_block(PIPE_MTE3, READY);

// Cube consumer
wait_intra_block(PIPE_MTE1, READY);
wait_intra_block(PIPE_MTE1, READY + 16);  // Wait for both Vec subblocks.
pipe_barrier(PIPE_ALL);
TMOV(l0a, l1_tile);
// After MTE1 has captured L1 into L0, release the slot.
set_intra_block(PIPE_MTE1, FREE);
set_intra_block(PIPE_MTE1, FREE + 16);
```

In practice, also drain the dependent compute/store pipes before reusing local
tiles:

```cpp
SetFlag<PIPE_MTE1, PIPE_M>(0);
WaitFlag<PIPE_MTE1, PIPE_M>(0);

TMATMUL(acc, l0a, l0b);
SetFlag<PIPE_M, PIPE_FIX>(0);
WaitFlag<PIPE_M, PIPE_FIX>(0);

TSTORE(out, acc);
pipe_barrier(PIPE_ALL);
```

## 6. Be Careful With Accumulator-to-Vec Type Conversion

A5 supports direct `L0C -> UB`, but not every type/split combination is valid.
For example, fp32 accumulator to fp16 Vec in dual-destination mode is not
accepted by PTO:

```cpp
// This may fail: fp32 Acc -> fp16 Vec with DualModeSplitM.
TMOV<VecTileHalf, AccTileFloat, AccToVecMode::DualModeSplitM>(dst, acc);
```

Use a supported direct path:

```cpp
// Native supported path: fp32 Acc -> fp32 Vec.
TMOV<VecTileFloat, AccTileFloat, AccToVecMode::DualModeSplitM>(dst, acc);
```

That may require matching the downstream tile and output type to `float`.

## 7. Do Not Benchmark Setup Work as Copy Bandwidth

If the goal is to measure V2C copy bandwidth, keep the timed loop focused on
the direct copy:

```cpp
// Setup once.
TLOAD(a_ub, a_global);
TLOAD(b_ub, b_global);
TADD(sum_nd, a_ub, b_ub);
TMOV(sum_nz, sum_nd);

// Timed loop: only UB -> L1 handoff plus synchronization.
for (int r = 0; r < num_iters; ++r) {
    if (r > 0) {
        wait_intra_block(PIPE_MTE3, FREE);
    }
    TINSERT(l1_tile, sum_nz, vid * HALF_TILE, 0);
    set_intra_block(PIPE_MTE3, READY);
}
```

If every iteration reloads from GM, does vector arithmetic, and converts layout,
the result is no longer a clean measurement of `copy_ubuf_to_cbuf`.

## 8. Validate on Real Hardware Early

Simulator runs are useful for occasional debugging, but the real device catches
the issues that matter for this port:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch_npu_dev
source /usr/local/Ascend/cann-9.0.0/set_env.sh
NPU_DEVICE=npu:0 python3 run_matmul.py
```

Before porting larger kernels, validate the environment with a small A5 smoke
test:

```bash
cd /home/jzhuang/ptoisa-a5-test/tests/torch_sim
NPU_DEVICE=npu:0 python3 -m common.build tadd
NPU_DEVICE=npu:0 python3 tadd/test_tadd.py
```

## 9. Calling Low-Level Direct-Copy Intrinsics

Prefer PTO tile APIs first:

- `TMOV(VecTile, AccTile, ...)` for `L0C -> UB`.
- `TINSERT(MatTile, VecTile, row, col)` for `UB -> L1`.
- `TEXTRACT` / `TMOV` variants for layout conversion.

Use low-level intrinsics directly only when the PTO wrapper cannot express the
shape, split, or synchronization you need. The intrinsics are internal and easier
to misuse.

### UB -> L1: `copy_ubuf_to_cbuf`

This is the instruction behind `TINSERT(Vec -> Mat)` on A5.

```cpp
// Runs on Vec/AIV.
// Units:
//   lenBurst, srcStride, dstStride are in 32-byte blocks.
//   nBurst is the number of bursts.
__ubuf__ half *src = (__ubuf__ half *)__cce_get_tile_ptr(vec_nz);
__cbuf__ half *dst = (__cbuf__ half *)__cce_get_tile_ptr(l1_tile)
                   + row_offset * TILE_SIZE + col_offset;

uint16_t nBurst = 1;
uint16_t lenBurst = (HALF_TILE * TILE_SIZE * sizeof(half)) / 32;
uint16_t srcStride = 0;
uint16_t dstStride = 0;

copy_ubuf_to_cbuf(dst, src, 0, nBurst, lenBurst, srcStride, dstStride);
pipe_barrier(PIPE_ALL);
set_intra_block(PIPE_MTE3, READY);
```

For a row-split V2C tile, either:

- call `copy_ubuf_to_cbuf` with `dst` offset to the subblock's rows, or
- use `TINSERT(l1_tile, vec_nz, vid * HALF_TILE, 0)`, which computes the same
  offset and burst parameters for you.

Important: convert row-major Vec data to the NZ layout expected by Cube before
copying into an L1 matrix tile:

```cpp
TMOV(vec_nz, vec_nd);
copy_ubuf_to_cbuf(... vec_nz ...);
```

### L1 -> UB: `copy_cbuf_to_ubuf`

This is useful for diagnostics or direct Cube-to-Vec helper copies from L1.

```cpp
// Runs on Cube/AIC.
// subBlockId selects the receiving Vec subblock in split cases.
__ubuf__ half *dst = (__ubuf__ half *)__cce_get_tile_ptr(vec_ub);
__cbuf__ half *src = (__cbuf__ half *)__cce_get_tile_ptr(l1_tile);

bool subBlockId = static_cast<bool>(get_subblockid());
uint16_t nBurst = 1;
uint16_t lenBurst = (HALF_TILE * TILE_SIZE * sizeof(half)) / 32;
uint16_t srcStride = 0;
uint16_t dstStride = 0;

copy_cbuf_to_ubuf(dst, src, subBlockId, nBurst, lenBurst, srcStride, dstStride);
pipe_barrier(PIPE_ALL);
```

Most production kernels should prefer higher-level PTO movement unless the
direct L1-to-UB copy is exactly what you need.

### L0C -> UB: `copy_matrix_cc_to_ub`

This is the instruction behind accumulator-to-Vec `TMOV` on A5.

```cpp
// Runs on Cube/AIC.
__ubuf__ float *dst = (__ubuf__ float *)__cce_get_tile_ptr(vec_float);
__cc__ float *src = (__cc__ float *)__cce_get_tile_ptr(acc_l0c);

uint16_t validCol = TILE_SIZE;
uint16_t validRow = TILE_SIZE;
uint32_t dstStride = TILE_SIZE;  // Element stride in UB layout.
uint16_t srcStride = TILE_SIZE;  // Accumulator source stride, usually aligned.

uint8_t dualDstCtl = 1;          // Split M across two Vec subblocks.
bool subBlockId = static_cast<bool>(get_subblockid());
uint8_t unitFlagCtl = 0;

copy_matrix_cc_to_ub(
    dst, src,
    0,                 // sid
    validCol,
    validRow,
    dstStride,
    srcStride,
    dualDstCtl,
    subBlockId,
    0,                 // clip/relu pre control
    unitFlagCtl,
    QuantMode_t::NoQuant,
    ReluPreMode::NoRelu,
    true,              // split enable
    true,              // NZ -> ND output when writing row-major Vec
    0,
    0,
    false,
    false,
    0,
    false,
    false,
    false,
    false,
    false,
    false);
pipe_barrier(PIPE_ALL);
```

The exact control bits are easy to get wrong. For accumulator-to-Vec copies,
prefer the PTO wrapper unless you need direct control:

```cpp
TMOV<VecTileFloat, AccTileFloat, AccToVecMode::DualModeSplitM>(vec, acc);
```

Known restriction from this port: fp32 accumulator to fp16 Vec in dual split mode
was rejected by PTO's type checks. fp32 accumulator to fp32 Vec worked.

## Checklist

- Compile with `dav-c310` for mixed Cube/Vec A5 kernels.
- Replace GM workspace exchange with `TMOV` for C2V and `TINSERT` for V2C.
- Remember that `TINSERT(Vec -> Mat)` uses `copy_ubuf_to_cbuf`.
- Use low-level intrinsics directly only when the PTO tile wrapper is too
  restrictive for the needed shape/layout.
- Convert Vec ND tiles to NZ before Cube consumes them.
- Use `set_intra_block` / `wait_intra_block` with separate ready/free flags.
- Wait for both Vec subblocks on Cube by also waiting for `flag + 16`.
- Use `pipe_barrier(PIPE_ALL)` around handoff ownership changes.
- Benchmark actual direct-copy bytes separately from old GM round-trip formulas.
- Validate multi-wave correctness, not only one-wave smoke tests.

