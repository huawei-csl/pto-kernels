# Synchronization And Architecture Notes

## A2A3 vs A5

| Topic | A2A3 / Ascend910B / DAV_2201 | A5 / Ascend950 / DAV_3510 |
| --- | --- | --- |
| PTO test dir | `tests/npu/a2a3/src/st/testcase` | `tests/npu/a5/src/st/testcase` |
| Include dir | `include/pto/npu/a2a3` | `include/pto/npu/a5` |
| Vector flag | `dav-c220-vec` | `dav-c310-vec` |
| Cube flag | `dav-c220-cube` or `--cce-soc-core-type=CubeCore` | `dav-c310-cube` |
| Mix flag | `dav-c220` | `dav-c310` |
| Memory macro | `-DMEMORY_BASE` | `-DREGISTER_BASE` |
| Typical core ratio | Cube:Vector = 1:2 | Cube:Vector = 1:2 |
| Important buffer delta | UB 192 KiB, L0C 128 KiB | UB 248 KiB, L0C 256 KiB |

Use runtime APIs for exact core counts and memory sizes. INI files are useful clues but not the final source of truth.

## Mix Kernels

Use "mix kernel" for kernels where Cube and Vector sides cooperate in one launch.

### A2A3 Pattern

Local files: `dynamic_multi_core/a2a3/matmul_add.cpp` and `dynamic_multi_core/a2a3/add_matmul.cpp`.

- `matmul_add_c2v`: Cube computes `A @ B`, stores to workspace, Vector adds `D`.
- `add_matmul_v2c`: Vector computes `A + B`, stores to workspace, Cube computes matmul.
- `raw_flag` is the clearest reference for manual FFTS.
- `gm_pipe` is the safer multi-round `TPipe` variant when newer PTO-ISA headers are available.

Known pitfalls:

- TileData `TPUSH`/`TPOP` with two Vec subblocks can desynchronize FIFO slot indices in multi-round kernels. Use `FIFO_DEPTH=1`, `gm_pipe`, or raw FFTS flags.
- Do not reuse conflicting FFTS `FlagID` values for kernels called sequentially in one process.
- Add `pipe_barrier(PIPE_ALL)` after the last `TSTORE` from L0C to avoid next-call `TMATMUL` read/write conflicts.

### A5 Pattern

Local files: `dynamic_multi_core/a5/matmul_add.cpp` and `dynamic_multi_core/a5/add_matmul.cpp`.

A5 has direct local Cube/Vector paths:

- Cube to Vector: `TMOV<VecTileFloat, AccTile, AccToVecMode::DualModeSplitM>` lowers to `copy_matrix_cc_to_ub`.
- Vector to Cube: convert Vec ND to NZ, then `TINSERT` into L1; this lowers to `copy_ubuf_to_cbuf`.
- Use `set_intra_block` / `wait_intra_block` and wait for both Vec subblocks (`flag` and `flag + 16` where required by the helper).

Porting checklist:

- Replace A2A3 guards such as `__DAV_C220_CUBE__` with A5-generic `__DAV_CUBE__` and `__DAV_VEC__`.
- Compile mixed kernels with `--cce-aicore-arch=dav-c310`.
- Convert Vec row-major data to NZ layout before Cube consumes it.
- Separate data-ready flags from slot-free flags in persistent V2C kernels.
- Benchmark hot copy loops separately from setup work.

## Static vs Dynamic Launch

Static single-core samples usually launch one block and one testcase shape:

```cpp
runTAdd<...><<<1, nullptr, stream>>>(out, src0, src1);
```

Dynamic multi-core samples usually launch physical core count and loop over runtime work:

```cpp
kernel<<<block_dim, nullptr, stream>>>(...);
// inside kernel: for each tile assigned to get_block_idx()
```

Use the static form to learn instruction syntax. Use the dynamic form for real workload testing.
