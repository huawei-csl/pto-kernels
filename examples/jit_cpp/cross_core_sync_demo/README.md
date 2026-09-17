Demonstrate different API abstractions for Cube-Vector data exchange and synchronization

There are currently 4 API sets that can express cross-core data passing:
1. `ffts_cross_core_sync` & `wait_flag_dev`
2. `TSYNC`
3. `TPUSH` & `TPOP`
4. `TPUSH` & `TPOP` & `TFREE` & `TALLOC`

Purpose of this demo directory: Use *clear, minimum code* to demonstrate the *syntax and performance* differences between those API styles.

- [stream_c2v_v2c](./stream_c2v_v2c)
- [matmul_add](./matmul_add)
- [linear_attn](./linear_attn)

## Known PTO API Issues

See **[PTO_API_BUGS.md](./PTO_API_BUGS.md)** for confirmed bugs and workarounds:

- **Bug 1**: `TPipe` TileData TPUSH/TPOP with `TILE_UP_DOWN` and 2 Vec sub-blocks — `tileIndex` shared counter causes slot desync for `num_rounds ≥ 2` (`FIFO_DEPTH=2`). Confirmed present in latest `pto-isa-master` (as of 2026-05-12). Workaround: use `FIFO_DEPTH=1`, `gm_pipe`, or `raw_flag`.
- **Bug 2**: FFTS flag collision when two pipes share the same `FlagID` (e.g., `TPipe<0, DIR_C2V>` and `TPipe<0, DIR_V2C>` both use flags 0/1).
- **Bug 3**: `TSTORE(c_global, c_l0)` FIX-pipe DMA in-flight at kernel exit conflicts with next-call `TMATMUL` in benchmark loops → `L0C read/write conflict`.
