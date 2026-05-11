Demonstrate different API abstractions for Cube-Vector data exchange and synchronization

There are currently 4 API sets that can express cross-core data passing:
- `ffts_cross_core_sync` & `wait_flag_dev`
- `TSYNC`
- `TPUSH` & `TPOP`
- `TPUSH` & `TPOP` & `TFREE` & `TALLOC`

Purpose of this demo directory: Use *clear, minimum code* to demonstrate the *syntax and performance* differences between those API styles.

- [stream_c2v_v2c](./stream_c2v_v2c)
- [matmul_add](./matmul_add)
- [linear_attn](./linear_attn)
