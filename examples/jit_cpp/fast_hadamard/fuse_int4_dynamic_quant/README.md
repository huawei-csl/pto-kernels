## Fused Dynamic Int4 Quant

This directory contains the packed dynamic int4 variant:

- `fast_hadamard_dynamic_quant.cpp`: fused blockwise Hadamard + dynamic packed int4 quantize
- `traffic_copy.cpp`: single-launch byte-copy kernel for a traffic-matched baseline
- `jit_util_hadamard_dynamic_quant.py`: JIT wrapper for fused Hadamard + dynamic int4 quantize
- `jit_util_traffic_copy.py`: JIT wrapper for the traffic-matched copy baseline
- `int4_cvt.hpp`: packed int4 conversion helper
- `bench_hadamard_dynamic_quant.py`: benchmark dynamic fused int4 against static fused and a single-launch traffic-matched copy baseline
- `test_hadamard_dynamic_quant.py`: correctness tests for fused Hadamard + dynamic int4 quantize

Outputs are packed into `torch.int8` tensors with shape `[batch, N // 2]`. Each
byte stores two signed int4 values. Row scales are written to a separate
`torch.float32` tensor with shape `[batch]`.

Shared helpers remain in the parent `fast_hadamard/` directory:

- `bench_common.py`
- `jit_util_common.py`
- `benchmark.py`
- `test.py`

### Common Commands

From `examples/jit_cpp/fast_hadamard`:

```bash
# correctness
python test.py --target hadamard_dynamic_quant_int4 -v --npu "npu:0"

# benchmark
python benchmark.py --target hadamard_dynamic_quant_int4 --npu 0

# plot
python plot.py --target hadamard_dynamic_quant_int4
```

If you restrict visibility to a single Ascend device with
`ASCEND_RT_VISIBLE_DEVICES`, use logical device `--npu 0` inside that shell.

### Current Limitation

The current dynamic reduction path uses additional UB scratch on top of the
Hadamard and int4-pack buffers. On the currently validated hardware
(`Ascend 910B4`, `dav-c220-vec`), the default benchmark grid is stable for:

- `N=128, 256, 512, 1024, 2048, 4096, 8192, 16384`
- batches `1, 5, 8, 10, 16, 20, 32, 40, 64`
- `hadamard_n=128`

The first observed failing points are:

- `batch=241` for `N=128`
- `batch=128` for `N>=256`

The failing cases report:

- `VEC instruction error: the ub address out of bounds`
