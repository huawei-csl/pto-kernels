## Fused Int4 Quant

This directory contains the packed int4 variants:

- `quantize.cpp`: fp16 -> packed int4 quantize using `vconv_f162s4`
- `fast_hadamard_quant.cpp`: fused Hadamard + packed int4 quantize
- `jit_util_quantize.py`: JIT wrapper for standalone int4 quantize
- `jit_util_hadamard_quant.py`: JIT wrapper for fused Hadamard + int4 quantize
- `bench_quantize.py`: benchmark standalone int4 quantize against a torch packed-int4 reference
- `bench_hadamard_quant.py`: benchmark fused int4 against separate PTO kernels and a torch packed-int4 unfused reference
- `test_quantize.py`: correctness tests for standalone int4 quantize
- `test_hadamard_quant.py`: correctness tests for fused Hadamard + int4 quantize

Outputs are packed into `torch.int8` tensors with shape `[batch, N // 2]`. Each
byte stores two signed int4 values.

Shared helpers remain in the parent `fast_hadamard/` directory:

- `bench_common.py`
- `plot_common.py`
- `jit_util_common.py`

### Common Commands

From `examples/jit_cpp/fast_hadamard`:

```bash
# correctness
python test.py --target quantize_int4 -v --npu "npu:0"
python test.py --target hadamard_quant_int4 -v --npu "npu:0"

# benchmark
python benchmark.py --target quantize_int4 --npu 0
python benchmark.py --target hadamard_quant_int4 --npu 0

# plot
python plot.py --target quantize_int4
python plot.py --target hadamard_quant_int4
```
