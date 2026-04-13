# SwiGLU Dynamic

This directory contains a standalone PTO `SwiGLU` kernel and a small validation
workflow against `torch_npu.npu_swiglu`.

Outputs:
- benchmark CSVs: `outputs/csv/`
- plots: `outputs/plots/`
- JIT shared objects: `outputs/so/`

The benchmark reports:
- runtime in microseconds
- effective elementwise TOPS
- PTO speedup over `torch_npu.npu_swiglu`

The TOPS estimate is not a real FLOPS metric: `exp` and `div` are counted as
one effective elementwise operation each so that PTO and `torch_npu` can be
compared on the same scale. The PTO kernel executes `5` vector operations per
output element:
- negate
- exp
- add
- divide
- multiply by the gate

Suggested workflow:

```bash
pytest test_swiglu.py -q --npu npu:0
python bench_swiglu.py --npu 0
python plot_swiglu.py
```
