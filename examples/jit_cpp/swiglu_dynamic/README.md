# SwiGLU Dynamic

This directory contains a standalone PTO `SwiGLU` kernel and a small validation
workflow against `torch_npu.npu_swiglu`.

Outputs:
- benchmark CSVs: `outputs/csv/`
- plots: `outputs/plots/`
- JIT shared objects: `outputs/so/`

The benchmark reports:
- runtime in microseconds
- effective TFLOPS
- PTO speedup over `torch_npu.npu_swiglu`

The TFLOPS estimate uses `6` effective floating-point operations per output
element:
- negate
- exp
- add
- divide
- multiply for SiLU
- multiply by the gate

Suggested workflow:

```bash
pytest test_swiglu.py -q --npu npu:0
python bench_swiglu.py --npu 0
python plot_swiglu.py
```
