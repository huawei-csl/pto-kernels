# Rough GDN backward baseline (v0)

This is a tested correctness baseline for packed B=1, D=C=128, before
sequence-wide stage fusion. It is not a fused backward megakernel and is not
yet connected to the forward megakernel through an autograd operator.

- `backward.py`: validation, six output allocations, one workspace, one enqueue.
- `launch.cpp`: native chunk loops and backward equations.
- `kernel.cpp`: PTO device primitives; no Torch tensor arithmetic.
- `binding.py`: separately cached device and native builds.

Backward recomputes from the inputs and retains only chunk boundary states.
One 1,024-token sequence currently takes **934 device launches**: 2 setup,
147 forward-state replay, 784 reverse-chunk processing, and 1 final write.
The next architecture target is roughly seven sequence-wide stages, with
forward/reverse state traversals inside device kernels.

## Preserved comparison baseline

[Raw measurements and source hashes](../../../tests/baselines/gdn_mega_bwd_rough_v0.json)
are fixed historical results; save future measurements separately.

| Tokens | Native backward wall time (ms) |
| --- | --- |
| 128 | 5.005 |
| 512 | 19.561 |
| 2048 | 77.748 |

Ascend 910B2, H=8, Hk=2, D=128; median of five calls after one warmup.
Includes allocation, recomputation and final synchronization; excludes build
and the model's forward pass. Enqueue alone measured approximately 0.35 ms.
**57 backward/reference tests passed**, with fixed `atol=2e-5, rtol=2e-4`.
See [development notes](../../../memory_bwd.md) for commands and limitations.

After sourcing CANN, run from the repository root:

```bash
OMP_NUM_THREADS=4 NPU_DEVICE=npu:0 python -m pytest tests/test_gdn_mega_bwd.py tests/test_gdn_mega_bwd_reference.py -q
OMP_NUM_THREADS=4 PYTHONPATH=. python tests/benchmark_gdn_mega_bwd.py
```
