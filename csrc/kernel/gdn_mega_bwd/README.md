# Sequence-wide GDN backward

Packed B=1, D=C=128, FP16/BF16/FP32 inputs and FP32 gradients. The backward
recomputes from its inputs; no saved forward intermediates are required.
Python validates, allocates six outputs and one workspace, then enqueues once.

## Seven device stages

| Stage | Work | Scheduling |
| --- | --- | --- |
| 0 | Pack inputs; prefix gates; Gram products; inverse; W/U | Independent chunk/head tasks |
| 1 | Replay incoming states and corrected values Z | Forward chunk loop inside each sequence/head task |
| 2 | Local value-update gradients and local state-gradient contribution | Independent chunk/head tasks |
| 3 | Complete value-update gradients and propagate state adjoints; write dh0 | Reverse chunk loop inside each sequence/head task |
| 4 | dQ, partial dK, dW, and gate-energy contributions | Independent chunk/head tasks |
| 5 | WY/inverse derivatives: dV, dBeta, remaining dK | Independent chunk/head tasks |
| 6 | Gate-gradient reverse scans, shared-head reduction, output assembly | Independent chunk tasks |

The host launches these seven stages once per group of up to 64 packed
sequences. Sequence descriptors travel inline in kernel arguments. Larger
batches use additional groups of seven launches; launch count does not grow
with token/chunk count. There are no host chunk loops or device-wide barriers
inside a stage. Kernel ordering supplies the dependencies between stages.

Each mixed cube/vector block owns its scratch. Both vector subcores process
disjoint rows where possible. Weighted triangular preparation and several
multiply/reduce chains are fused so their products stay on chip. Other matrix
temporaries still use GM inside stages; further fusion and tiling are possible.

All dense matrix products use Cube instructions. FP16/BF16 q/k/v use native
FP32 Cube accumulation. When any of q/k/v is FP32, products use four K=32
Cube partials with compensated addition of the results. That accuracy path
also sums gate intervals directly, reconstructs dQ within 16-token blocks
inside each 128-token chunk, and computes dBeta as `dot(dUpdate, Z)/beta`
where beta is nonzero (the general WY derivative handles beta=0).
No matrix product falls back to vector dot loops. All of this remains within
the same seven device stages; Python only validates and allocates.

## Files

- `backward.py`: validation, allocation and one native enqueue.
- `binding.py`: C ABI and separately cached device/native builds.
- `launch.cpp`: packed-sequence metadata and Torch queue integration.
- `kernel.cpp`: seven stage bodies and their device task loops.
- `device_ops.h`: cube/vector cooperation and scratch helpers.
- `primitives.h`: FP32 Cube matrix products and vector elementwise/reduction operations.
- `layout.h`: stage buffer lifetimes and metadata ABI.

## Comparison with rough baseline

Commit **150fed0** preserves the original working implementation. Its
[raw measurements and source hashes](../../../tests/baselines/gdn_mega_bwd_rough_v0.json)
remain unchanged. One 1,024-token sequence previously required 934 device
launches; a hardware trace now shows seven.

Archived comparison before replacing vector matrix products ([raw samples and trace summary](../../../tests/baselines/gdn_mega_bwd_staged_v1.json)):

| Tokens | Rough native baseline (ms) | Seven stages (ms) |
| --- | --- | --- |
| 128 | 5.005 | 7.279 |
| 512 | 19.561 | 14.033 |
| 2048 | 77.748 | 41.082 |

Ascend 910B2, H=8, Hk=2; median of five calls after one warmup. Includes
allocation, recomputation and final synchronization, excluding compilation.
That version used compensated vector matrix products in stages 2 and 4 and
passed 66 checks at `atol=2e-5, rtol=2e-4`. These are historical numbers.

Current Cube results ([raw samples, profile and source hashes](../../../tests/baselines/gdn_mega_bwd_cube_v2.json)),
H=16, Hk=2, D=128, FP16 inputs with FP32 gates, same timing method:

| Tokens | Vector products v1 (ms) | Cube products v2 (ms) | Speedup |
| --- | --- | --- | --- |
| 2048 | 75.378 | 5.534 | 13.62x |
| 8192 | 293.984 | 20.481 | 14.35x |

The 8K trace still contains seven launches. Local/output gradients fell from
275.42 ms combined to 2.57 ms; preparation/recomputation now takes 10.98 ms.
These speedups describe the FP16 path; FP32 inputs use the additional accuracy
work described above. **71 tests pass** (61 hardware, 10 CPU reference) in
109.04 seconds with cached builds.

## Accuracy checks

Each current hardware case runs the independent recursive PyTorch reference
twice with identical rounded inputs and upstream gradients. The golden run
computes in FP32 for FP16/BF16 inputs, or FP64 for FP32 inputs. The baseline
computes in the input dtype, distributing beta through the state update and
scaling q before its contraction. Autograd differentiates both recurrences.
For each of dq, dk, dv, dg, dbeta and dh0, the kernel's maximum absolute error
must be at most twice the baseline's maximum absolute error plus `1e-5`. Golden
gradients retain their computation precision, since this API returns FP32
gradients. CPU tests also check that the reordered equations and derivatives
agree in FP64. Only the reference runs forward here; the device backward
recomputes its inputs and does not call the forward megakernel.

Sequence-wide stages use more GM storage: 12 FP32 matrices and seven FP32
vectors per chunk/head, plus 45 MiB of reusable per-block scratch (including
four Cube partial-result slots). At 8K tokens
and 16 value heads the workspace is **816.5 MiB**, versus the rough baseline's
162 MiB. Slots are reused after their last consumer (W→dW, U→Z, Q→dQ,
dOut→dV, state adjoint→dK). This workspace exists during backward only.

The model autograd wrapper and forward-cache integration remain separate work.
See [development notes](../../../memory_bwd.md) for validation and profiling.

After sourcing CANN, run from the repository root:

```bash
OMP_NUM_THREADS=4 NPU_DEVICE=npu:0 python -m pytest tests/test_gdn_mega_bwd.py tests/test_gdn_mega_bwd_reference.py -q
OMP_NUM_THREADS=4 PYTHONPATH=. python tests/benchmark_gdn_mega_bwd.py
OMP_NUM_THREADS=4 PYTHONPATH=. python tests/benchmark_gdn_mega_bwd.py --tokens 1024 --profile build/gdn_mega_bwd/profile/current
```
