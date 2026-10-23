# Production GDN mega kernel

One launch, two global phases, one compiled production schedule:

1. Vector gate prefix scan and layout conversion run concurrently with cube
   KKᵀ/QKᵀ products shared across key heads. One global barrier joins them.
2. A head-persistent pipeline alternates two vector preparation slots while the
   cube performs the inverse, WY products, state recurrence, and output. State
   stays on-chip across chunks. Slot handshakes enforce producer/consumer order.

These are execution phases, not file counts. The inverse still uses bounded
runtime term selection with a full 64-term fallback. Nonzero initial states,
finite decay, tails, and packed sequences remain supported.

| File | Used for |
|---|---|
| `kernel_gdn_mega.cpp` | Two-phase orchestration, kernel entry, runtime head dispatch |
| `common.hpp` | Device synchronization, packed tile I/O, packed sequence indexing |
| `gate_scan.hpp` | Parallel prefix scan and gate/beta layout conversion |
| `shared_grams.hpp` | Shared KKᵀ/QKᵀ; single-wave and pipelined multi-wave execution |
| `inverse.hpp` | Interleaved 64×64 diagonal solves and off-diagonal join |
| `recurrence_pipeline.hpp` | Overlapped WY, state, and output matrix operations |
| `fused_pipeline.hpp` | Vector operand preparation, persistent slots, startup/final state |

Every helper is included by the entry point. There are no legacy mega stages,
probe kernels, timing hooks, early-exit debug modes, or tuning `#ifdef`s.
Compiler/architecture guards remain because cube and vector execute different
instructions. Target validation is Ascend 910B2; the inherited A5 synchronization
path is not validated and still lacks a verified device-wide barrier.

## Caller contract

The public entry point remains `pto_gdn_mega`. Its arguments, shapes, dtypes,
allocation sizes, and in-place output behavior are unchanged. T and packed
sequence count N are dynamic; H dispatches among 8/12/16/24/32/48/64 and Hg divides
H. D=C=128. The supported launch policy uses the runtime physical cube-core count
(24 on the tested 910B2), not one hardware block per token/chunk.

Only `out` and `final_state` are primary outputs. Existing output scaling by
`128**-0.5` is applied by the reference/test caller, not introduced by this branch.

| Supplied buffer | Production role |
|---|---|
| `g_sum`, `g_t`, `beta_t` | Prefix sums and transposed gates/beta |
| `A` | Signed gated matrix, internally packed by head |
| `A_inv_f32` | Reinterpreted FP16 storage for scaled K and shared QK |
| `A_inv` | Scaled Q, internally packed by head |
| `w`, `u` | Prepared WY operands, internally packed by head |
| `v_new` | Shared KK products |
| `kkt_workspace` | Per-core inverse/decay control metadata |
| `h_workspace` | Two attention/decay producer slots per cube |
| `s`, `wy_workspace_a1/a2`, `o_workspace_qk/qs/gated` | Reserved compatibility arguments; currently unused |

All these buffers are scratch, not a post-call intermediate-results API. Keep
allocating their existing shapes; do not infer their layout from historical
names. No extra allocation is needed. The exact declarations are documented in
[`torch_gdn_mega.h`](../../host/torch_gdn_mega.h), and allocation is demonstrated
by `_prepare_gdn_mega` in [`test_gdn_mega.py`](../../../tests/test_gdn_mega.py).

## Build, test, benchmark

From the repository root, using the prepared CANN container:

```bash
bash scripts/gdn_mega_docker.sh bash scripts/build_gdn_mega.sh
bash scripts/gdn_mega_docker.sh timeout 120 python -m pytest tests/test_gdn_mega.py -q
bash scripts/gdn_mega_docker.sh timeout 300 python benchmarking/gdn_mega.py \
  --initial-state --lengths 128 256 1024 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/production.json
bash scripts/gdn_mega_docker.sh timeout 300 python benchmarking/gdn_mega.py \
  --initial-state --gate-scale 0.01 --lengths 4096 8192 16384 32768 \
  --output benchmark_data/gdn_mega/production_weak.json
```

The build-scope option `GDN_MEGA_ONLY=ON` avoids compiling unrelated public
operators. All supported mega head specializations use the same production
schedule. No algorithm configuration is required.

Measured comparisons, FLOP accounting, memory sizes, and acceptance results are
in the [branch report](../../../docs/gdn_mega/optimization_report.md).

For a shape sweep and annotated plot, see the [heatmap benchmark](../../../docs/gdn_mega/heatmap.md).
