# GDN input ranges and numerical scope

The primary workload uses normalized Q/K, beta in [0,1], and nonpositive log
gates g; per-token retention is alpha=exp(g). The ABI consumes FP16 Q/K/V/beta
and initial/final state, and FP32 log gates. Outputs are FP16; raw output is
scaled by 128**-0.5 by the caller. Gaussian values/states have no finite
mathematical magnitude bound.

## Observed Qwen inputs

Historical captures from the user's Qwen3.5-0.8B forward passes contain six
prompts, 346–1024 tokens each, across 18 linear-attention layers and 16 value
heads: 1,404,288 token-head samples. These are observations, not bounds on all
models or on long-context inference.

| Quantity | Minimum | p1 | Median | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Log gate g | -9.31484 | -3.13377 | -0.006174 | -4.827e-7 | -5.989e-11 |
| Beta before FP16 conversion | 1.182e-12 | 0.002551 | 0.352994 | 0.980206 | 0.999992 |
| Beta consumed by kernel/oracle | 0 | 0.002550 | 0.353027 | 0.979980 | 1 |

36.97% of gates exceed -0.001; 1.60% exceed -1e-6. Weak decay occurs in real inputs,
so it must not automatically be treated as a rare artificial corner. Beta can
round to either endpoint. Captured max abs(V) was 19.40625 and supplied-state
max was about 12.60. Correlation and temporal structure matter; constant tiny
gates across 32K tokens are a different experiment from these captures.
Raw capture/report artifacts are not required by the portable regression suite.
Historical evidence remains in Git history before the documentation cleanup.

## Regression distributions

The single test file includes 178 fixed cases, retaining the exact seeds and
parameters from the pre-cleanup suite. They cover Gaussian and positive inputs,
zero/supplied/large state, beta endpoints and interior values, zero/tiny/strong
gates, identical/nearly correlated keys, correlation sweeps, partial chunks and
packed/empty sequences through 32K. A few explicitly exploratory cases use beta
above 1 or expanding gates; their passing examples do not extend the supported
primary regime to arbitrary such inputs.

Positive/Gaussian, logit, mask and gate-activation cases were adapted from FLA's
[pinned GDN tests](https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/tests/ops/test_gdn.py).
They adapt distributions to this FP16-input/D128 ABI, not the whole FLA API,
backward pass or dtype/shape matrix.

## Numerical design and limitations

Highly correlated keys, large beta and weak decay increase coupling in the
triangular solve. Large states/values amplify absolute error; cancellation
near zero makes relative error alone misleading. Long recurrences also
accumulate state-update and prefix errors. These concerns extend beyond the
inverse and gate exponentiation: Gram products, projections, RHS formation,
state updates and final FP16 rounding also contribute.

The kernel keeps Gram matrices and the authoritative state in FP32. It solves
16×16 inverse leaves by FP32 substitution and joins blocks with native FP32
cube operations. Projections use two FP16 state components with FP32
accumulation. Gate differences retain G_i-G_j, and Gamma uses exp(G_end-G_i);
there is no inverse-power truncation, gate clipping or sequential fallback.

Every output/state element is checked against an independent CPU FP64 token
recurrence with fixed atol=1e-4 and rtol=1e-3, plus finiteness and RMSE checks.
This is a mixed bound, not a maximum absolute error of 1e-4 for all magnitudes.
The user approved increasing the historical atol from 1e-5; previously failing
stress cases remain regular tests. See the [error heatmap](error_heatmap.md)
for scope. Run `python benchmarking/gdn_mega.py test` for the full test suite.
