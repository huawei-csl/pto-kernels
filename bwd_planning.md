# GDN backward planning

Scope: backward for `csrc/kernel/gdn_mega/kernel_gdn_mega.cpp`. FLA observations refer to commit `12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e`.

## What is the forward interface?

`pto_kernels.pto_gdn_mega` takes `q, k, v, g, beta`, packed sequence boundaries, an optional initial state, and caller-allocated outputs/workspaces. Here `g = log(alpha)`. Head dimension and chunk size are 128. The raw output is unscaled; the test wrapper applies `128**-0.5`.

## What should backward take and return?

Inputs: forward inputs, `dout`, optional `dfinal_state`, and any saved intermediates. Outputs: `dq, dk, dv, dg, dbeta`, plus `dh0` if needed. Sequence boundaries and launch metadata have no gradients.

## What are the gradient formulas?

For one token/head, use column vectors and a key-by-value state matrix. Let `c` be the output scale: 1 for the raw kernel.

```text
Forward:
    a = exp(g)
    P = a * S_prev
    r = v - P.T @ k
    u = beta * r
    S = P + outer(k, u)
    o = c * S.T @ q

Backward, processing tokens in reverse:
    B       = dS + c * outer(q, do)
    dq      = c * S @ do
    du      = B.T @ k
    dbeta   = dot(r, du)
    dv      = beta * du
    dk      = B @ u - P @ dv
    dP      = B - outer(k, dv)
    dg      = sum(dP * P)
    dS_prev = a * dP
```

Initialize `dS` with `dfinal_state`, or zero. Sum `dq/dk` contributions across value heads sharing a query/key head. These equations define the mathematical recurrence gradient; the optimized implementation must be validated numerically.

## Does chunk i depend on other chunks' gradients?

Yes. It consumes the gradient of its outgoing state, summarizing all later chunks' contributions. It returns the preceding state's gradient. It does not need later chunks' individual `dq/dk/dv` tensors. The standard backward state traversal runs from the last chunk to the first.

## Do we need dh0 when the sequence spans multiple chunks?

State gradients are required between every pair of internal chunks. Returning `dh0` for the whole call is unnecessary when its initial state is fixed zero. Return it when that state is trainable or comes from an earlier differentiable call. These are different boundaries.

## How does FLA implement chunked backward?

FLA recomputes `w/u` and forward chunk states, computes local output contributions, propagates state gradients backward, computes output/state derivatives, differentiates WY construction, and combines contributions. A reverse cumulative sum converts cumulative-gate gradients into per-token `dg`. It saves the inverse rather than recomputing it. Logical stages do not imply a fixed launch count. See [FLA backward orchestration][fla-chunk].

## Does FLA have a PyTorch reference and call backward on it?

Yes. Its main chunk tests use a differentiable FP32 token recurrence. They run the same scalar loss on reference and optimized outputs:

```python
loss = (out * dout).sum() + (final_state * dfinal_state).sum()
loss.backward()
```

They compare both forward outputs and all six input gradients. The optimized call uses a custom autograd backward; PyTorch derives the reference gradients automatically. Packed references run per sequence; grouped heads are repeated. Comparisons primarily use relative RMS error. The specifically named prefill test is forward-only; other chunk tests cover backward. See [tests][fla-tests] and [reference][fla-ref].

## Is our existing reference independent and suitable?

`recurrent_reference` in `tests/test_gdn_mega.py` remains the independent FP64 forward oracle. It detaches inputs and uses in-place/scratch operations. The new `recurrent_gdn_reference` in `tests/gdn_mega_bwd_reference.py` uses out-of-place updates and preserves autograd, with CPU FP64 computation. It defaults to raw, unscaled output.

With `tests` on `PYTHONPATH`, use independent CPU FP64 leaves for golden gradients:

```python
from gdn_mega_bwd_reference import recurrent_gdn_reference

q, k, v, g, beta, h0 = (
    x.detach().cpu().double().requires_grad_()
    for x in (q, k, v, g, beta, h0)
)
out, ht = recurrent_gdn_reference(q, k, v, g, beta, h0, cu_seqlens)
loss = (out * dout.cpu()).sum() + (ht * dfinal_state.cpu()).sum()
loss.backward()
# Compare q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad with kernel grads.
```

For a fixed zero initial state, omit `h0` or pass `None`. Run standalone CPU gradient checks with `python -m pytest --noconftest tests/test_gdn_mega_bwd_reference.py -q`; `--noconftest` bypasses the suite's NPU setup.

## Must our forward run before testing our backward?

Only if backward consumes saved forward intermediates. Otherwise it can run directly from inputs and upstream gradients. The PyTorch reference must run forward to build its autograd graph. To compare our forward outputs too, explicitly run our forward kernel.

## Is backward harder, especially numerically?

The recurrence derivatives are simple. Efficient chunked implementation is harder because it needs forward intermediates, reverse state propagation, multiple gradient paths, and reductions. Cancellation and inverse conditioning can increase error. Start with FP32 state-gradient accumulation/reductions and validate against FP64 autograd, including output-only, final-state-only, and combined losses.

## Should backward use the same chunk size?

Start at 128; benchmark 64 if local-memory pressure limits performance. Smaller chunks reduce each chunk matrix quadratically but increase chunk count and boundary-state storage. Changing chunk size requires rebuilding chunk-dependent representations. Neither speed nor accuracy is guaranteed to improve. Compare total training time, peak memory, and gradient error.

## Where should the PTO code live?

Keep it in the requested separate directory `csrc/kernel/gdn_mega_bwd/`. Share helpers where appropriate while keeping forward and backward scheduling independently tunable.

## Does backward also need a C-by-C inverse?

The chunked WY backward uses the forward inverse. If `A = M^-1`, then `dM = -A.T @ dA @ A.T`, retaining only the strictly lower triangle for the variable part of `M`. This uses matrix products, not a new inversion. Save `A`, recompute it, or use equivalent triangular solves. Token-level backward needs no chunk inverse. The forward’s optional `inverse_cache` preserves its factored representation `[D0,0;P,D1]`; backward reconstructs the missing inverse block as `P*D0`. Other opaque scratch buffers do not promise a preserved inverse.

## How much memory does saving intermediates cost?

FLA stores `A` as `[B, L, Hv, C]` in the key dtype, with dense storage despite triangular structure. For FP16/BF16 and total token count `T`, this is `2*T*Hv*C` bytes. At 8,192 tokens and 16 heads: 16 MiB for `C=64`, or 32 MiB for `C=128`, per layer. At `C=D=128`, it equals one FP16 value tensor.

FP32 cumulative gates add `4*T*Hv` bytes. Inputs are retained, not necessarily copied. Optional preprocessing adds tensors. FLA recomputes WY tensors and chunk states during backward; these still count toward peak memory.

## How does this compare with attention?

Conventional attention can retain quadratic `[B,H,L,L]` probabilities. FlashAttention retains Q/K/V, output, and small softmax statistics, then recomputes score/probability tiles. At 8,192 tokens, 16 heads, and dimension 128, one FP16 dense attention matrix is 2 GiB; one output tensor is 32 MiB. Our `C=128` inverse is also 32 MiB. This compares individual tensors, not total training memory. See [FlashAttention][flash].

## Can intermediates be offloaded to CPU RAM or SSD?

Yes: activation offloading moves saved tensors away and restores them before backward. Early layers offer a long transfer window, but aggregate bandwidth and prefetch timing determine stalls. CPU offload and recomputation are initial options; NVMe offload also exists, for example [SSDTrain][ssdtrain]. Handle transfers in the training runtime; PTO kernels consume device-resident tensors.

## Does FLA have an Ascend backward?

Yes, using Triton-Ascend. State backward and output derivatives live under `fla/ops/common/backends/triton_ascend/`; WY backward lives under `fla/ops/gated_delta_rule/backends/triton_ascend/`. Ascend A2 CI includes GDN tests. This establishes implementation/test configuration, not measured performance or a verified CI pass.

## Do six gradients need six kernels, or can we fuse them?

They share substantial work: `dv/dbeta` share `du`, `dk` reuses `dv`, and `dg/dS_prev` share `dP`. `dq` does not depend on future state gradients. Group work by local output derivatives, reverse state propagation, output/state derivatives, and WY derivatives—not by final gradient tensor.

A single megakernel is possible. It can retain the running state gradient and reuse intermediates, but excessive fusion can increase local-memory pressure or serialize chunk-local work. Define logical stages first; decide launch boundaries through profiling.

## What is implemented now, and which forward does it use?

`csrc/kernel/gdn_mega_bwd/` now implements seven sequence-wide PTO stages: chunk preparation, forward-state replay, local gradients, reverse-state propagation, output derivatives, WY backward, and final assembly. Independent stages distribute chunk/head tasks across blocks. State stages traverse chunks inside device kernels. The host submits seven launches for up to 64 packed sequences; larger batches use additional groups. It has no loop over token chunks.

The backward does not call the forward megakernel. Without a cache, it reconstructs the inverse, W/U and chunk states from inputs. The optional `gate_cache=(g_t,g_sum,beta_t)` reuses existing head-major high/low gate prefixes and beta without extra forward computation (1.25 MiB retained at 8K/H=16). `h0` is already reused, chunk states are replayed, and this forward computes no normalization `rstd`. With `inverse_cache` from mega forward, it completes the saved factored inverse using one 64x64 Cube product and skips the triangular inversion; W/U and states are still recomputed. Forward exports existing factors without adding matrix products or launches. The separate FP32 cache uses `[num_chunks,H,128,128]` in packed sequence/chunk order and must remain alive until backward finishes. Stage outputs are retained until their consumers finish. All tensor computation, including packing and assembly, runs in PTO. Python allocates six outputs plus one workspace and makes one native enqueue through Torch's host task queue.

Chunks use 128 tokens, D=128, with FP32 arithmetic. Matrix products run on Cube; vectors handle elementwise operations and reductions. FP32 inputs use shorter Cube accumulations and local dQ reconstruction to limit cancellation. Some chains are fused on chip; matrix temporaries still use GM within stages. Each gradient must satisfy `max_error <= 2 * reordered_PyTorch_max_error + 1e-5`: the golden recurrence uses FP32 for FP16/BF16 inputs, FP64 for FP32 inputs, and the reordered baseline uses the input dtype. Both reference runs use autograd and identical rounded inputs/upstream gradients. Commit `150fed0` and `tests/baselines/gdn_mega_bwd_rough_v0.json` preserve the earlier 934-launch baseline. Commit `c03edb2` preserves the Cube baseline before forward-cache integration. The model autograd wrapper is still separate work. See `memory_bwd.md` for timing, memory and current limitations.

[fla-chunk]: https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/fla/ops/gated_delta_rule/chunk.py
[fla-tests]: https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/tests/ops/test_gdn.py
[fla-ref]: https://github.com/fla-org/flash-linear-attention/blob/12cb9d40bae8cd58325d1f8cbec4c8c8c55e9f3e/fla/ops/gated_delta_rule/naive.py
[flash]: https://arxiv.org/abs/2205.14135
[ssdtrain]: https://arxiv.org/abs/2408.10013
