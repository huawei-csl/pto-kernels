# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""Differentiable CPU FP64 oracle for GDN mega backward validation."""

import torch


def recurrent_gdn_reference(
    q, k, v, g, beta, initial_state=None, cu_seqlens=None, *, scale=1.0
):
    """Return output and final states using the literal token recurrence.

    Inputs use packed layouts: q/k [1, T, Hk, K], v [1, T, Hv, V], and
    g/beta [1, T, Hv]. Hv must be divisible by Hk. Gates are log retention,
    i.e. alpha = exp(g). Inputs are used as supplied, without normalization
    or a sigmoid. cu_seqlens [N+1] delimits independent sequences; without
    it the input is one sequence. initial_state is optional [N, Hv, K, V].

    Computation and results are CPU float64. Tensor conversions preserve
    autograd: gradients reach the original leaves in their original dtype
    and device. For FP64 gradient comparisons, create CPU float64 leaves
    before calling this function. There are no in-place state updates.

    Returns output [1, T, Hv, V] and final_state [N, Hv, K, V], including
    unchanged states for empty sequences. scale=1 matches the raw megakernel;
    use scale=K**-0.5 to match attention-scaled output (states are unscaled).
    This is a small-case correctness oracle: autograd retains token states.

    Example (with inputs already requiring gradients)::

        out, ht = recurrent_gdn_reference(q, k, v, g, beta, h0, cu)
        loss = (out * dout.cpu()).sum() + (ht * dht.cpu()).sum()
        loss.backward()  # q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad
    """
    if q.ndim != 4 or k.shape != q.shape or q.shape[0] != 1:
        raise ValueError("q/k must have the same packed shape [1, T, Hk, K]")
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("v must have packed shape [1, T, Hv, V]")
    if g.shape != v.shape[:3] or beta.shape != g.shape:
        raise ValueError("g/beta must have shape [1, T, Hv]")
    tokens, key_heads, key_dim = q.shape[1:]
    heads, value_dim = v.shape[2:]
    if key_heads == 0 or heads == 0 or heads % key_heads:
        raise ValueError("Hv must be a positive multiple of Hk")
    if cu_seqlens is None:
        boundaries = [0, tokens]
    else:
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cu_seqlens must be a one-dimensional integer tensor")
        boundaries = cu_seqlens.cpu().tolist()
    if (
        len(boundaries) < 2
        or boundaries[0] != 0
        or boundaries[-1] != tokens
        or any(a > b for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("cu_seqlens must be nondecreasing and span [0, T]")
    if initial_state is not None and initial_state.shape != (
        len(boundaries) - 1,
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError("initial_state must have shape [N, Hv, K, V]")

    q, k, v, g, beta = (
        x.to(device="cpu", dtype=torch.float64) for x in (q, k, v, g, beta)
    )
    if initial_state is not None:
        initial_state = initial_state.to(device="cpu", dtype=torch.float64)
    # Autograd sums contributions back into shared query/key heads.
    q = q.repeat_interleave(heads // key_heads, dim=2)
    k = k.repeat_interleave(heads // key_heads, dim=2)

    outputs, states = [], []
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (
            v.new_zeros(heads, key_dim, value_dim)
            if initial_state is None
            else initial_state[seq]
        )
        for t in range(start, end):
            qt, kt, vt = q[0, t], k[0, t], v[0, t]
            decayed_state = g[0, t].exp()[:, None, None] * state
            prediction = torch.einsum("hk,hkv->hv", kt, decayed_state)
            update = beta[0, t, :, None] * (vt - prediction)
            state = decayed_state + kt[:, :, None] * update[:, None, :]
            outputs.append(scale * torch.einsum("hk,hkv->hv", qt, state))
        states.append(state)

    final_state = torch.stack(states)
    if outputs:
        out = torch.stack(outputs, dim=0).unsqueeze(0)
    else:
        # Keep zero gradients connected when the entire packed input is empty.
        out = v + (q.sum() + k.sum() + g.sum() + beta.sum()) * 0
        final_state = final_state + out.sum()
    return out, final_state
