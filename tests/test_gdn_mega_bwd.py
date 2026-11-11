# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""Per-gradient max error <= 2x reordered PyTorch autograd's max error + 1e-5.

FP16/BF16 inputs use an FP32 golden recurrence; FP32 uses FP64. Golden leaves
and gradients retain the higher precision, matching the kernel's FP32 gradient
return contract. The baseline computes in the input dtype. All three use the
same rounded inputs and upstream gradients, with a 1e-5 absolute allowance.

CPU oracle checks: pytest --noconftest tests/test_gdn_mega_bwd.py -k reference
All NPU cases:     pytest tests/test_gdn_mega_bwd.py
"""
from contextlib import nullcontext

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def recurrent_gdn_reference(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    cu_seqlens=None,
    *,
    scale=1.0,
    dtype=torch.float64,
    reorder_ops=False,
):
    """CPU token recurrence; ordinary autograd computes all six gradients.

    q/k: [1,T,Hk,K], v: [1,T,Hv,V], g/beta: [1,T,Hv]. g is log retention.
    Optional h0: [N,Hv,K,V]; cu_seqlens delimits packed, independent sequences.
    Returns output [1,T,Hv,V] and final states [N,Hv,K,V]. Empty states survive.
    No normalization/sigmoid; scale affects output only (raw megakernel: 1).
    Casts preserve gradients to the caller's leaves. FP64 is the default;
    reorder_ops=True in the input dtype supplies the numerical error baseline.
    No chunk algorithm, in-place state updates or device kernel dependencies.
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

    q, k, v, g, beta = (x.to(device="cpu", dtype=dtype) for x in (q, k, v, g, beta))
    if initial_state is not None:
        initial_state = initial_state.to(device="cpu", dtype=dtype)
    # Autograd sums contributions back into shared query/key heads.
    q = q.repeat_interleave(heads // key_heads, dim=2)
    k = k.repeat_interleave(heads // key_heads, dim=2)

    # Unbind once: per-token indexing would make backward scatter a full
    # sequence tensor at every step. The recurrence and arithmetic are unchanged.
    qs, ks, vs, gs, bs = (x[0].unbind(0) for x in (q, k, v, g, beta))
    outputs, states = [], []
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (
            v.new_zeros(heads, key_dim, value_dim)
            if initial_state is None
            else initial_state[seq]
        )
        for t in range(start, end):
            qt, kt, vt = qs[t], ks[t], vs[t]
            decayed_state = gs[t].exp()[:, None, None] * state
            prediction = torch.einsum("hk,hkv->hv", kt, decayed_state)
            if reorder_ops:
                kb = kt * bs[t][:, None]
                state = (
                    decayed_state
                    + kb[:, :, None] * vt[:, None, :]
                    - kb[:, :, None] * prediction[:, None, :]
                )
                outputs.append(torch.einsum("hk,hkv->hv", qt * scale, state))
            else:
                update = bs[t][:, None] * (vt - prediction)
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


def _reference_inputs(tokens=4, sequences=1, dtype=torch.float64):
    rng = torch.Generator().manual_seed(42)

    def randn(*shape):
        return torch.randn(*shape, generator=rng, dtype=dtype) * 0.2

    # Small unequal K/V dimensions and shared Q/K heads make gradcheck cheap.
    values = (
        randn(1, tokens, 1, 2),
        randn(1, tokens, 1, 2),
        randn(1, tokens, 2, 3),
        -torch.rand(1, tokens, 2, generator=rng, dtype=dtype),
        torch.rand(1, tokens, 2, generator=rng, dtype=dtype),
        randn(sequences, 2, 2, 3),
    )
    return tuple(x.requires_grad_() for x in values)


@pytest.mark.parametrize("with_initial_state", [False, True])
def test_reference_gradcheck(with_initial_state):
    inputs = _reference_inputs(sequences=4)
    cu = torch.tensor([0, 0, 1, 4, 4])
    if not with_initial_state:
        inputs = inputs[:5]
    # Finite differences independently validate every input and both outputs.
    assert torch.autograd.gradcheck(
        lambda *xs: recurrent_gdn_reference(*xs, cu_seqlens=cu, scale=0.7),
        inputs,
    )


@pytest.mark.parametrize("loss_kind", ["output", "state", "both"])
def test_reference_backward_across_call_boundary(loss_kind):
    whole_inputs = _reference_inputs(tokens=5)
    split_inputs = tuple(x.detach().clone().requires_grad_() for x in whole_inputs)
    whole_o, whole_h = recurrent_gdn_reference(*whole_inputs)
    q, k, v, g, beta, h0 = split_inputs
    first_o, first_h = recurrent_gdn_reference(
        *(x[:, :2] for x in (q, k, v, g, beta)), h0
    )
    second_o, split_h = recurrent_gdn_reference(
        *(x[:, 2:] for x in (q, k, v, g, beta)), first_h
    )
    split_o = torch.cat((first_o, second_o), dim=1)
    torch.testing.assert_close(split_o, whole_o)
    torch.testing.assert_close(split_h, whole_h)

    rng = torch.Generator().manual_seed(17)
    do = torch.randn(whole_o.shape, generator=rng, dtype=torch.float64)
    dh = torch.randn(whole_h.shape, generator=rng, dtype=torch.float64)
    for out, state in ((whole_o, whole_h), (split_o, split_h)):
        loss = 0
        if loss_kind != "state":
            loss = loss + (out * do).sum()
        if loss_kind != "output":
            loss = loss + (state * dh).sum()
        loss.backward()
    for whole, split in zip(whole_inputs, split_inputs):
        if whole.grad is None:  # Queries have no effect on the final state.
            assert split.grad is None
        else:
            torch.testing.assert_close(split.grad, whole.grad)
    assert torch.count_nonzero(whole_inputs[1].grad[:, :2]) > 0


def test_reference_packed_sequences_are_independent():
    inputs = _reference_inputs(tokens=5, sequences=3)
    q, k, v, g, beta, h0 = inputs
    cu = torch.tensor([0, 2, 2, 5])
    out, ht = recurrent_gdn_reference(*inputs, cu_seqlens=cu)
    for seq, (start, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        seq_out, seq_ht = recurrent_gdn_reference(
            *(x[:, start:end] for x in (q, k, v, g, beta)), h0[seq : seq + 1]
        )
        torch.testing.assert_close(out[:, start:end], seq_out)
        torch.testing.assert_close(ht[seq : seq + 1], seq_ht)
    # A loss on the last sequence must not leak into preceding sequences.
    out[:, 2:].square().sum().backward()
    for x in inputs[:5]:
        assert torch.count_nonzero(x.grad[:, :2]) == 0
    assert torch.count_nonzero(h0.grad[:2]) == 0


def test_reference_empty_sequences():
    inputs = _reference_inputs(tokens=0, sequences=2)
    out, ht = recurrent_gdn_reference(*inputs, cu_seqlens=torch.tensor([0, 0, 0]))
    assert out.shape == (1, 0, 2, 3)
    torch.testing.assert_close(ht, inputs[-1])
    (out.sum() + ht.sum()).backward()
    for x in inputs[:5]:
        assert x.grad is not None and x.grad.numel() == 0
    torch.testing.assert_close(inputs[-1].grad, torch.ones_like(inputs[-1]))


def test_reference_conversions_preserve_gradients_and_inputs():
    inputs = _reference_inputs(dtype=torch.float32)
    snapshots = [x.detach().clone() for x in inputs]
    out, ht = recurrent_gdn_reference(*inputs, scale=2**-0.5)
    assert out.dtype == ht.dtype == torch.float64
    (out.square().sum() + ht.square().sum()).backward()
    for x, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(x, snapshot, rtol=0, atol=0)
        assert x.grad is not None and x.grad.dtype == x.dtype
        assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("tokens", [0, 5])
def test_reference_reordered_reference_matches_equations(tokens):
    inputs = _reference_inputs(tokens=tokens, sequences=3)
    cu = torch.tensor([0, 0, tokens // 2, tokens])
    golden = recurrent_gdn_reference(*inputs, cu_seqlens=cu, scale=0.7)
    reordered = recurrent_gdn_reference(
        *inputs, cu_seqlens=cu, scale=0.7, reorder_ops=True
    )
    upstream = tuple(torch.randn_like(x) for x in golden)
    for a, b in zip(golden, reordered):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    for a, b in zip(
        torch.autograd.grad(golden, inputs, upstream),
        torch.autograd.grad(reordered, inputs, upstream),
    ):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def backward(*args, **kwargs):
    # Keep CPU reference checks independent of CANN and the compiled extension.
    from pto_kernels import pto_gdn_mega_bwd

    return pto_gdn_mega_bwd(*args, **kwargs)


# The five saved-buffer variants have identical fixed-seed model inputs.
# Reuse only their CPU oracle results within this process; every NPU run and
# assertion still executes. Nothing is cached on disk or in production code.
saved_references = {}


def test_native_binding():
    import inspect
    from pto_kernels import pto_gdn_mega_bwd as backward, pto_kernels_ops

    assert inspect.isbuiltin(backward)
    assert backward is pto_kernels_ops.pto_gdn_mega_bwd


@pytest.mark.parametrize(
    "invalid", ["device", "dtype", "beta", "state", "bounds", "inverse", "gates"]
)
def test_invalid_inputs(invalid, npu_device):
    q = torch.empty(1, 2, 2, 128, device=npu_device, dtype=torch.float16)
    v = torch.empty(1, 2, 8, 128, device=npu_device, dtype=q.dtype)
    g = torch.empty(1, 2, 8, device=npu_device)
    args = {"q": q, "k": q, "v": v, "g": g, "beta": g, "dout": v}
    changes = {
        "device": lambda: {"q": q.cpu()},
        "dtype": lambda: {"q": torch.empty_like(q, dtype=torch.int32)},
        "beta": lambda: {"beta": g[..., :7]},
        "state": lambda: {"initial_state": v},
        "bounds": lambda: {"cu_seqlens": torch.tensor([0, 3, 2])},
        "inverse": lambda: {"inverse_cache": v},
        "gates": lambda: {"gate_cache": (g, g)},
    }
    with pytest.raises(RuntimeError, match="gdn_mega_bwd:"):
        backward(**{**args, **changes[invalid]()})


@pytest.mark.parametrize("metadata_device", ["cpu", "npu"])
def test_strided_int64_boundaries(metadata_device, npu_device):
    q = torch.randn(1, 2, 2, 128, device=npu_device) * 0.1
    v = torch.randn(1, 2, 8, 128, device=npu_device)
    g = -torch.rand(1, 2, 8, device=npu_device) * 0.1
    args = (q, q, v, g, torch.rand_like(g), v)
    expected = backward(*args)
    cu = torch.tensor(
        [0, -1, 2],
        dtype=torch.int64,
        device=npu_device if metadata_device == "npu" else "cpu",
    )[::2]
    actual = backward(*args, cu_seqlens=cu)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("cache_dtype", [None, torch.float16, torch.float32])
@pytest.mark.parametrize("heads,lengths", [(8, [256] + [0] * 64), (64, [1408, 0])])
def test_graph_device_boundaries(cache_dtype, heads, lengths, npu_device):
    """Replay changing chunk counts, cache offsets and the fusion threshold."""
    inputs, dtypes, _, scale = backward_inputs(
        lengths,
        heads,
        heads // 4,
        "both",
        torch.float16,
        "both" if cache_dtype else None,
    )
    inputs = [x.detach().to(npu_device, dt) for x, dt in zip(inputs, dtypes)]
    do, dh = torch.randn_like(inputs[2]), torch.randn_like(inputs[5])
    tokens = sum(lengths)
    # A strided int64 view also checks device indexing during replay.
    cu = torch.empty(2 * (len(lengths) + 1), device=npu_device, dtype=torch.int64)[::2]
    cache = (
        torch.empty(
            (tokens + 127) // 128 + 1,
            heads,
            128,
            128,
            device=npu_device,
            dtype=cache_dtype,
        )
        if cache_dtype
        else None
    )
    gates = None
    graph, actual = None, None
    for packing in (lengths, [1] + [0] * (len(lengths) - 2) + [tokens - 1], lengths):
        cpu_cu = torch.tensor([0, *torch.tensor(packing).cumsum(0).tolist()])
        cu.copy_(cpu_cu)
        inputs[0].neg_()
        do.neg_()
        if cache_dtype:
            fresh_cache, fresh_gates = forward_caches(
                inputs, cpu_cu, "both", cache.dtype, False
            )
            cache[: fresh_cache.shape[0]].copy_(fresh_cache)
            if gates is None:
                gates = fresh_gates
            else:
                for dst, src in zip(gates, fresh_gates):
                    dst.copy_(src)
        args = (*inputs[:5], do, inputs[5], dh)
        kwargs = {"scale": scale, "inverse_cache": cache, "gate_cache": gates}
        expected = backward(*args, cpu_cu, **kwargs)
        if graph is None:
            for _ in range(2):
                backward(*args, cu, **kwargs)
            torch.npu.synchronize(npu_device)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                actual = backward(*args, cu, **kwargs)
        graph.replay()
        torch.npu.synchronize(npu_device)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("cpu_bounds", [False, True])
def test_graph_static_boundaries(cpu_bounds, npu_device):
    inputs, dtypes, cu, scale = backward_inputs(
        [129] + [0] * 63 + [128] if cpu_bounds else [257],
        8,
        2,
        "both",
        torch.float32,
        None,
    )
    inputs = [x.detach().to(npu_device, dt) for x, dt in zip(inputs, dtypes)]
    do, dh = torch.randn_like(inputs[2]), torch.randn_like(inputs[5])
    boundaries = cu.clone() if cpu_bounds else None
    args = (*inputs[:5], do, inputs[5], dh)
    for _ in range(2):
        backward(*args, boundaries, scale=scale)
    torch.npu.synchronize(npu_device)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        with AllocationsOnly() as guard:
            actual = backward(*args, boundaries, scale=scale)
    assert guard.allocations == 7
    if boundaries is not None:
        boundaries.fill_(-1)
    del boundaries
    for _ in range(3):
        inputs[0].neg_()
        do.neg_()
        expected = backward(*args, cu if cpu_bounds else None, scale=scale)
        graph.replay()
        torch.npu.synchronize(npu_device)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


class AllocationsOnly(TorchDispatchMode):
    """Reject accidental Torch computation inside the backward implementation."""

    allocations = 0

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        assert func == torch.ops.aten.empty.memory_format, str(func)
        self.allocations += 1
        return func(*args, **(kwargs or {}))


def forward_caches(inputs, cu, cache_mode, cache_dtype, roundtrip):
    cache, gate_cache = None, None
    if not cache_mode:
        return cache, gate_cache
    from test_gdn_mega import _prepare_gdn_mega, pto_gdn_mega

    launch = _prepare_gdn_mega(*inputs[:5], cu.to(inputs[0].device), inputs[5])
    heads = inputs[2].shape[2]
    if cache_mode != "gates":
        cache = torch.empty(
            launch["num_matrices"] // heads,
            heads,
            128,
            128,
            device=inputs[0].device,
            dtype=(torch.float32 if roundtrip else (cache_dtype or torch.float32)),
        )
    pto_gdn_mega(**launch)
    out_before, state_before = (
        launch["out"].clone(),
        launch["final_state"].clone(),
    )
    pto_gdn_mega(**launch, inverse_cache=cache)
    torch.testing.assert_close(launch["out"], out_before, atol=0, rtol=0)
    torch.testing.assert_close(launch["final_state"], state_before, atol=0, rtol=0)
    if cache is not None:
        if cache.dtype != torch.float32:
            full = torch.empty_like(cache, dtype=torch.float32)
            pto_gdn_mega(**launch, inverse_cache=full)
            torch.testing.assert_close(cache, full.to(cache.dtype), atol=0, rtol=0)
        # Numerical storage experiment only; production does no Torch math.
        if roundtrip:
            cache = cache.to(cache_dtype).float()
    if cache_mode != "inverse":
        gate_cache = tuple(launch[name] for name in ("g_t", "g_sum", "beta_t"))
    return cache, gate_cache


def check_gradients(expected, baseline, actual, expected64, error_report):
    failures = []
    for index, (name, ref, pt, result) in enumerate(
        zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), expected, baseline, actual)
    ):
        if not ref.numel():
            assert result.shape == ref.shape
            continue
        result = result.cpu().double()
        assert all(torch.isfinite(x).all() for x in (ref, pt, result)), name
        error = (result - ref.double()).abs().max().item()
        budget = 2 * (pt.double() - ref.double()).abs().max().item() + 1e-5
        record = {"gradient": name, "error": error, "budget": budget}
        if expected64 is not None:
            ref64 = expected64[index]
            diff = result - ref64
            error_l2 = torch.linalg.vector_norm(diff).item()
            reference_l2 = torch.linalg.vector_norm(ref64).item()
            passed = error_l2 <= 0.02 * reference_l2
            record["flashqla"] = {
                "error_l2": error_l2,
                "reference_l2": reference_l2,
                "relative_l2": (
                    error_l2 / reference_l2
                    if reference_l2
                    else (0.0 if error_l2 == 0 else None)
                ),
                "max_abs_error": diff.abs().max().item(),
                "reference_max_abs": ref64.abs().max().item(),
                "reference_rms": reference_l2 / ref64.numel() ** 0.5,
                "threshold": 0.02,
                "passed": passed,
            }
            if not passed:
                failures.append(
                    f"{name}: FlashQLA L2 error {error_l2:.9g} > 0.02 * reference L2 {reference_l2:.9g}"
                )
        if error_report is not None:
            error_report.append(record)
        if error > budget:
            failures.append(
                f"{name}: kernel error {error:.9g} > 2x PyTorch error + 1e-5 = {budget:.9g}"
            )
    return failures


def backward_inputs(lengths, heads, key_heads, mode, dtype, cache_mode):
    torch.manual_seed(42)
    T, N = sum(lengths), len(lengths)
    q = torch.randn(1, T, key_heads, 128)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, T, heads, 128)
    g = -torch.rand(1, T, heads) * 0.1
    beta = torch.rand_like(g)
    h0 = torch.randn(N, heads, 128, 128) * 0.1
    if mode in ("zero_state", "defaults"):
        h0.zero_()
    if mode == "zero_beta":
        beta.zero_()
    beta_exponent = {"tiny_beta": -20, "small_beta": -15, "normal_beta": -12}
    if mode in beta_exponent:
        beta.fill_(2 ** beta_exponent[mode])
    if mode == "weak":
        g.mul_(1e-4)
    if mode in ("correlated_weak", "correlated"):
        k = torch.nn.functional.normalize(k[:, :1] + 1e-3 * k, dim=-1)
        beta.fill_(1)
        if mode == "correlated_weak":
            g.mul_(1e-4)
    if mode == "strong_decay":
        g.mul_(5).sub_(0.25)  # log retention [-0.75, -0.25], exp(g) ~0.47–0.78.
    if mode == "replace":
        k.zero_()
        k[..., 0] = 1
        beta.fill_(1)
        g.zero_()
    scale = 1.0 if mode in ("replace", "zero_beta") else 0.125
    # Use the exact rounded inputs supplied to the device kernel.
    golden_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    input_dtypes = [dtype] * 6
    if mode == "mixed":
        input_dtypes[3] = input_dtypes[5] = torch.float32
    if cache_mode:
        input_dtypes[3] = torch.float32  # Forward mega's gate dtype.
    inputs = [
        x.to(dt).to(golden_dtype).requires_grad_()
        for x, dt in zip((q, k, v, g, beta, h0), input_dtypes)
    ]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return inputs, input_dtypes, cu, scale


@pytest.mark.parametrize(
    "lengths,heads,key_heads,mode",
    [
        ([2], 2, 1, "both"),
        ([129], 8, 2, "both"),
        ([0, 1, 0, 17, 0], 8, 2, "output"),
        ([3, 5], 8, 8, "state"),
        ([0, 0], 8, 2, "both"),
        ([257], 8, 1, "zero_state"),
        ([129], 8, 2, "zero_beta"),
        ([129], 8, 2, "replace"),
        ([17], 64, 8, "both"),
        ([17, 0, 19], 64, 16, "both"),
        ([129] + [1] * 10, 64, 16, "both"),
        ([513], 8, 2, "weak"),
        ([257], 8, 2, "correlated_weak"),
        ([256], 8, 2, "strong_decay"),
        ([7], 8, 2, "defaults"),
        ([128], 8, 2, "both"),
        ([256], 8, 2, "both"),
        ([129, 0, 131], 8, 2, "strided"),
        ([257], 8, 2, "replace"),
        ([513], 8, 2, "replace"),
        ([127, 1, 0, 129], 3, 1, "both"),
        ([0] * 64 + [1], 2, 1, "both"),
        pytest.param([0, 1, 0, 3] * 16, 8, 2, "both", id="many_short_sequences"),
        ([2], 64, 1, "both"),
        ([129], 8, 2, "mixed"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_backward(
    lengths,
    heads,
    key_heads,
    mode,
    dtype,
    npu_device,
    cache_mode=None,
    cache_dtype=None,
    *,
    error_report=None,
    roundtrip=False,
    flashqla_bound=False,
):
    # Load the extension before the allocation-only guard (including no-cache runs).
    from pto_kernels import pto_gdn_mega_bwd as backward

    inputs, input_dtypes, cu, scale = backward_inputs(
        lengths, heads, key_heads, mode, dtype, cache_mode
    )
    golden_dtype = inputs[0].dtype
    key = (tuple(lengths), heads, key_heads, mode, dtype, tuple(input_dtypes))
    cached = saved_references.get(key) if cache_mode else None
    if cached is None:
        out, ht = recurrent_gdn_reference(
            *inputs, cu_seqlens=cu, scale=scale, dtype=golden_dtype
        )
        do, dh = torch.randn_like(out).to(dtype), torch.randn_like(ht).to(dtype)
        if mode == "state":
            do.zero_()
        if mode in ("output", "defaults"):
            dh.zero_()
        expected = torch.autograd.grad(
            (out, ht), inputs, (do.to(golden_dtype), dh.to(golden_dtype))
        )
        pt_inputs = [x.detach().to(dtype).requires_grad_() for x in inputs]
        pt_out, pt_ht = recurrent_gdn_reference(
            *pt_inputs, cu_seqlens=cu, scale=scale, dtype=dtype, reorder_ops=True
        )
        baseline = torch.autograd.grad((pt_out, pt_ht), pt_inputs, (do, dh))
        if cache_mode:
            saved_references[key] = do, dh, expected, baseline
    else:
        do, dh, expected, baseline = cached
    expected64 = None
    if flashqla_bound:
        # Keep the original rounded inputs AND upstream gradients. Only raise
        # oracle precision; generating new FP64 random adjoints changes the case.
        leaves64 = [x.detach().double().requires_grad_() for x in inputs]
        out64, ht64 = recurrent_gdn_reference(
            *leaves64, cu_seqlens=cu, scale=scale, dtype=torch.float64
        )
        expected64 = torch.autograd.grad(
            (out64, ht64), leaves64, (do.double(), dh.double())
        )
    device_inputs = [
        x.detach().to(device=npu_device, dtype=dt)
        for x, dt in zip(inputs, input_dtypes)
    ]
    context = (
        torch.npu.stream(torch.npu.Stream(device=npu_device))
        if mode in ("defaults", "strided")
        else nullcontext()
    )
    with context:
        device_do, device_dh = do.to(npu_device), dh.to(npu_device)
        device_cu = cu.to(npu_device)
        cache, gate_cache = forward_caches(
            device_inputs, cu, cache_mode, cache_dtype, roundtrip
        )
        cache_before = cache.clone() if cache is not None else None
        gates_before = tuple(x.clone() for x in gate_cache) if gate_cache else ()
        if mode in ("defaults", "strided"):
            device_inputs = [
                x.transpose(-1, -2).contiguous().transpose(-1, -2)
                for x in device_inputs
            ]
            device_do, device_dh = [
                x.transpose(-1, -2).contiguous().transpose(-1, -2)
                for x in (device_do, device_dh)
            ]
        with AllocationsOnly() as guard:
            actual = backward(
                *device_inputs[:5],
                device_do,
                None if mode in ("zero_state", "defaults") else device_inputs[5],
                None if mode in ("output", "defaults") else device_dh,
                None if mode == "defaults" else device_cu,
                scale=scale,
                inverse_cache=cache,
                gate_cache=gate_cache,
            )
        assert guard.allocations == 7  # Six outputs and one reusable workspace.
        torch.npu.synchronize(npu_device)
    failures = check_gradients(expected, baseline, actual, expected64, error_report)
    for actual_input, original in zip(device_inputs, inputs):
        torch.testing.assert_close(
            actual_input.cpu().double(), original.double(), atol=0, rtol=0
        )
    if cache is not None:
        torch.testing.assert_close(cache, cache_before, atol=0, rtol=0)
    if gate_cache is not None:
        for x, before in zip(gate_cache, gates_before):
            torch.testing.assert_close(x, before, atol=0, rtol=0)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "lengths,heads,key_heads,mode",
    [
        ([129], 8, 2, "both"),
        ([256], 16, 4, "both"),
        ([0, 1, 17, 0, 129], 8, 2, "strided"),
        ([257], 8, 2, "replace"),
        ([17, 0, 19], 64, 16, "both"),
        ([129] + [1] * 10, 64, 16, "both"),
        ([513], 8, 2, "weak"),
        ([257], 8, 2, "correlated_weak"),
        ([256], 8, 2, "strong_decay"),
        ([0, 0], 8, 2, "both"),
        ([1] + [0] * 63 + [1], 8, 2, "both"),
        ([1] * 48, 48, 16, "both"),
        *[([65], h, h // 4, "both") for h in (12, 24, 32, 48, 64)],
    ],
)
@pytest.mark.parametrize(
    "cache_mode,cache_dtype",
    [
        ("inverse", torch.float32),
        ("gates", torch.float32),
        ("both", torch.float32),
        ("both", torch.float16),
        ("both", torch.bfloat16),
    ],
)
def test_saved_forward(
    lengths, heads, key_heads, mode, cache_mode, cache_dtype, npu_device
):
    test_backward(
        lengths,
        heads,
        key_heads,
        mode,
        torch.float16,
        npu_device,
        cache_mode,
        cache_dtype,
    )


def test_queued_calls(npu_device):
    """Torch producers/consumers and workspace reuse without intervening waits."""
    torch.manual_seed(42)
    q = torch.randn(1, 129, 1, 128, device=npu_device) * 0.1
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, 129, 2, 128, device=npu_device)
    g = -torch.rand(1, 129, 2, device=npu_device) * 0.1
    beta, do = torch.rand_like(g), torch.randn_like(v)
    first = backward(q, k, v, g, beta, do)
    consumed = tuple(x.clone() for x in first)
    second = backward(q, k, v, g, beta, -do)
    torch.npu.synchronize(npu_device)
    for a, b, saved in zip(first, second, consumed):
        torch.testing.assert_close(a, saved, atol=0, rtol=0)
        torch.testing.assert_close(a, -b, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("cached", [False, True])
def test_cross_stream_inputs(cached, npu_device):
    """Drop inputs/caches after enqueue, then reuse memory on their producer stream."""
    torch.manual_seed(42)
    q = torch.randn(1, 129, 2, 128, device=npu_device, dtype=torch.float16) * 0.1
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(1, 129, 8, 128, device=npu_device, dtype=q.dtype)
    g = -torch.rand(1, 129, 8, device=npu_device) * 0.1
    inputs = [q, k, v, g, torch.rand_like(g).half(), torch.randn_like(v)]
    if cached:
        from test_gdn_mega import _prepare_gdn_mega, pto_gdn_mega

        launch = _prepare_gdn_mega(
            *inputs[:5], torch.tensor([0, 129], device=npu_device)
        )
        cache = torch.empty(2, 8, 128, 128, device=npu_device, dtype=q.dtype)
        pto_gdn_mega(**launch, inverse_cache=cache)
        inputs += [cache, *(launch[name] for name in ("g_t", "g_sum", "beta_t"))]

    def call(xs):
        return backward(
            *xs[:6],
            inverse_cache=xs[6] if cached else None,
            gate_cache=tuple(xs[7:-1]) if cached else None,
            cu_seqlens=xs[-1],
        )

    inputs.append(torch.tensor([0, 129], device=npu_device))
    expected = call(inputs)
    producer = torch.npu.current_stream(npu_device)
    consumer = torch.npu.Stream(device=npu_device)
    copies = [x.clone() for x in inputs]
    consumer.wait_stream(producer)
    with torch.npu.stream(consumer):
        actual = call(copies)
    del copies
    pressure = [torch.empty_like(x).fill_(7) for _ in range(4) for x in inputs]
    producer.wait_stream(consumer)
    torch.npu.synchronize(npu_device)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    del pressure


@pytest.mark.parametrize(
    "length,heads,key_heads,mode,cache_dtype",
    [
        *[
            (t, h, k, "both", torch.bfloat16)
            for t, h, k in (
                (512, 64, 16),
                (1536, 16, 2),
                (384, 64, 64),
                (768, 32, 16),
                (1536, 48, 16),
                (8192, 8, 2),
            )
        ],
        *[
            (512, 64, 16, "correlated", dtype)
            for dtype in (torch.float32, torch.float16)
        ],
        pytest.param(
            512,
            64,
            16,
            "correlated",
            torch.bfloat16,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="99b370b82: BF16 inverse rounding exceeds gradient bounds for correlated keys",
            ),
        ),
        *[
            (384, 64, 64, mode, torch.bfloat16)
            for mode in ("tiny_beta", "small_beta", "normal_beta")
        ],
    ],
)
def test_grouped(length, heads, key_heads, mode, cache_dtype, npu_device):
    test_backward(
        [length], heads, key_heads, mode, torch.float16, npu_device, "both", cache_dtype
    )


@pytest.mark.parametrize("with_initial_state", [False, True])
def test_forward_oracle_consistency(with_initial_state):
    from test_gdn_mega import _make_inputs, recurrent_reference

    q, k, v, g, beta = _make_inputs(131, 8, 2, 42)
    cu = torch.tensor([0, 0, 129, 131, 131], dtype=torch.int32)
    initial = (
        torch.randn(4, 8, 128, 128, dtype=torch.float16) * 0.1
        if with_initial_state
        else None
    )
    expected = recurrent_reference(q, k, v, g, beta, initial, cu)
    actual = recurrent_gdn_reference(q, k, v, g, beta, initial, cu)
    for ref, out in zip(expected, actual):
        torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
