# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""Per-gradient max error <= 2x reordered PyTorch autograd's max error + 1e-5.

FP16/BF16 inputs use an FP32 golden recurrence; FP32 uses FP64. Golden leaves
and gradients retain the higher precision, matching the kernel's FP32 gradient
return contract. The baseline computes in the input dtype. All three use the
same rounded inputs and upstream gradients, with a 1e-5 absolute allowance.
"""
from contextlib import nullcontext

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from csrc.kernel.gdn_mega_bwd.backward import backward
from gdn_mega_bwd_reference import recurrent_gdn_reference


# The five saved-buffer variants have identical fixed-seed model inputs.
# Reuse only their CPU oracle results within this process; every NPU run and
# assertion still executes. Nothing is cached on disk or in production code.
_saved_references = {}


class AllocationsOnly(TorchDispatchMode):
    """Reject accidental Torch computation inside the backward implementation."""

    allocations = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func == torch.ops.aten.empty.memory_format, str(func)
        self.allocations += 1
        return func(*args, **(kwargs or {}))


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
    error_report=None,
    roundtrip=False,
    flashqla_bound=False,
):
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
    if mode == "weak":
        g.mul_(1e-4)
    if mode == "correlated_weak":
        k = torch.nn.functional.normalize(k[:, :1] + 1e-3 * k, dim=-1)
        beta.fill_(1)
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
    key = (tuple(lengths), heads, key_heads, mode, dtype, tuple(input_dtypes))
    cached = _saved_references.get(key) if cache_mode else None
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
            _saved_references[key] = do, dh, expected, baseline
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
        cache, gate_cache = None, None
        if cache_mode:
            from test_gdn_mega import _prepare_gdn_mega, pto_gdn_mega

            launch = _prepare_gdn_mega(
                *device_inputs[:5], cu.to(npu_device), device_inputs[5]
            )
            if cache_mode != "gates":
                cache = torch.empty(
                    launch["num_matrices"] // heads,
                    heads,
                    128,
                    128,
                    device=npu_device,
                    dtype=(
                        torch.float32 if roundtrip else (cache_dtype or torch.float32)
                    ),
                )
            pto_gdn_mega(**launch)
            out_before, state_before = (
                launch["out"].clone(),
                launch["final_state"].clone(),
            )
            pto_gdn_mega(**launch, inverse_cache=cache)
            torch.testing.assert_close(launch["out"], out_before, atol=0, rtol=0)
            torch.testing.assert_close(
                launch["final_state"], state_before, atol=0, rtol=0
            )
            if cache is not None:
                if cache.dtype != torch.float32:
                    full = torch.empty_like(cache, dtype=torch.float32)
                    pto_gdn_mega(**launch, inverse_cache=full)
                    torch.testing.assert_close(
                        cache, full.to(cache.dtype), atol=0, rtol=0
                    )
                # Numerical storage experiment only; production does no Torch math.
                if roundtrip:
                    cache = cache.to(cache_dtype).float()
                cache_before = cache.clone()
            if cache_mode != "inverse":
                gate_cache = tuple(launch[name] for name in ("g_t", "g_sum", "beta_t"))
                gates_before = tuple(x.clone() for x in gate_cache)
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
                None if mode == "defaults" else cu,
                scale=scale,
                inverse_cache=cache,
                gate_cache=gate_cache,
            )
        assert guard.allocations == 7  # Six outputs and one reusable workspace.
        torch.npu.synchronize(npu_device)
    failures = []
    for index, (name, ref, pt, result) in enumerate(
        zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), expected, baseline, actual)
    ):
        if not ref.numel():
            assert result.shape == ref.shape
            continue
        assert all(torch.isfinite(x).all() for x in (ref, pt, result.cpu())), name
        error = (result.cpu().double() - ref.double()).abs().max().item()
        budget = 2 * (pt.double() - ref.double()).abs().max().item() + 1e-5
        record = dict(gradient=name, error=error, budget=budget)
        if expected64 is not None:
            ref64 = expected64[index]
            diff = result.cpu().double() - ref64
            error_l2 = torch.linalg.vector_norm(diff).item()
            reference_l2 = torch.linalg.vector_norm(ref64).item()
            passed = error_l2 <= 0.02 * reference_l2
            record["flashqla"] = dict(
                error_l2=error_l2,
                reference_l2=reference_l2,
                relative_l2=(
                    error_l2 / reference_l2
                    if reference_l2
                    else (0.0 if error_l2 == 0 else None)
                ),
                max_abs_error=diff.abs().max().item(),
                reference_max_abs=ref64.abs().max().item(),
                reference_rms=reference_l2 / ref64.numel() ** 0.5,
                threshold=0.02,
                passed=passed,
            )
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
