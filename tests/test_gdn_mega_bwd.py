# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""PTO backward against CPU FP64 autograd; compile only the standalone kernel."""
from contextlib import nullcontext

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from csrc.kernel.gdn_mega_bwd.backward import backward
from gdn_mega_bwd_reference import recurrent_gdn_reference


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
        ([513], 8, 2, "weak"),
        ([7], 8, 2, "defaults"),
        ([128], 8, 2, "both"),
        ([256], 8, 2, "both"),
        ([129, 0, 131], 8, 2, "strided"),
        ([257], 8, 2, "replace"),
        ([513], 8, 2, "replace"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_backward(lengths, heads, key_heads, mode, dtype, npu_device):
    torch.manual_seed(42)
    T, N = sum(lengths), len(lengths)
    q = torch.randn(1, T, key_heads, 128) * 0.1
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
    if mode == "replace":
        k.zero_()
        k[..., 0] = 1
        beta.fill_(1)
        g.zero_()
    scale = 1.0 if mode in ("replace", "zero_beta") else 0.125
    # Use the exact rounded inputs supplied to the device kernel.
    inputs = [x.to(dtype).double().requires_grad_() for x in (q, k, v, g, beta, h0)]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    out, ht = recurrent_gdn_reference(*inputs, cu_seqlens=cu, scale=scale)
    do, dh = torch.randn_like(out), torch.randn_like(ht)
    if mode == "state":
        do.zero_()
    if mode in ("output", "defaults"):
        dh.zero_()
    do, dh = do.float(), dh.float()
    ((out * do).sum() + (ht * dh).sum()).backward()
    device_inputs = [x.detach().to(device=npu_device, dtype=dtype) for x in inputs]
    context = (
        torch.npu.stream(torch.npu.Stream(device=npu_device))
        if mode in ("defaults", "strided")
        else nullcontext()
    )
    with context:
        device_do, device_dh = do.to(npu_device), dh.to(npu_device)
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
            )
        assert guard.allocations == 7  # Six outputs and one reusable workspace.
        torch.npu.synchronize(npu_device)
    for name, x, result in zip(
        ("dq", "dk", "dv", "dg", "dbeta", "dh0"), inputs, actual
    ):
        torch.testing.assert_close(
            result.cpu().double(),
            x.grad,
            atol=2e-5,
            rtol=2e-4,
            msg=lambda m: f"{name}: {m}",
        )
    for actual_input, original in zip(device_inputs, inputs):
        torch.testing.assert_close(
            actual_input.cpu().double(), original, atol=0, rtol=0
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
