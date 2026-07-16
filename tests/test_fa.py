# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import math

import pytest
import torch

from pto_kernels import pto_fa


HEAD_SIZE = 128
NON_CAUSAL_ATOL = 1e-4
CAUSAL_ATOL = 5e-4


def _inputs(device, batch, num_q_heads, num_kv_heads, s0, s1, seed=0, f=1.0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, num_q_heads, s0, HEAD_SIZE, generator=generator).half()
    k = torch.randn(batch, num_kv_heads, s1, HEAD_SIZE, generator=generator).half()
    v = torch.randn(batch, num_kv_heads, s1, HEAD_SIZE, generator=generator).half()
    return f * q.to(device), f * k.to(device), f * v.to(device)


def _reference(q, k, v, causal):
    q = q.float().cpu()
    k = k.float().cpu()
    v = v.float().cpu()
    groups = q.size(1) // k.size(1)
    if groups != 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_SIZE)
    if causal:
        query_positions = torch.arange(q.size(2)).view(-1, 1)
        key_positions = torch.arange(k.size(2)).view(1, -1)
        scores.masked_fill_(key_positions > query_positions, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


CASES = [
    pytest.param(1, 1, 1, 128, 512, id="minimum"),
    pytest.param(1, 1, 1, 256, 1024, id="multiple-tiles"),
    pytest.param(1, 4, 4, 512, 512, id="mha"),
    pytest.param(1, 4, 1, 1024, 1024, id="mqa"),
    pytest.param(1, 4, 2, 256, 512, id="gqa"),
    pytest.param(2, 4, 2, 128, 1024, id="batch-gqa"),
]


@pytest.mark.parametrize("causal", [False, True], ids=["non-causal", "causal"])
@pytest.mark.parametrize("batch,nq,nkv,s0,s1", CASES)
def test_pto_fa_matches_reference(npu_device, causal, batch, nq, nkv, s0, s1):
    q, k, v = _inputs(npu_device, batch, nq, nkv, s0, s1)

    actual = pto_fa(q, k, v, causal=causal)
    expected = _reference(q, k, v, causal)

    assert actual.shape == q.shape
    assert actual.dtype == torch.float32
    assert actual.device == q.device
    atol = CAUSAL_ATOL if causal else NON_CAUSAL_ATOL
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=atol)


@pytest.mark.parametrize("qk_preload", [2, 4, 8])
def test_pto_fa_qk_preload(npu_device, qk_preload):
    q, k, v = _inputs(npu_device, 1, 1, 1, 128, 1024)
    actual = pto_fa(q, k, v, qk_preload=qk_preload)
    expected = _reference(q, k, v, causal=False)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=NON_CAUSAL_ATOL)


def test_pto_fa_back_to_back_causal_launches(npu_device):
    q, k, v = _inputs(npu_device, 1, 2, 1, 256, 512)
    outputs = [pto_fa(q, k, v, causal=True) for _ in range(4)]
    torch.npu.synchronize()
    expected = _reference(q, k, v, causal=True)
    for output in outputs:
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=CAUSAL_ATOL)


def test_pto_fa_uses_current_stream(npu_device):
    q, k, v = _inputs(npu_device, 1, 1, 1, 128, 512)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        actual = pto_fa(q, k, v)
    stream.synchronize()
    expected = _reference(q, k, v, causal=False)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=NON_CAUSAL_ATOL)


def test_pto_fa_zero_queries_and_keys_average_values(npu_device):
    q = torch.zeros(1, 1, 128, HEAD_SIZE, dtype=torch.float16, device=npu_device)
    k = torch.zeros(1, 1, 512, HEAD_SIZE, dtype=torch.float16, device=npu_device)
    v = torch.randn(1, 1, 512, HEAD_SIZE, dtype=torch.float16, device=npu_device)
    expected = v.float().cpu().mean(dim=2, keepdim=True).expand_as(q.float().cpu())
    actual = pto_fa(q, k, v)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=NON_CAUSAL_ATOL)


def test_pto_fa_validates_inputs(npu_device):
    q, k, v = _inputs(npu_device, 1, 4, 2, 128, 512)

    with pytest.raises(RuntimeError, match="4D BNSD"):
        pto_fa(q.squeeze(0), k, v)
    with pytest.raises(RuntimeError, match="dtype fp16"):
        pto_fa(q.float(), k, v)
    with pytest.raises(RuntimeError, match="head dimension must be 128"):
        pto_fa(
            q[..., :64].contiguous(), k[..., :64].contiguous(), v[..., :64].contiguous()
        )
    with pytest.raises(RuntimeError, match="S0 must be a multiple of 128"):
        pto_fa(q[:, :, :64].contiguous(), k, v)
    with pytest.raises(RuntimeError, match="S1 must be a multiple of 512"):
        pto_fa(q, k[:, :, :256].contiguous(), v[:, :, :256].contiguous())
    with pytest.raises(RuntimeError, match="same shape"):
        pto_fa(q, k, v[:, :, :256].contiguous())
    with pytest.raises(RuntimeError, match="must divide"):
        bad_k = k[:, :2]
        bad_v = v[:, :2]
        pto_fa(q[:, :3].contiguous(), bad_k, bad_v)
    with pytest.raises(RuntimeError, match="contiguous"):
        pto_fa(q.transpose(3, 2), k.transpose(3, 2), v.transpose(3, 2))
    with pytest.raises(RuntimeError, match="qk_preload"):
        pto_fa(q, k, v, qk_preload=1)
    with pytest.raises(RuntimeError, match="qk_preload"):
        pto_fa(q, k, v, qk_preload=9)


def test_pto_fa_rejects_cpu_tensors():
    q = torch.empty(1, 1, 128, HEAD_SIZE, dtype=torch.float16)
    k = torch.empty(1, 1, 512, HEAD_SIZE, dtype=torch.float16)
    v = torch.empty_like(k)
    with pytest.raises(RuntimeError, match="must be on NPU"):
        pto_fa(q, k, v)
