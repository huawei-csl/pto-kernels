# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
"""Numerical tests for the fused chunk-GDN mega kernel (``pto_gdn_mega``).

The mega kernel runs the whole chunk GatedDeltaNet forward pass in a single
launch. It is compared against a straightforward PyTorch reference that
implements the same chunked recurrence in fp32.

``pto_gdn_mega`` is caller-allocates: the helper below owns every output,
intermediate and per-core workspace. That allocation is expected to move into
the C++ host wrapper later, at which point ``_run_gdn_mega`` collapses into a
direct call.
"""

import pytest
import torch
import torch.nn.functional as F

from pto_kernels import pto_gdn_mega

C = 128  # chunk size, fixed by the kernel build
D = 128  # head dimension, fixed by the kernel build

MAX_ABS = 1e-2
MAX_RMSE_RATIO = 0.05

# (num_value_heads, num_key_heads) pairs the kernel dispatches.
SUPPORTED_HEAD_CONFIGS = [
    pytest.param(8, 2, id="H8-Hg2"),
    pytest.param(8, 8, id="H8-Hg8"),
    pytest.param(16, 4, id="H16-Hg4"),
    pytest.param(16, 16, id="H16-Hg16"),
    pytest.param(32, 8, id="H32-Hg8"),
]


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _inv_tril(A: torch.Tensor) -> torch.Tensor:
    """Inverse of (I + A) for a strictly lower-triangular A, computed in place."""
    assert A.shape[-2] == A.shape[-1]
    chunk_size = A.shape[-1]
    for i in range(1, chunk_size):
        row = A[..., i, :i].clone()
        sub = A[..., :i, :i].clone()
        A[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return A + torch.eye(chunk_size, dtype=A.dtype, device=A.device)


def _reference(query, key, value, g, beta, initial_state=None):
    """Chunked GatedDeltaNet forward in fp32.

    All inputs use [B, T, H, D] layout with B == 1, except g and beta which are
    [B, T, H]. Returns (out [B, T, H, D], final_state [B, H, D, D]).
    """
    chunk_size = C
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_len = seq_len + pad_size

    scale = 1 / (k_head_dim**0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    attn = _inv_tril(attn)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        state = torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, device=query.device
        ).to(value)
    else:
        state = initial_state.to(value)

    core_attn_out = torch.zeros_like(value)
    strict_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(tot_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(
            strict_mask, 0
        )
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    return core_attn_out, state


# ---------------------------------------------------------------------------
# Kernel driver (caller-allocates)
# ---------------------------------------------------------------------------


def _total_chunks(cu_seqlens) -> int:
    cu = cu_seqlens.cpu().tolist()
    return sum((e - s + C - 1) // C for s, e in zip(cu, cu[1:]))


def _run_gdn_mega(q, k, v, g, beta, cu_seqlens, initial_state=None):
    """Allocate every buffer the kernel needs, launch it, return (out, final_state)."""
    device = q.device
    _, total_tokens, _, head_dim = q.shape
    num_value_heads = v.shape[-2]

    q, k, v, beta = (t.half() for t in (q, k, v, beta))
    cu32 = cu_seqlens.to(torch.int32)
    num_sequences = cu32.numel() - 1
    num_chunks = _total_chunks(cu32)
    num_matrices = num_chunks * num_value_heads

    mask_lower = torch.tril(torch.ones(C, C, device=device), diagonal=-1).float()
    mask_full = torch.tril(torch.ones(C, C, device=device), diagonal=0).float()
    minus_identity = torch.zeros(C, C, device=device, dtype=torch.float16)
    minus_identity.fill_diagonal_(-1)

    g_sum = torch.empty_like(g, dtype=torch.float32)
    g_t = torch.empty(num_value_heads, total_tokens, device=device, dtype=torch.float32)
    beta_t = torch.empty(
        num_value_heads, total_tokens, device=device, dtype=torch.float16
    )

    A = torch.zeros(
        1, total_tokens, num_value_heads, C, device=device, dtype=torch.float16
    )
    A_inv_f32 = torch.zeros_like(A, dtype=torch.float32)
    A_inv = torch.zeros_like(A)

    w = torch.empty_like(v)
    u = torch.empty_like(v)
    h = torch.zeros(
        num_chunks * num_value_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float16,
    )
    v_new = torch.empty_like(v)
    final_state = torch.zeros(
        num_sequences * num_value_heads,
        head_dim,
        head_dim,
        device=device,
        dtype=torch.float16,
    )

    has_initial_state = initial_state is not None
    initial_state = initial_state.half() if has_initial_state else final_state

    block_dim = max(
        1, int(getattr(torch.npu.get_device_properties(device), "cube_core_num", 24))
    )
    kkt_workspace = torch.zeros(block_dim * 2, C, C, device=device, dtype=torch.float16)
    wy_workspace_a1 = torch.zeros(block_dim, C, C, device=device, dtype=torch.float16)
    wy_workspace_a2 = torch.zeros_like(wy_workspace_a1)
    h_workspace = torch.zeros(
        block_dim * 4, head_dim, head_dim, device=device, dtype=torch.float16
    )
    o_workspace_qk = torch.zeros(block_dim, C, C, device=device, dtype=torch.float16)
    o_workspace_qs = torch.zeros(
        block_dim, C, head_dim, device=device, dtype=torch.float16
    )
    o_workspace_gated = torch.zeros_like(o_workspace_qk)
    out = torch.empty_like(v)

    pto_gdn_mega(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        mask_lower=mask_lower,
        mask_full=mask_full,
        minus_identity=minus_identity,
        cu_seqlens=cu32,
        out=out,
        g_sum=g_sum,
        g_t=g_t,
        beta_t=beta_t,
        A=A,
        A_inv_f32=A_inv_f32,
        A_inv=A_inv,
        w=w,
        u=u,
        s=h,
        v_new=v_new,
        final_state=final_state,
        initial_state=initial_state,
        has_initial_state=has_initial_state,
        kkt_workspace=kkt_workspace,
        wy_workspace_a1=wy_workspace_a1,
        wy_workspace_a2=wy_workspace_a2,
        h_workspace=h_workspace,
        o_workspace_qk=o_workspace_qk,
        o_workspace_qs=o_workspace_qs,
        o_workspace_gated=o_workspace_gated,
        block_dim=block_dim,
        batch_size=num_sequences,
        seq_len=total_tokens,
        total_tokens=total_tokens,
        num_matrices=num_matrices,
    )

    scale = head_dim**-0.5
    final_state_out = final_state.view(
        num_sequences, num_value_heads, head_dim, head_dim
    ).float()
    return out.float() * scale, final_state_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual.float().cpu() - expected.float().cpu()).abs()
    max_abs = diff.max().item()
    if max_abs <= MAX_ABS:
        return
    rmse = torch.sqrt((diff.flatten() ** 2).mean()).item()
    base = torch.sqrt((expected.float().cpu().flatten() ** 2).mean()).item()
    ratio = rmse / max(base, 1e-8)
    assert (
        ratio < MAX_RMSE_RATIO
    ), f"{name}: max_abs={max_abs:.6f} rmse_ratio={ratio:.6f}"


def _make_inputs(seq_len, num_value_heads, num_key_heads, seed=0):
    """Build CPU inputs. q/k are L2-normalised so that the reference's
    128-step triangular inverse stays well conditioned."""
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(1, seq_len, num_key_heads, D), p=2, dim=-1).half()
    k = F.normalize(torch.randn(1, seq_len, num_key_heads, D), p=2, dim=-1).half()
    v = torch.randn(1, seq_len, num_value_heads, D, dtype=torch.float16)
    g = F.logsigmoid(torch.randn(1, seq_len, num_value_heads, dtype=torch.float32))
    beta = torch.rand(1, seq_len, num_value_heads, dtype=torch.float16)
    return q, k, v, g, beta


def _expand_kv_heads(q, k, num_value_heads):
    """Repeat key heads so the reference sees one key head per value head."""
    num_key_heads = q.shape[2]
    if num_key_heads == num_value_heads:
        return q, k
    group = num_value_heads // num_key_heads
    return q.repeat_interleave(group, dim=2), k.repeat_interleave(group, dim=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gdn_mega_no_initial_state(npu_device):
    seq_len = 256
    num_value_heads = num_key_heads = 16
    q, k, v, g, beta = _make_inputs(seq_len, num_value_heads, num_key_heads)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=npu_device)

    actual, actual_final = _run_gdn_mega(
        q.to(npu_device),
        k.to(npu_device),
        v.to(npu_device),
        g.to(npu_device),
        beta.to(npu_device),
        cu_seqlens,
    )
    torch.npu.synchronize()

    q_ref, k_ref = _expand_kv_heads(q, k, num_value_heads)
    expected, expected_final = _reference(q_ref, k_ref, v, g, beta)

    _assert_close("out", actual, expected)
    _assert_close("final_state", actual_final, expected_final)


@pytest.mark.parametrize("num_value_heads,num_key_heads", SUPPORTED_HEAD_CONFIGS)
def test_gdn_mega_head_configs(num_value_heads, num_key_heads, npu_device):
    seq_len = 256
    q, k, v, g, beta = _make_inputs(seq_len, num_value_heads, num_key_heads)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=npu_device)

    actual, _ = _run_gdn_mega(
        q.to(npu_device),
        k.to(npu_device),
        v.to(npu_device),
        g.to(npu_device),
        beta.to(npu_device),
        cu_seqlens,
    )
    torch.npu.synchronize()

    q_ref, k_ref = _expand_kv_heads(q, k, num_value_heads)
    expected, _ = _reference(q_ref, k_ref, v, g, beta)

    _assert_close("out", actual, expected)


def test_gdn_mega_with_initial_state(npu_device):
    seq_len = 256
    num_value_heads = num_key_heads = 16
    q, k, v, g, beta = _make_inputs(seq_len, num_value_heads, num_key_heads)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=npu_device)
    initial_state = torch.randn(1, num_value_heads, D, D, dtype=torch.float16) * 0.1

    actual, actual_final = _run_gdn_mega(
        q.to(npu_device),
        k.to(npu_device),
        v.to(npu_device),
        g.to(npu_device),
        beta.to(npu_device),
        cu_seqlens,
        initial_state=initial_state.to(npu_device),
    )
    torch.npu.synchronize()

    q_ref, k_ref = _expand_kv_heads(q, k, num_value_heads)
    expected, expected_final = _reference(
        q_ref, k_ref, v, g, beta, initial_state=initial_state
    )

    _assert_close("out", actual, expected)
    _assert_close("final_state", actual_final, expected_final)
