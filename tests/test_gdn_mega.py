# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""GDN mega tests, independent FP64 oracle, fixed cases and caller allocation.

Run all tests with `python -m pytest tests/test_gdn_mega.py -q` in CANN.
The benchmark imports the same input/ABI helpers and acceptance policy.
The FP32 chunked reference is retained only as an additional cross-check;
all 178 adversarial audit cases use the independent FP64 token recurrence.
"""
from functools import partial
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import tempfile

import pytest
import torch
import torch.nn.functional as F
from pto_kernels import pto_gdn_mega


@pytest.fixture(scope="session", autouse=True)
def verify_selected_kernel():
    torch.set_num_threads(4)
    expected = os.environ.get("GDN_EXPECTED_LIBRARY_SHA256")
    if expected:
        libraries = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
            if line.split()[-1].endswith("/libno_workspace_kernel.so")
        }
        assert len(libraries) == 1
        assert (
            hashlib.sha256(Path(libraries.pop()).read_bytes()).hexdigest() == expected
        )


# User-approved absolute floor; keep the relative bound fixed.
ATOL = 1e-4
RTOL = 1e-3
MAX_RMSE_RATIO = 0.005
POLICY = {
    "atol": ATOL,
    "rtol": RTOL,
    "max_relative_rmse": MAX_RMSE_RATIO,
    "criterion": "every element: abs(error) <= atol + rtol * abs(reference), AND relative RMSE",
}


def recurrent_reference(q, k, v, g, beta, initial_state=None, cu_seqlens=None):
    """Return raw (unscaled) output and final state from exact supplied inputs."""
    q, k, v, g, beta = [x.detach().cpu().double() for x in (q, k, v, g, beta)]
    _, tokens, heads, dim = v.shape
    group = heads // k.shape[2]
    q, k = q.repeat_interleave(group, 2), k.repeat_interleave(group, 2)
    boundaries = [0, tokens] if cu_seqlens is None else cu_seqlens.cpu().tolist()
    output = torch.empty_like(v)
    decay = g.exp()
    states = []
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (
            torch.zeros(heads, dim, dim, dtype=torch.float64)
            if initial_state is None
            else initial_state[seq].detach().cpu().double().clone()
        )
        # Reuse one state-sized scratch tensor; keep the same FP64 reduction
        # and update order as the literal token recurrence.
        scratch = torch.empty_like(state)
        for token in range(start, end):
            state.mul_(decay[0, token, :, None, None])
            torch.mul(k[0, token, :, :, None], state, out=scratch)
            delta = beta[0, token, :, None] * (v[0, token] - scratch.sum(dim=1))
            torch.mul(k[0, token, :, :, None], delta[:, None, :], out=scratch)
            state.add_(scratch)
            torch.mul(q[0, token, :, :, None], state, out=scratch)
            torch.sum(scratch, dim=1, out=output[0, token])
        states.append(state)
    return output, torch.stack(states)


def tensor_fingerprint(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        if tensor is None:
            digest.update(b"None")
        else:
            value = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            if value.numel():
                digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def shared_reference(path, identity, compute):
    """Cache only CPU oracle tensors; reject mismatched identities or bytes."""
    if path is not None and path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (
            saved["identity"] != identity
            or tensor_fingerprint(saved["tensors"]) != saved["sha256"]
        ):
            raise RuntimeError("FP64 reference cache identity or content mismatch")
        return tuple(saved["tensors"])
    result = compute()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(
                {
                    "identity": identity,
                    "tensors": result,
                    "sha256": tensor_fingerprint(result),
                },
                temporary_path,
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return result


def cached_recurrent_reference(*inputs):
    """Opt-in cache keyed by exact inputs, oracle function source and PyTorch.

    No device results or pass/fail decisions are cached. Unrelated test edits
    do not invalidate the oracle, but changes to its implementation do.
    """
    directory = os.environ.get("GDN_REFERENCE_CACHE")
    if not directory:
        return recurrent_reference(*inputs)
    identity = {
        "inputs": tensor_fingerprint(inputs),
        "torch_version": str(torch.__version__),
        "oracle_sha256": hashlib.sha256(
            inspect.getsource(recurrent_reference).encode()
        ).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return shared_reference(
        Path(directory) / f"{key}.pt", identity, partial(recurrent_reference, *inputs)
    )


def error_metrics(actual, expected, *, include_percentiles=False):
    """No broadcasting, rescaling, fitting, or outlier removal is allowed."""
    assert (
        actual.shape == expected.shape
    ), f"shape mismatch: {actual.shape} != {expected.shape}"
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {
            "passed": False,
            "finite": False,
            "elements": actual.numel(),
            "actual_nonfinite": int((~torch.isfinite(actual)).sum()),
            "reference_nonfinite": int((~torch.isfinite(expected)).sum()),
            "reference_abs_max": (
                expected.abs().max().item()
                if bool(torch.isfinite(expected).all())
                else None
            ),
        }
    diff = (actual - expected).abs()
    if not diff.numel():
        return {
            "passed": True,
            "finite": True,
            "elements": 0,
            "max_abs": 0.0,
            "relative_rmse": 0.0,
            "pointwise_violations": 0,
        }
    allowed = ATOL + RTOL * expected.abs()
    relative_rmse = (
        (diff.square().mean() / expected.square().mean().clamp_min(1e-16)).sqrt().item()
    )
    violations = int((diff > allowed).sum())
    max_index = int(diff.flatten().argmax())
    tolerance_ratio = diff / allowed
    worst_index = int(tolerance_ratio.flatten().argmax())
    # Exact nearest-rank percentiles. torch.quantile rejects tensors above 2**24
    # elements (already exceeded by T=32768, H=8); never subsample outliers.
    percentiles = (
        {
            name: diff.flatten()
            .kthvalue(max(1, math.ceil(p * diff.numel())))
            .values.item()
            for name, p in (("p50_abs", 0.5), ("p99_abs", 0.99), ("p999_abs", 0.999))
        }
        if include_percentiles
        else {}
    )
    centered = (expected - expected.mean()).square().sum()
    r_squared = 1 - diff.square().sum().item() / max(centered.item(), 1e-30)
    return {
        "passed": violations == 0 and relative_rmse < MAX_RMSE_RATIO,
        "finite": True,
        "elements": diff.numel(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        **percentiles,
        "required_atol_at_fixed_rtol": (diff - RTOL * expected.abs())
        .clamp_min_(0)
        .max()
        .item(),
        "relative_rmse": relative_rmse,
        "r_squared": r_squared,
        "pointwise_violations": violations,
        "max_tolerance_ratio": tolerance_ratio.flatten()[worst_index].item(),
        "worst_tolerance_flat_index": worst_index,
        "actual_at_worst_tolerance": actual.flatten()[worst_index].item(),
        "reference_at_worst_tolerance": expected.flatten()[worst_index].item(),
        "abs_error_at_worst_tolerance": diff.flatten()[worst_index].item(),
        "allowed_at_worst_tolerance": allowed.flatten()[worst_index].item(),
        "reference_abs_max": expected.abs().max().item(),
        "max_error_flat_index": max_index,
        "actual_at_max_error": actual.flatten()[max_index].item(),
        "reference_at_max_error": expected.flatten()[max_index].item(),
        "abs_gt_1e-4": int((diff > 1e-4).sum()),
        "abs_gt_1e-3": int((diff > 1e-3).sum()),
    }


def assert_numerically_close(name, actual, expected):
    metrics = error_metrics(actual, expected)
    assert metrics["passed"], f"{name}: policy={POLICY}; errors={metrics}"


# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------


C = 128  # chunk size, fixed by the kernel build
D = 128  # head dimension, fixed by the kernel build

# (num_value_heads, num_key_heads) pairs the kernel dispatches.
SUPPORTED_HEAD_CONFIGS = [
    pytest.param(8, 2, id="H8-Hg2"),
    pytest.param(8, 8, id="H8-Hg8"),
    pytest.param(12, 3, id="H12-Hg3"),
    pytest.param(16, 4, id="H16-Hg4"),
    pytest.param(16, 16, id="H16-Hg16"),
    pytest.param(24, 6, id="H24-Hg6"),
    pytest.param(32, 8, id="H32-Hg8"),
    pytest.param(48, 12, id="H48-Hg12"),
    pytest.param(64, 16, id="H64-Hg16"),
]


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _inv_tril(A: torch.Tensor) -> torch.Tensor:
    """Inverse of (I - A) for a strictly lower-triangular A, computed in place."""
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


def _prepare_gdn_mega(q, k, v, g, beta, cu_seqlens, initial_state=None):
    """Allocate once; return keyword arguments reusable across kernel launches."""
    device = q.device
    _, total_tokens, _, head_dim = q.shape
    num_value_heads = v.shape[-2]

    q, k, v, beta = (t.half() for t in (q, k, v, beta))
    cu32 = cu_seqlens.to(torch.int32)
    num_sequences = cu32.numel() - 1
    num_chunks = _total_chunks(cu32)
    num_matrices = num_chunks * num_value_heads

    # CPU-created constants also work when msprof filters out unrelated NPU ops.
    mask_lower = torch.tril(torch.ones(C, C), diagonal=-1).to(device)
    mask_full = torch.tril(torch.ones(C, C), diagonal=0).to(device)
    minus_identity = (-torch.eye(C, dtype=torch.float16)).to(device)

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

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "mask_lower": mask_lower,
        "mask_full": mask_full,
        "minus_identity": minus_identity,
        "cu_seqlens": cu32,
        "out": out,
        "g_sum": g_sum,
        "g_t": g_t,
        "beta_t": beta_t,
        "A": A,
        "A_inv_f32": A_inv_f32,
        "A_inv": A_inv,
        "w": w,
        "u": u,
        "s": h,
        "v_new": v_new,
        "final_state": final_state,
        "initial_state": initial_state,
        "has_initial_state": has_initial_state,
        "kkt_workspace": kkt_workspace,
        "wy_workspace_a1": wy_workspace_a1,
        "wy_workspace_a2": wy_workspace_a2,
        "h_workspace": h_workspace,
        "o_workspace_qk": o_workspace_qk,
        "o_workspace_qs": o_workspace_qs,
        "o_workspace_gated": o_workspace_gated,
        "block_dim": block_dim,
        "batch_size": num_sequences,
        "seq_len": total_tokens,
        "total_tokens": total_tokens,
        "num_matrices": num_matrices,
    }


def _run_gdn_mega(q, k, v, g, beta, cu_seqlens, initial_state=None):
    args = _prepare_gdn_mega(q, k, v, g, beta, cu_seqlens, initial_state)
    pto_gdn_mega(**args)
    head_dim = q.shape[-1]
    final_state_out = (
        args["final_state"]
        .view(args["batch_size"], v.shape[-2], head_dim, head_dim)
        .float()
    )
    return args["out"].float() * head_dim**-0.5, final_state_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert_numerically_close(name, actual, expected)
    if name.startswith("out"):
        # Helpers historically return attention-scaled output. Also enforce the
        # absolute tolerance in the raw units of the public kernel's output.
        assert_numerically_close(
            name + " (raw units)", actual.double() * D**0.5, expected.double() * D**0.5
        )


def _make_inputs(seq_len, num_value_heads, num_key_heads, seed=0):
    """Build one ordinary normalized-input distribution, not an accuracy bound.

    Adversarial keys, amplitudes and independent references live in
    the fixed audit cases below; normalized keys alone do not bound conditioning.
    """
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


@pytest.mark.parametrize(
    "lengths", [(1,), (127,), (129,), (257,), (1024,), (63, 130, 257)]
)
@pytest.mark.parametrize("gate_scale", [1.0, 0.01, 4.0])
def test_gdn_mega_tails_and_memory(lengths, gate_scale, npu_device):
    """Tails, packed sequence boundaries, and gates retaining recurrent memory."""
    heads, key_heads = 16, 4
    inputs = list(_make_inputs(sum(lengths), heads, key_heads, seed=42))
    inputs[3] *= gate_scale
    initial = torch.randn(len(lengths), heads, D, D, dtype=torch.float16) * 0.1
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    expected_parts, expected_states = [], []
    for i, (start, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        q, k, v, g, beta = [x[:, start:end] for x in inputs]
        q, k = _expand_kv_heads(q, k, heads)
        out, state = _reference(q, k, v, g, beta, initial[i : i + 1])
        expected_parts.append(out)
        expected_states.append(state)
    expected = torch.cat(expected_parts, dim=1)
    expected_state = torch.cat(expected_states, dim=0)
    args = _prepare_gdn_mega(
        *(t.to(npu_device) for t in inputs), cu.to(npu_device), initial.to(npu_device)
    )
    # Short packed tails exercise unequal vector-subblock work and scalar
    # metadata publication; repeat the weak-decay regression more aggressively.
    repeats = 50 if len(lengths) > 1 and gate_scale == 0.01 else 5
    for _ in range(repeats):
        pto_gdn_mega(**args)
        torch.npu.synchronize()
        _assert_close("out", args["out"].float() * D**-0.5, expected)
        _assert_close(
            "state", args["final_state"].view_as(expected_state), expected_state
        )


@pytest.mark.parametrize("beta_value", [0.0, 0.02])
def test_gdn_mega_correlated_keys(beta_value, npu_device):
    """Identity and high-row-norm inverses in the parallel chunk solve."""
    length, heads, key_heads = 256, 16, 4
    q, k, v, g, beta = _make_inputs(length, heads, key_heads, seed=17)
    k = k[:, :1].expand_as(k).contiguous()
    g.zero_()
    beta.fill_(beta_value)
    initial = torch.randn(1, heads, D, D, dtype=torch.float16) * 0.1
    qr, kr = _expand_kv_heads(q, k, heads)
    expected, expected_state = _reference(qr, kr, v, g, beta, initial)
    args = _prepare_gdn_mega(
        *(t.to(npu_device) for t in (q, k, v, g, beta)),
        torch.tensor([0, length], device=npu_device, dtype=torch.int32),
        initial.to(npu_device),
    )
    for _ in range(5):
        pto_gdn_mega(**args)
        torch.npu.synchronize()
        pairs = [
            (args["out"].cpu().float() * D**-0.5, expected),
            (args["final_state"].cpu().view_as(expected_state).float(), expected_state),
        ]
        for name, (actual, reference) in zip(("out", "final_state"), pairs):
            _assert_close(name, actual, reference)


def test_gdn_mega_changing_decay(npu_device):
    """Alternate finite and half-zero chunk decays with a nonzero state."""
    inputs = list(_make_inputs(512, 16, 4, seed=73))
    for chunk, gate_scale in enumerate([4.0, 0.01, 4.0, 0.01]):
        inputs[3][:, chunk * C : (chunk + 1) * C] *= gate_scale
    initial = torch.randn(1, 16, D, D, dtype=torch.float16) * 0.1
    q, k = _expand_kv_heads(inputs[0], inputs[1], 16)
    expected, expected_state = _reference(
        q, k, inputs[2], inputs[3], inputs[4], initial_state=initial
    )
    args = _prepare_gdn_mega(
        *(t.to(npu_device) for t in inputs),
        torch.tensor([0, 512], dtype=torch.int32, device=npu_device),
        initial.to(npu_device),
    )
    for _ in range(5):
        pto_gdn_mega(**args)
        torch.npu.synchronize()
        _assert_close("out", args["out"].cpu().float() * D**-0.5, expected)
        _assert_close(
            "state", args["final_state"].cpu().view_as(expected_state), expected_state
        )


# Fixed before evaluating the kernel: no tuning the oracle or tolerances to results.
# fmt: off
STANDARD_CASES = [
    {"name": "ordinary_seed2027", "seed": 2027, "lengths": (257,), "heads": 16, "key_heads": 4},
    {"name": "weak_seed119", "seed": 119, "lengths": (513,), "heads": 16, "key_heads": 4, "gate_scale": .01},
    {"name": "no_decay", "seed": 911, "lengths": (257,), "heads": 8, "key_heads": 8, "gate_scale": 0.},
    {"name": "zero_state_tail", "seed": 1709, "lengths": (129,), "heads": 12, "key_heads": 3, "initial": False},
    {"name": "packed_empty", "seed": 31, "lengths": (0, 1, 127, 0, 129, 0), "heads": 16, "key_heads": 4, "gate_scale": .01},
    {"name": "packed_empty_zero_state", "seed": 313, "lengths": (0, 1, 129, 0), "heads": 8, "key_heads": 2, "initial": False},
    {"name": "large_values_and_state", "seed": 191, "lengths": (257,), "heads": 8, "key_heads": 2, "amplitude": 32.},
    {"name": "tiny_values_and_state", "seed": 509, "lengths": (257,), "heads": 8, "key_heads": 2, "amplitude": 1e-3},
    {"name": "zero_beta_no_decay", "seed": 809, "lengths": (257,), "heads": 8, "key_heads": 2, "gate_scale": 0., "beta": 0.},
    {"name": "zero_query", "seed": 811, "lengths": (257,), "heads": 8, "key_heads": 2, "zero_query": True},
    {"name": "zero_values_and_state", "seed": 821, "lengths": (257,), "heads": 8, "key_heads": 2, "amplitude": 0.},
    {"name": "correlated_beta002", "seed": 823, "lengths": (256,), "heads": 8, "key_heads": 2,
     "correlated": True, "gate_scale": 0., "beta": .02},
    {"name": "correlated_beta05", "seed": 827, "lengths": (256,), "heads": 8, "key_heads": 2,
     "correlated": True, "gate_scale": 0., "beta": .5},
    {"name": "correlated_beta1", "seed": 829, "lengths": (256,), "heads": 8, "key_heads": 2,
     "correlated": True, "gate_scale": 0., "beta": 1.},
    {"name": "mixed_decay", "seed": 839, "lengths": (513,), "heads": 8, "key_heads": 2, "mixed": True},
    {"name": "equal_heads64", "seed": 853, "lengths": (129,), "heads": 64, "key_heads": 64},
    {"name": "single_key_head", "seed": 857, "lengths": (129,), "heads": 24, "key_heads": 1},
    {"name": "keys_norm2_stable_beta", "seed": 859, "lengths": (257,), "heads": 8, "key_heads": 2, "key_scale": 2., "beta": .1},
    {"name": "long_8k_weak", "seed": 863, "lengths": (8192,), "heads": 8, "key_heads": 2, "gate_scale": .01},
    {"name": "long_32k", "seed": 877, "lengths": (32768,), "heads": 8, "key_heads": 2},
    {"name": "correlated_random_beta09", "seed": 883, "lengths": (513,), "heads": 8, "key_heads": 2,
         "correlated_random": True, "gate_scale": 0., "beta": .9, "amplitude": 32.},
    {"name": "nearly_correlated_beta1", "seed": 887, "lengths": (257,), "heads": 16, "key_heads": 4,
         "nearly_correlated": True, "gate_scale": 0., "beta": 1., "amplitude": 32.},
    {"name": "positive_log_gates", "seed": 907, "lengths": (257,), "heads": 8, "key_heads": 2, "gate_scale": -.01},
    {"name": "long_16k_no_decay", "seed": 919, "lengths": (16384,), "heads": 8, "key_heads": 2, "gate_scale": 0.},
    {"name": "delayed_nearly_correlated", "seed": 929, "lengths": (385,), "heads": 8, "key_heads": 2,
         "nearly_correlated": True, "correlated_after": 128, "gate_scale": 0., "beta": 1., "amplitude": 32.},
    {"name": "strong_correlated_decay", "seed": 937, "lengths": (257,), "heads": 8, "key_heads": 2,
         "nearly_correlated": True, "gate_scale": .01, "beta": .9, "amplitude": 32.},
    # Distribution families from FLA test_gdn.py at 12cb9d4, adapted to our
    # FP16 input/state ABI and D128. Keep our independent FP64 oracle/policy.
    {"name": "fla_positive_no_decay", "seed": 42, "lengths": (1024,), "heads": 8, "key_heads": 2,
         "fla": "positive", "gate_mask": 1., "initial": False},
    {"name": "fla_positive_weak", "seed": 43, "lengths": (1500,), "heads": 8, "key_heads": 2,
         "fla": "positive", "gate_normalizer": 10., "initial": False},
    {"name": "fla_positive_strong", "seed": 44, "lengths": (257,), "heads": 8, "key_heads": 2,
         "fla": "positive", "gate_normalizer": .01, "initial": False},
    {"name": "fla_positive_masked", "seed": 45, "lengths": (513,), "heads": 8, "key_heads": 2,
         "fla": "positive", "gate_mask": .5, "initial": False},
    {"name": "fla_gaussian_initial", "seed": 46, "lengths": (1024,), "heads": 8, "key_heads": 2,
         "fla": "gaussian", "gate_normalizer": 10., "state_scale": 1.},
    {"name": "fla_negative_eigenvalues", "seed": 47, "lengths": (513,), "heads": 8, "key_heads": 2,
         "fla": "positive", "beta_logits": True, "negative_eigenvalues": True, "initial": False},
    {"name": "fla_gate_activation", "seed": 48, "lengths": (513,), "heads": 8, "key_heads": 2,
         "fla": "positive", "gate_activation": True, "initial": False},
    {"name": "fla_gaussian_logits", "seed": 49, "lengths": (513,), "heads": 8, "key_heads": 2,
         "fla": "gaussian", "beta_logits": True, "gate_logits": True, "state_scale": 1.},
    {"name": "fla_positive_strong_tail75", "seed": 51, "lengths": (75,), "heads": 8, "key_heads": 2,
         "fla": "positive", "gate_normalizer": .01, "initial": False},
    {"name": "strong_decay_tail92", "seed": 53, "lengths": (92,), "heads": 8, "key_heads": 2, "gate_scale": 4.},
]

# Fixed before evaluation. All critical cases obey the user's beta range [0,1].
# Named groups allow standalone audit subsets; ordinary pytest runs all groups.
CRITICAL_CASES = [
    {"name": f'critical_boundary_{rate}_{amplitude}', "seed": 3001, "lengths": (513,), "heads": 8, "key_heads": 2,
         "correlated": True, "beta": rate, "gate_scale": 0., "amplitude": amplitude, "state_scale": 1.}
    for rate in (.499755859375, .5, .50048828125, 1.) for amplitude in (1., 32.)
] + [
    {"name": f'critical_tiny_gate_{gate}_{length}', "seed": 3007, "lengths": (length,), "heads": 8, "key_heads": 2,
         "correlated": True, "beta": 1., "constant_log_gate": gate, "state_scale": 1.}
    for gate in (-1e-8, -1e-7, -1e-6) for length in (4097, 32768)
] + [
    {"name": f'critical_nearly_correlated_{seed}_{amplitude}', "seed": seed, "lengths": (1025,), "heads": 8, "key_heads": 2,
         "nearly_correlated": True, "beta": 1., "constant_log_gate": -1e-6, "amplitude": amplitude, "state_scale": 1.}
    for seed in (3011, 3019, 3023) for amplitude in (1., 32., 128.)
] + [
    {"name": f'critical_gaussian_{rate}_{gate}', "seed": 3037, "lengths": (4097,), "heads": 8, "key_heads": 2,
         "beta": rate, "constant_log_gate": gate, "state_scale": 1.}
    for rate in (0., .5, 1.) for gate in (0., -1e-6, -80.)
] + [
    {"name": f'critical_group_{heads}_{keys}', "seed": 3041, "lengths": (1025,), "heads": heads, "key_heads": keys,
         "beta": 1., "constant_log_gate": -1e-6, "state_scale": 1.}
    for heads, keys in ((24, 8), (48, 16), (64, 16))
] + [
    {"name": f'critical_positive_{seed}', "seed": seed, "lengths": (4097,), "heads": 8, "key_heads": 2,
         "fla": 'positive', "beta": 1., "constant_log_gate": -1e-6, "state_scale": 1.}
    for seed in (3049, 3061)
] + [
    {"name": 'critical_random_beta_32k', "seed": 3067, "lengths": (32768,), "heads": 8, "key_heads": 2,
         "constant_log_gate": -1e-6, "state_scale": 1.},
    {"name": 'critical_alternating_beta', "seed": 3079, "lengths": (4097,), "heads": 8, "key_heads": 2,
         "nearly_correlated": True, "alternating_beta": True, "constant_log_gate": -1e-7, "state_scale": 1.},
    {"name": 'critical_packed', "seed": 3083, "lengths": (0, 127, 128, 129, 0, 513), "heads": 24, "key_heads": 8,
         "beta": 1., "constant_log_gate": -1e-6, "state_scale": 1.},
    {"name": 'critical_zero_initial', "seed": 3089, "lengths": (4097,), "heads": 8, "key_heads": 2,
         "nearly_correlated": True, "beta": 1., "constant_log_gate": -1e-6, "initial": False},
    {"name": 'critical_large_static_state', "seed": 3109, "lengths": (1025,), "heads": 8, "key_heads": 2,
         "beta": 0., "gate_scale": 0., "state_scale": 128.},
]


# Matched controls added after the initial critical sweep to isolate failures.
CRITICAL_CASES += [
    {"name": 'control_nearly_correlated_no_decay', "seed": 3011, "lengths": (1025,), "heads": 8, "key_heads": 2,
         "nearly_correlated": True, "beta": 1., "constant_log_gate": 0., "state_scale": 1.},
    {"name": 'control_zero_initial_no_decay', "seed": 3089, "lengths": (4097,), "heads": 8, "key_heads": 2,
         "nearly_correlated": True, "beta": 1., "constant_log_gate": 0., "initial": False},
    {"name": 'control_coordinate_zero_initial', "seed": 3007, "lengths": (32768,), "heads": 8, "key_heads": 2,
         "correlated": True, "beta": 1., "constant_log_gate": -1e-7, "initial": False},
    {"name": 'control_coordinate_beta_zero', "seed": 3007, "lengths": (32768,), "heads": 8, "key_heads": 2,
         "correlated": True, "beta": 0., "constant_log_gate": -1e-7, "state_scale": 1.},
    {"name": 'control_coordinate_no_decay', "seed": 3007, "lengths": (32768,), "heads": 8, "key_heads": 2,
         "correlated": True, "beta": 1., "constant_log_gate": 0., "state_scale": 1.},
]


# Interior rates distinguish endpoint behavior from the supported interval.
CRITICAL_CASES += [
    {"name": f'interior_beta_{rate}_{kind}', "seed": 3163, "lengths": (4097,), "heads": 8, "key_heads": 2,
         kind: True, "beta": rate, "constant_log_gate": -1e-7, "state_scale": 1.}
    for rate in (.25, .5, .6, .9) for kind in ('correlated', 'nearly_correlated')
] + [
    {"name": 'interior_random_beta_correlated', "seed": 3163, "lengths": (4097,), "heads": 8, "key_heads": 2,
         "correlated": True, "constant_log_gate": -1e-7, "state_scale": 1.}
]


# Matched unit-scale sweep: correlation is varied without changing Q/V/h0.
# These are synthetic sensitivity measurements, not measured model inputs.
CORRELATION_CASES = [
    {"name": f"correlation_rho{rho}_beta{rate}_g{gate}", "seed": 3827,
         "lengths": (4096,), "heads": 8, "key_heads": 2, "key_correlation": rho,
         "beta": rate, "constant_log_gate": gate, "state_scale": .1}
    for rho in (0., .25, .5, .75, .95, 1.)
    for rate in (.5, 1.) for gate in (0., -.001, -.1, -1.)
] + [
    {"name": f"correlation_long_rho{rho}", "seed": 3827, "lengths": (32768,),
         "heads": 8, "key_heads": 2, "key_correlation": rho, "beta": 1., "constant_log_gate": 0., "state_scale": .1}
    for rho in (.95, 1.)
] + [
    {"name": f"correlation_large_rho{rho}_g{gate}", "seed": 3827, "lengths": (4096,),
         "heads": 8, "key_heads": 2, "key_correlation": rho, "beta": 1., "constant_log_gate": gate,
         "amplitude": 32., "state_scale": .1}
    for rho in (0., .25, .5, .75, .95, 1.) for gate in (0., -.001)
] + [
    {"name": f"correlation_initial_rho{rho}_g{gate}", "seed": 3827, "lengths": (4096,),
         "heads": 8, "key_heads": 2, "key_correlation": rho, "beta": 1., "constant_log_gate": gate,
         "state_scale": 1.}
    for rho in (.75, .95, 1.) for gate in (0., -.001)
]


# Preserve grouped, single-key and multi-head regression cases, including
# partial tiles, from the former value-column scheduling implementation.
# Existing critical cases cover H8/Hg2 at 1K–32K with extreme gates and beta.
SCHEDULING_CASES = [
    {"name": f'parallel_columns_hg{keys}_t{length}_initial{initial}',
         "seed": 4111, "lengths": (length,), "heads": 8, "key_heads": keys,
         "nearly_correlated": True, "beta": 1., "constant_log_gate": -1e-6,
         "amplitude": 32., "initial": initial}
    for keys in (1, 4, 8) for length in (1024, 2048, 2139) for initial in (False, True)
]

AUDIT_CASES = STANDARD_CASES + CRITICAL_CASES + CORRELATION_CASES + SCHEDULING_CASES
# fmt: on


READ_ONLY = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "mask_lower",
    "mask_full",
    "minus_identity",
    "cu_seqlens",
)


def apply_fla_inputs(case, q, k, v, g, beta):
    """Apply the FLA input distribution, preserving RNG call order."""
    if case["fla"] == "positive":
        q.copy_(torch.nn.functional.normalize(torch.rand_like(q), dim=-1))
        k.copy_(torch.nn.functional.normalize(torch.rand_like(k), dim=-1))
        v.copy_(torch.rand_like(v))
    beta_source = torch.randn_like(g) if case.get("beta_logits") else torch.rand_like(g)
    beta.copy_(
        (beta_source.sigmoid() * (2 if case.get("negative_eigenvalues") else 1)).half()
    )
    gate_source = torch.randn_like(g) if case.get("gate_logits") else torch.rand_like(g)
    g.copy_(
        torch.nn.functional.logsigmoid(gate_source) / case.get("gate_normalizer", 1.0)
    )
    if "gate_mask" in case:
        g.mul_(torch.rand_like(g) > case["gate_mask"])
    if case.get("gate_activation"):
        a_log = torch.randn(case["heads"])
        bias = torch.randn(case["heads"])
        g.copy_(-a_log.exp() * torch.nn.functional.softplus(torch.randn_like(g) + bias))


def make_audit_inputs(case):
    q, k, v, g, beta = _make_inputs(
        sum(case["lengths"]), case["heads"], case["key_heads"], seed=case["seed"]
    )
    inputs = [q, k, v, g, beta]
    if "fla" in case:
        apply_fla_inputs(case, q, k, v, g, beta)
    g.mul_(case.get("gate_scale", 1.0))
    if "constant_log_gate" in case:
        g.fill_(case["constant_log_gate"])
    amplitude = case.get("amplitude", 1.0)
    v.mul_(amplitude)
    if case.get("correlated"):
        # Exact unit vectors: stable delta-rule recurrence even with beta=1.
        k.zero_()
        k[..., 0] = 1
    if case.get("correlated_random") or case.get("nearly_correlated"):
        keys = k[:, :1].float().expand_as(k).clone()
        if case.get("nearly_correlated"):
            keys += 0.01 * torch.randn_like(keys)
            keys /= keys.norm(dim=-1, keepdim=True)
        after = case.get("correlated_after", 0)
        k[:, after:].copy_(keys[:, after:].half())
    if "key_correlation" in case:
        # A shared unit direction plus independent orthogonal noise gives
        # expected inter-token cosine rho. A private RNG keeps Q/V/g/beta and
        # the initial state identical across the matched correlation sweep.
        rho = case["key_correlation"]
        assert 0.0 <= rho <= 1.0
        generator = torch.Generator().manual_seed(case["seed"] + 17011)
        common = torch.nn.functional.normalize(
            torch.randn(1, 1, k.shape[2], D, generator=generator), dim=-1
        )
        noise = k.float() - (k.float() * common).sum(-1, keepdim=True) * common
        noise = torch.nn.functional.normalize(noise, dim=-1)
        k.copy_((rho**0.5 * common + (1 - rho) ** 0.5 * noise).half())
    k.mul_(case.get("key_scale", 1.0))
    if "beta" in case:
        beta.fill_(case["beta"])
    if case.get("alternating_beta"):
        beta[:, ::2] = 0.0
        beta[:, 1::2] = 1.0
    if case.get("zero_query"):
        q.zero_()
    if case.get("mixed"):
        for chunk, start in enumerate(range(0, q.shape[1], 128)):
            g[:, start : start + 128].mul_([4.0, 0.01, 0.0, 1.0][chunk % 4])
    initial = None
    if case.get("initial", True):
        initial = torch.randn(
            len(case["lengths"]), case["heads"], D, D, dtype=torch.float16
        ) * (case.get("state_scale", 0.1) * amplitude)
    cu = torch.tensor(
        [0, *torch.tensor(case["lengths"]).cumsum(0).tolist()], dtype=torch.int32
    )
    return inputs, initial, cu


def make_capture_inputs(record, start=0):
    """Replay captured tensors in the kernel ABI, with no normalization/clipping.

    For continuation, derive the supplied FP16 state from the independent FP64
    prefix recurrence. The suffix oracle consumes that same rounded state.
    """
    inputs = [
        record[name].detach().cpu().to(dtype).unsqueeze(0).contiguous()
        for name, dtype in (
            ("q", torch.float16),
            ("k", torch.float16),
            ("v", torch.float16),
            ("g", torch.float32),
            ("beta", torch.float16),
        )
    ]
    assert 0 <= start < inputs[0].shape[1]
    initial = None
    if start:
        _, initial = recurrent_reference(*(x[:, :start] for x in inputs))
        initial = initial.half()
        inputs = [x[:, start:].contiguous() for x in inputs]
    cu = torch.tensor([0, inputs[0].shape[1]], dtype=torch.int32)
    return inputs, initial, cu


def run_audit_case(case, device, *, prepared_inputs=None, benchmark=None):
    inputs, initial, cu = (
        make_audit_inputs(case) if prepared_inputs is None else prepared_inputs
    )
    digest = hashlib.sha256()
    for tensor in [*inputs, initial, cu]:
        if tensor is not None:
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(memoryview(tensor.contiguous().numpy()).cast("B"))
    # Reference stays on the CPU and is never passed to the device/launch helper.
    expected, expected_state = cached_recurrent_reference(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], initial, cu
    )
    launch = _prepare_gdn_mega(
        *(x.to(device) for x in inputs),
        cu.to(device),
        initial.to(device) if initial is not None else None,
    )
    snapshots = {name: launch[name].clone() for name in READ_ONLY}
    if initial is not None:
        snapshots["initial_state"] = launch["initial_state"].clone()
    guards = []
    for name in ["out", "final_state"]:
        old = launch[name]
        backing = torch.full(
            (old.numel() + 128,), -123.25, dtype=old.dtype, device=device
        )
        launch[name] = backing[64:-64].view_as(old)
        guards.append(backing)
    if initial is None:
        launch["initial_state"] = launch[
            "final_state"
        ]  # Deliberately poisoned, must be ignored.
    metrics, first = [], None
    for fill in [0.0, float("nan"), 17.0, -9.0, float("nan")]:
        for name, tensor in launch.items():
            if (
                isinstance(tensor, torch.Tensor)
                and name not in snapshots
                and name != "initial_state"
            ):
                tensor.fill_(fill)
        pto_gdn_mega(**launch)
        torch.npu.synchronize()
        actual = launch["out"].cpu().clone()
        state = launch["final_state"].cpu().view_as(expected_state).clone()
        integrity = all(
            torch.equal(launch[name], before) for name, before in snapshots.items()
        )
        integrity &= all(
            bool((x[:64] == -123.25).all() and (x[-64:] == -123.25).all())
            for x in guards
        )
        if first is None:
            first = (actual.clone(), state.clone())
        # Byte comparison remains meaningful even if numerical checks expose NaNs.
        independent = all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip((actual, state), first)
        )
        # Reuse statistics only after checking every output/state bit. All five
        # device launches and all integrity checks still execute independently.
        numerical = (
            {
                key: metrics[0][key]
                for key in ("raw_output", "scaled_output", "final_state")
            }
            if metrics and independent
            else {
                "raw_output": error_metrics(actual, expected),
                "scaled_output": error_metrics(
                    actual.double() * D**-0.5, expected * D**-0.5
                ),
                "final_state": error_metrics(state, expected_state),
            }
        )
        metrics.append(
            {
                **numerical,
                "inputs_and_guards_intact": integrity,
                "scratch_independent": independent,
            }
        )
    timing = {}
    if benchmark is not None:
        timing["timing"] = benchmark(launch)
        actual = launch["out"].cpu().clone()
        state = launch["final_state"].cpu().view_as(expected_state).clone()
        timing["after_replay"] = {
            "raw_output": error_metrics(actual, expected),
            "final_state": error_metrics(state, expected_state),
            "inputs_and_guards_intact": all(
                torch.equal(launch[name], before) for name, before in snapshots.items()
            )
            and all(
                bool((x[:64] == -123.25).all() and (x[-64:] == -123.25).all())
                for x in guards
            ),
        }
    correlation = {}
    if "key_correlation" in case:
        cosine = torch.nn.functional.cosine_similarity(
            inputs[1][:, 1:].double(), inputs[1][:, :-1].double(), dim=-1
        )
        correlation = {
            "adjacent_key_cosine_mean": cosine.mean().item(),
            "adjacent_key_cosine_min": cosine.min().item(),
            "adjacent_key_cosine_max": cosine.max().item(),
        }
    per_head = {}
    if case.get("per_head"):
        fields = (
            "passed",
            "finite",
            "elements",
            "max_abs",
            "pointwise_violations",
            "max_tolerance_ratio",
            "abs_error_at_worst_tolerance",
            "relative_rmse",
        )
        per_head["per_head"] = [
            {
                "head": head,
                **{
                    name: {
                        key: value
                        for key, value in error_metrics(a, e).items()
                        if key in fields
                    }
                    for name, a, e in (
                        ("raw_output", actual[:, :, head], expected[:, :, head]),
                        ("final_state", state[:, head], expected_state[:, head]),
                    )
                },
            }
            for head in range(case["heads"])
        ]
    return {
        "case": case,
        "policy": POLICY,
        "repeats": metrics,
        "input_sha256": digest.hexdigest(),
        **per_head,
        **timing,
        "input_ranges": {
            **correlation,
            "beta": [inputs[4].min().item(), inputs[4].max().item()],
            "log_gate": [inputs[3].min().item(), inputs[3].max().item()],
            "key_norm_squared": [
                inputs[1].double().square().sum(-1).min().item(),
                inputs[1].double().square().sum(-1).max().item(),
            ],
            "value_abs_max": inputs[2].abs().max().item(),
            "initial_state_abs_max": (
                initial.abs().max().item() if initial is not None else 0.0
            ),
        },
        "fp16_output_rounding_floor": error_metrics(expected.half(), expected),
        "fp16_state_rounding_floor": error_metrics(
            expected_state.half(), expected_state
        ),
        "passed": (
            not timing
            or (
                timing["after_replay"]["raw_output"]["passed"]
                and timing["after_replay"]["final_state"]["passed"]
                and timing["after_replay"]["inputs_and_guards_intact"]
            )
        )
        and all(
            m["raw_output"]["passed"]
            and m["final_state"]["passed"]
            and m["inputs_and_guards_intact"]
            and m["scratch_independent"]
            for m in metrics
        ),
    }


@pytest.mark.parametrize("case", AUDIT_CASES, ids=[c["name"] for c in AUDIT_CASES])
def test_independent_recurrence_and_memory(case, npu_device):
    result = run_audit_case(case, npu_device)
    assert result["passed"], result


def test_correlation_sweep_preserves_non_key_inputs():
    base = {
        "seed": 3827,
        "lengths": (1024,),
        "heads": 8,
        "key_heads": 2,
        "beta": 1.0,
        "constant_log_gate": -0.001,
        "state_scale": 0.1,
    }
    ordinary, initial, cu = make_audit_inputs(base)
    for rho in (0.0, 0.75, 1.0):
        inputs, state, lengths = make_audit_inputs(dict(base, key_correlation=rho))
        assert all(torch.equal(inputs[i], ordinary[i]) for i in (0, 2, 3, 4))
        assert torch.equal(state, initial) and torch.equal(lengths, cu)
        norm = inputs[1].double().norm(dim=-1)
        torch.testing.assert_close(norm, torch.ones_like(norm), rtol=0, atol=5e-4)
        cosine = torch.nn.functional.cosine_similarity(
            inputs[1][:, 1:].double(), inputs[1][:, :-1].double(), dim=-1
        )
        assert abs(cosine.mean().item() - rho) < 0.02


def test_capture_replay_preserves_inputs_and_rounds_prefix_state():
    generator = torch.Generator().manual_seed(27101)
    record = {
        name: torch.randn(3, 2, D, generator=generator).half()
        for name in ("q", "k", "v")
    }
    record["g"] = -torch.rand(3, 2, generator=generator) * 0.01
    record["beta"] = torch.rand(3, 2, generator=generator)
    snapshots = {name: value.clone() for name, value in record.items()}
    inputs, initial, cu = make_capture_inputs(record)
    assert initial is None and cu.tolist() == [0, 3]
    for value, name in zip(inputs, ("q", "k", "v", "g", "beta")):
        assert torch.equal(value[0], record[name].to(value.dtype))
    _, prefix_state = recurrent_reference(*(x[:, :2] for x in inputs))
    suffix, initial, cu = make_capture_inputs(record, start=2)
    assert torch.equal(initial, prefix_state.half()) and cu.tolist() == [0, 1]
    assert all(torch.equal(a, b[:, 2:]) for a, b in zip(suffix, inputs))
    assert all(torch.equal(record[name], value) for name, value in snapshots.items())


def test_reference_analytic_two_tokens():
    q = k = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]])
    v = torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]]]])
    g = torch.zeros(1, 2, 1)
    beta = torch.full((1, 2, 1), 0.5)
    out, state = recurrent_reference(q, k, v, g, beta)
    torch.testing.assert_close(
        out,
        torch.tensor([[[[0.5, 0.0]], [[1.25, 0.0]]]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        state,
        torch.tensor([[[[1.25, 0.0], [0.0, 0.0]]]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_reference_correlated_beta_one_is_stable():
    case = next(c for c in AUDIT_CASES if c["name"] == "correlated_beta1")
    inputs, initial, cu = make_audit_inputs(case)
    out, state = recurrent_reference(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], initial, cu
    )
    q, _ = _expand_kv_heads(inputs[0], inputs[1], case["heads"])
    # With k=e_0, beta=1 and g=0, each step replaces precisely the first
    # state row with v_t; the other rows retain their initial values.
    analytic = torch.einsum("bthd,hde->bthe", q.double(), initial[0].double())
    analytic += q[..., :1].double() * (
        inputs[2].double() - initial[0, :, 0, :].double()
    )
    final = initial.double().clone()
    final[0, :, 0, :] = inputs[2][0, -1].double()
    torch.testing.assert_close(out, analytic, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(state, final, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed,gate_scale", [(0, 1.0), (42, 0.01), (93, 0.0)])
def test_chunked_reference_against_recurrence(seed, gate_scale):
    inputs = list(_make_inputs(257, 8, 2, seed))
    inputs[3].mul_(gate_scale)
    initial = torch.randn(1, 8, D, D, dtype=torch.float16) * 0.1
    raw, state64 = recurrent_reference(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], initial
    )
    q, k = _expand_kv_heads(inputs[0], inputs[1], 8)
    out, state = _reference(q, k, inputs[2], inputs[3], inputs[4], initial)
    torch.testing.assert_close(out.double(), raw * D**-0.5, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(state.double(), state64, atol=1e-6, rtol=1e-5)


def test_checker_rejects_spike_hidden_by_rmse():
    expected = torch.ones(1_000_000)
    actual = expected.clone()
    actual[123] += 0.1
    assert error_metrics(actual, expected)["relative_rmse"] < 0.005
    with pytest.raises(AssertionError):
        assert_numerically_close("spike", actual, expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_checker_rejects_nonfinite(bad):
    with pytest.raises(AssertionError):
        assert_numerically_close("nonfinite", torch.tensor([bad]), torch.zeros(1))


def test_checker_rejects_broadcasting():
    with pytest.raises(AssertionError):
        assert_numerically_close("shape", torch.ones(8), torch.ones(1))


def test_contiguous_views_respect_storage_offset(npu_device):
    """Regression for storage().data() ignoring a contiguous view's offset.

    Compare identical launches bit for bit; this tests pointer handling, without
    relying on or relaxing the separate numerical acceptance criterion.
    """
    inputs, initial, cu = make_audit_inputs(AUDIT_CASES[0])
    launch = _prepare_gdn_mega(
        *(x.to(npu_device) for x in inputs), cu.to(npu_device), initial.to(npu_device)
    )
    pto_gdn_mega(**launch)
    torch.npu.synchronize()
    expected = {name: launch[name].clone() for name in ("out", "final_state")}
    guards = []
    for name in (*READ_ONLY[:-1], "initial_state", "out", "final_state"):
        old = launch[name]
        backing = torch.full(
            (old.numel() + 128,), -123.25, dtype=old.dtype, device=npu_device
        )
        launch[name] = backing[64:-64].view_as(old)
        launch[name].copy_(old if name not in expected else torch.zeros_like(old))
        guards.append(backing)
    for _ in range(5):
        pto_gdn_mega(**launch)
        torch.npu.synchronize()
        for name in expected:
            assert torch.equal(launch[name], expected[name]), name
        assert all(
            bool((x[:64] == -123.25).all() and (x[-64:] == -123.25).all())
            for x in guards
        )


def test_changed_inputs_on_reused_workspace(npu_device):
    """A second request must depend on its inputs, not previous scratch data."""
    case = AUDIT_CASES[0]
    inputs, initial, cu = make_audit_inputs(case)
    launch = _prepare_gdn_mega(
        *(x.to(npu_device) for x in inputs), cu.to(npu_device), initial.to(npu_device)
    )
    pto_gdn_mega(**launch)
    torch.npu.synchronize()
    inputs2, initial2, _ = make_audit_inputs({**case, "seed": 6599})
    for name, tensor in zip(("q", "k", "v", "g", "beta"), inputs2):
        launch[name].copy_(tensor)
    launch["initial_state"].copy_(initial2)
    pto_gdn_mega(**launch)
    torch.npu.synchronize()
    reused = {name: launch[name].clone() for name in ("out", "final_state")}
    fresh = _prepare_gdn_mega(
        *(x.to(npu_device) for x in inputs2), cu.to(npu_device), initial2.to(npu_device)
    )
    pto_gdn_mega(**fresh)
    torch.npu.synchronize()
    for name in reused:
        assert torch.equal(reused[name], fresh[name]), name


@pytest.mark.parametrize(
    "gate,beta_value,initial_supplied",
    [
        (1.0, 0.9, True),
        (0.0, 1.0, True),
        (0.01, 1.999, True),
        (0.0, 1.999, False),
        (-0.01, 0.5, True),
    ],
)
def test_reused_scratch_oracle_against_literal_fp64(gate, beta_value, initial_supplied):
    """The faster golden must preserve the independent literal recurrence."""
    inputs = list(_make_inputs(129, 8, 2, seed=185))
    inputs[3].mul_(gate)
    inputs[4].fill_(beta_value)
    initial = torch.randn(3, 8, D, D).half() if initial_supplied else None
    cu = torch.tensor([0, 63, 63, 129], dtype=torch.int32)
    snapshots = [x.clone() for x in inputs]
    initial_snapshot = initial.clone() if initial is not None else None
    q, k, v, g, beta = [x.double() for x in inputs]
    q, k = _expand_kv_heads(q, k, 8)
    expected = torch.empty_like(v)
    states = []
    for i, (start, end) in enumerate(zip(cu.tolist(), cu.tolist()[1:])):
        state = (
            initial[i].double().clone()
            if initial is not None
            else torch.zeros(8, D, D, dtype=torch.float64)
        )
        for t in range(start, end):
            state = state * g[0, t].exp()[:, None, None]
            prediction = (k[0, t, :, :, None] * state).sum(1)
            residual = beta[0, t, :, None] * (v[0, t] - prediction)
            state = state + k[0, t, :, :, None] * residual[:, None, :]
            expected[0, t] = (q[0, t, :, :, None] * state).sum(1)
        states.append(state)
    actual, final = recurrent_reference(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], initial, cu
    )
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(final, torch.stack(states), atol=1e-14, rtol=1e-14)
    assert all(torch.equal(a, b) for a, b in zip(inputs, snapshots))
    if initial is not None:
        assert torch.equal(initial, initial_snapshot)


@pytest.mark.parametrize("case", ["rounded", "spike", "nan", "inf", "empty"])
def test_omitting_percentiles_preserves_every_acceptance_metric(case):
    torch.manual_seed(123)
    expected = torch.randn(3, 17) * 32
    actual = expected.half()
    if case == "spike":
        expected[0, 0], actual[0, 0] = 0, 0.01
    elif case in ("nan", "inf"):
        actual[0, 0] = float(case)
    elif case == "empty":
        expected, actual = expected[:0], actual[:0]
    full = error_metrics(actual, expected, include_percentiles=True)
    compact = error_metrics(actual, expected, include_percentiles=False)
    assert compact == {
        k: v for k, v in full.items() if k not in ("p50_abs", "p99_abs", "p999_abs")
    }
    if case in ("spike", "nan", "inf"):
        assert not compact["passed"]


def test_repeat_summary_keeps_worst_error_and_nonfinite_failure(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "benchmarking")
    )
    spec = importlib.util.spec_from_file_location(
        "gdn_error_worker",
        Path(__file__).resolve().parents[1] / "benchmarking/gdn_mega.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = torch.zeros(2, 4)
    repeats = [
        {
            "raw_output": error_metrics(
                torch.full_like(expected, value), expected, include_percentiles=False
            )
        }
        for value in (0.0, 0.01, 0.001)
    ]
    result = module.summarize(repeats, "raw_output")
    assert result["max_abs"] == repeats[1]["raw_output"]["max_abs"]
    assert result["pointwise_violations_max"] == 8
    assert not result["passed"]
    repeats.append(
        {"raw_output": error_metrics(torch.full_like(expected, float("nan")), expected)}
    )
    result = module.summarize(repeats, "raw_output")
    assert not result["finite"] and result["max_abs"] is None
    assert result["actual_nonfinite_max"] == 8


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (
            [],
            [
                (8, 2),
                (12, 3),
                (16, 4),
                (24, 6),
                (24, 8),
                (32, 8),
                (48, 12),
                (48, 16),
                (64, 16),
            ],
        ),
        (["--head-configs", "24/6", "24/8", "48/16"], [(24, 6), (24, 8), (48, 16)]),
        (["--head-configs", "24/8", "48/16"], [(24, 8), (48, 16)]),
    ],
)
def test_explicit_head_pairs_keep_distinct_gqa_rows(monkeypatch, arguments, expected):
    import argparse

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "benchmarking")
    )
    from gdn_mega import parse_head_configs

    parser = argparse.ArgumentParser()
    parser.add_argument("--head-configs", nargs="+")
    args = parser.parse_args(arguments)
    assert parse_head_configs(args.head_configs) == expected


@pytest.mark.parametrize(
    "pairs", [["24/0"], ["24/7"], ["24/8", "24/8"], ["24"], ["24/8/1"]]
)
def test_invalid_head_pairs_are_rejected(monkeypatch, pairs):
    import argparse

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "benchmarking")
    )
    from gdn_mega import parse_head_configs

    parser = argparse.ArgumentParser()
    parser.add_argument("--head-configs", nargs="+")
    with pytest.raises(ValueError):
        parse_head_configs(parser.parse_args(["--head-configs", *pairs]).head_configs)


def test_shared_golden_requires_matching_inputs_source_and_contents(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "benchmarking")
    )
    path = tmp_path / "golden.pt"
    identity = {"inputs": "input-hash", "oracle_sha256": "oracle-hash"}
    expected = (
        torch.arange(17, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
    )
    calls = []

    def compute():
        calls.append(1)
        return expected

    first = shared_reference(path, identity, compute)
    second = shared_reference(path, identity, compute)
    assert len(calls) == 1
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    for field in identity:
        with pytest.raises(RuntimeError, match="mismatch"):
            shared_reference(path, {**identity, field: "changed"}, compute)
    saved = torch.load(path, weights_only=True)
    saved["tensors"][0][-1] += 1
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="mismatch"):
        shared_reference(path, identity, compute)


def test_regression_golden_cache_invalidates_and_rejects_corruption(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GDN_REFERENCE_CACHE", str(tmp_path))
    calls = []

    def compute(value, initial):
        calls.append(1)
        return (
            value.double() + (0 if initial is None else initial),
            value.double().sum(),
        )

    monkeypatch.setitem(globals(), "recurrent_reference", compute)
    value = torch.arange(8, dtype=torch.float16)
    first = cached_recurrent_reference(value, None)
    second = cached_recurrent_reference(value.clone(), None)
    assert len(calls) == 1 and all(torch.equal(a, b) for a, b in zip(first, second))
    cached_recurrent_reference(value + 1, None)
    cached_recurrent_reference(value, torch.tensor(0.0))
    assert len(calls) == 3
    for path in tmp_path.glob("*.pt"):
        saved = torch.load(path, weights_only=True)
        saved["tensors"][0][0] += 1
        torch.save(saved, path)
    with pytest.raises(RuntimeError, match="mismatch"):
        cached_recurrent_reference(value, None)
    monkeypatch.delenv("GDN_REFERENCE_CACHE")
    cached_recurrent_reference(value, None)
    assert len(calls) == 4


@pytest.mark.parametrize(
    "baseline,current,additional,expected",
    [
        ("passed", "passed", {}, (False, False)),
        ("failed", "passed", {}, (True, False)),
        ("passed", "failed", {}, (False, True)),
        ("passed", "passed", {"baseline_accuracy_status": "failed"}, (True, False)),
        ("failed", "passed", {"baseline_accuracy_status": "passed"}, (True, False)),
    ],
)
def test_numerical_asterisk_is_per_cell_and_implementation(
    monkeypatch, baseline, current, additional, expected
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "benchmarking")
    )
    from gdn_mega import numerical_failure

    row = {
        "baseline": {"status": baseline},
        "current": {"status": current},
        **additional,
    }
    assert tuple(numerical_failure(row, v) for v in ("baseline", "current")) == expected


def test_worst_tolerance_witness_is_distinct_from_maximum_absolute_error():
    expected = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    actual = torch.tensor([2 * ATOL, 0.5004, 1.0009], dtype=torch.float64)
    m = error_metrics(actual, expected)
    assert not m["passed"] and m["pointwise_violations"] == 1
    assert m["max_error_flat_index"] == 2
    assert m["worst_tolerance_flat_index"] == 0
    assert m["actual_at_worst_tolerance"] == 2 * ATOL
    assert m["reference_at_worst_tolerance"] == 0.0
    assert m["abs_error_at_worst_tolerance"] == 2 * ATOL
    assert m["allowed_at_worst_tolerance"] == ATOL
    assert m["max_tolerance_ratio"] == 2.0


@pytest.mark.parametrize("reference_value", [0.0, 1.0, 100.0])
def test_approved_elementwise_policy_rejects_a_single_regressing_element(
    reference_value,
):
    # Thousands of exact elements make the RMSE check pass even for the spike.
    # The elementwise bound must still distinguish the two sides of the limit.
    assert ATOL == 1e-4 and RTOL == 1e-3
    expected = torch.ones(4096, dtype=torch.float64)
    expected[0] = reference_value
    allowed = ATOL + RTOL * abs(reference_value)
    for factor, passed in ((0.99, True), (1.01, False)):
        actual = expected.clone()
        actual[0] += factor * allowed
        result = error_metrics(actual, expected)
        assert result["relative_rmse"] < MAX_RMSE_RATIO
        assert result["passed"] is passed
        assert result["pointwise_violations"] == (0 if passed else 1)


@pytest.fixture
def benchmark():
    spec = importlib.util.spec_from_file_location(
        "gdn_benchmark_worker",
        Path(__file__).resolve().parents[1] / "benchmarking/gdn_mega.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    launch = {
        "out": torch.zeros(1, 2, 1, 128, dtype=torch.float16),
        "final_state": torch.zeros(1, 1, 128, 128, dtype=torch.float16),
    }
    return (
        launch,
        torch.zeros_like(launch["out"]),
        torch.zeros_like(launch["final_state"]),
    )


def test_identical_outputs_reuse_only_the_validated_statistics(monkeypatch, benchmark):
    launch, expected, state = inputs()
    previous = benchmark.check_outputs(launch, expected, state)

    def unexpected_recalculation(*args):
        raise AssertionError("Byte-identical tensors do not need repeated percentiles")

    monkeypatch.setattr(benchmark, "error_metrics", unexpected_recalculation)
    assert benchmark.check_outputs(launch, expected, state, previous) is previous


@pytest.mark.parametrize("field", ["out", "final_state"])
@pytest.mark.parametrize("bad", [0.01, float("nan")])
def test_changed_element_is_rechecked_even_after_a_cached_pass(field, bad, benchmark):
    launch, expected, state = inputs()
    previous = benchmark.check_outputs(launch, expected, state)
    launch[field].flatten()[-1] = bad
    with pytest.raises(benchmark.AccuracyFailure):
        benchmark.check_outputs(launch, expected, state, previous)


def test_a_different_reference_invalidates_cached_metrics(benchmark):
    launch, expected, state = inputs()
    previous = benchmark.check_outputs(launch, expected, state)
    expected = expected.clone()
    expected.flatten()[-1] = 0.01
    with pytest.raises(benchmark.AccuracyFailure):
        benchmark.check_outputs(launch, expected, state, previous)


@pytest.mark.parametrize("mutated", ["output_reference", "state_reference"])
def test_inplace_reference_change_invalidates_cached_pass(mutated, benchmark):
    launch, expected, state = inputs()
    previous = benchmark.check_outputs(launch, expected, state)
    (expected if mutated == "output_reference" else state).flatten()[-1] = 0.01
    with pytest.raises(benchmark.AccuracyFailure):
        benchmark.check_outputs(launch, expected, state, previous)


def test_benchmark_golden_is_in_raw_output_units(benchmark):
    launch, expected, state = inputs()
    launch["out"].fill_(0.5)
    expected.fill_(0.5)
    checked = benchmark.check_outputs(launch, expected, state)
    assert checked[2]["raw_output"]["max_abs"] == 0
    with pytest.raises(benchmark.AccuracyFailure):
        benchmark.check_outputs(launch, expected * 128**-0.5, state)


def test_timing_failed_outputs_does_not_turn_them_into_a_pass(benchmark):
    launch, expected, state = inputs()
    launch["out"].flatten()[-1] = 0.01
    checked = benchmark.check_outputs(launch, expected, state, allow_failed=True)
    assert not checked[2]["raw_output"]["passed"]
    assert checked[2]["raw_output"]["pointwise_violations"] == 1
    assert (
        benchmark.check_outputs(launch, expected, state, checked, allow_failed=True)
        is checked
    )
    # A permissive timing result cannot leak into a subsequent strict check.
    with pytest.raises(benchmark.AccuracyFailure):
        benchmark.check_outputs(launch, expected, state, checked)
