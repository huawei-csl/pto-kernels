# Copyright (c) 2026 Huawei Technologies Co., Ltd. See repository LICENSE.
"""Validate, allocate, and enqueue the native PTO backward once."""
import ctypes

import torch

from .binding import Input, build


@torch.no_grad()
def backward(
    q,
    k,
    v,
    g,
    beta,
    dout,
    initial_state=None,
    dfinal_state=None,
    cu_seqlens=None,
    *,
    scale=1.0,
):
    """Return FP32 (dq,dk,dv,dg,dbeta,dh0), packed B=1, C=128, D=128.

    The native launcher owns replay, chunk traversal, and workspace reuse.
    Inputs may be FP16/BF16/FP32. Enqueued on the current Torch NPU stream.
    """
    device = q.device
    if device.type not in ("npu", "privateuseone"):
        raise ValueError("inputs must be on NPU")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or k.shape != q.shape:
        raise ValueError("q/k must have shape [1,T,Hk,128]")
    T, HK = q.shape[1:3]
    if v.ndim != 4 or v.shape[:2] != (1, T) or v.shape[-1] != 128:
        raise ValueError("v must have shape [1,T,Hv,128]")
    H = v.shape[2]
    if (
        not HK
        or not H
        or H % HK
        or g.shape != (1, T, H)
        or beta.shape != g.shape
        or dout.shape != v.shape
    ):
        raise ValueError("invalid heads, gates or dout shapes")
    bounds = [0, T]
    if cu_seqlens is not None:
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.numel() < 2
        ):
            raise ValueError("cu_seqlens must be a 1D integer tensor")
        bounds = cu_seqlens.cpu().tolist()  # scheduling metadata, never model data
    if (
        bounds[0] != 0
        or bounds[-1] != T
        or any(a > b for a, b in zip(bounds, bounds[1:]))
    ):
        raise ValueError("cu_seqlens must span [0,T] in nondecreasing order")
    N = len(bounds) - 1
    for x in (q, k, v, g, beta, dout, initial_state, dfinal_state):
        if x is not None and (
            x.device != device
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError(
                "all data tensors must share the NPU and use FP16/BF16/FP32"
            )
    for x in (initial_state, dfinal_state):
        if x is not None and x.shape != (N, H, 128, 128):
            raise ValueError("state must have shape [N,Hv,128,128]")

    with torch.npu.device(device):
        native = build()
        cu = (ctypes.c_int64 * len(bounds))(*bounds)
        descriptors = (Input * 8)(
            *[
                Input.from_tensor(x)
                for x in (q, k, v, g, beta, dout, initial_state, dfinal_state)
            ]
        )
        outputs = [
            torch.empty(x.shape, device=device, dtype=torch.float32)
            for x in (q, k, v, g, beta)
        ]
        outputs.append(
            torch.empty((N, H, 128, 128), device=device, dtype=torch.float32)
        )
        workspace = torch.empty(
            native.workspace_elements(cu, N, H), device=device, dtype=torch.float32
        )
        pointers = (ctypes.c_void_p * 6)(*[x.data_ptr() for x in outputs])
        native.gdn_backward(
            descriptors,
            cu,
            N,
            H,
            HK,
            pointers,
            workspace.data_ptr(),
            scale,
            native.stage_launcher,
        )
        return tuple(outputs)
