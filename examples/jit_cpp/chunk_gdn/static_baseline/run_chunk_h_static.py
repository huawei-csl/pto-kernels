"""
Compile the static chunk_h PTO kernel, load it, and compare to the PyTorch reference.

Shapes are fixed to match the generated TileLang specialization:
B=2, H=16, L=16384, DK=128, DV=128, C=128 (chunk_num=128).
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache

import torch
import torch.nn.functional as F

import pto_static_common  # noqa: F401 — env validation
from pto_static_common import compile_pto_kernel

torch_npu = torch.npu  # noqa: F401 — register NPU

# Matches tilelang test / generated kernel
B, H, L, DK, DV, C = 2, 16, 16384, 128, 128, 128
CHUNK_NUM = (L + C - 1) // C
BV_NUM = (DV + DV - 1) // DV
assert CHUNK_NUM == 128
assert B * H * BV_NUM == 32


@lru_cache(maxsize=1)
def get_lib():
    lib_path = compile_pto_kernel("chunk_h_kernel.cpp", "chunk_h_static.so")
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


def ref_chunk_h(k, w, u, g, C_):
    """Same logic as tilelang opt_gdn_chunk_h.ref_chunk_h."""
    B_, H_, L_, DK_ = k.shape
    DV_ = u.shape[-1]
    chunk_num = (L_ + C_ - 1) // C_
    s = torch.zeros((B_, H_, chunk_num, DK_, DV_), device=k.device, dtype=torch.float32)
    new_v = torch.zeros((B_, H_, L_, DV_), device=k.device, dtype=torch.float32)
    kf = k.float()
    uf = u.float()

    for i in range(chunk_num):
        las_s = s[:, :, i, :, :]
        k_c = kf[:, :, i * C_ : (i + 1) * C_, :]
        w_c = w[:, :, i * C_ : (i + 1) * C_, :]
        u_c = uf[:, :, i * C_ : (i + 1) * C_, :]
        g_c = g[:, :, i * C_ : (i + 1) * C_]
        ws = torch.matmul(w_c, las_s.to(torch.float16)).float()
        new_v_c = u_c - ws
        new_v[:, :, i * C_ : (i + 1) * C_, :] = new_v_c
        g_last = g[:, :, (i + 1) * C_ - 1].view(B_, H_, 1, 1)
        coeff_k = g_last - g_c.view(B_, H_, C_, 1)
        g_last_e = torch.exp(g_last)
        coeff_k = torch.exp(coeff_k)
        k_c = (k_c * coeff_k).transpose(-2, -1)
        las_s = las_s * g_last_e
        kv = torch.matmul(k_c.to(torch.float16), new_v_c.to(torch.float16)).float()
        s_c = las_s + kv
        if i < chunk_num - 1:
            s[:, :, i + 1, :, :] = s_c

    return s.to(torch.float16), new_v.to(torch.float16), s_c.to(torch.float16)


def ref_chunk_cumsum(g, C_):
    B_, H_, L_ = g.shape
    chunk_num = (L_ + C_ - 1) // C_
    g = g.view(B_, H_, chunk_num, C_)
    g_sum = torch.cumsum(g, dim=-1)
    return g_sum.view(B_, H_, L_)


def run_chunk_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    workspace_1: torch.Tensor,
    workspace_2: torch.Tensor,
    workspace_3: torch.Tensor,
    workspace_4: torch.Tensor,
    s: torch.Tensor,
    v_out: torch.Tensor,
    fs_out: torch.Tensor,
):
    lib = get_lib()
    stream = torch.npu.current_stream()._as_parameter_
    lib.call(
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(u.data_ptr()),
        ctypes.c_void_p(g.data_ptr()),
        ctypes.c_void_p(workspace_1.data_ptr()),
        ctypes.c_void_p(workspace_2.data_ptr()),
        ctypes.c_void_p(workspace_3.data_ptr()),
        ctypes.c_void_p(workspace_4.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
        ctypes.c_void_p(v_out.data_ptr()),
        ctypes.c_void_p(fs_out.data_ptr()),
        stream,
    )


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    k = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    w = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    u = torch.randn((B, H, L, DV), device="npu", dtype=torch.float16)
    g = torch.randn((B, H, L), device="npu", dtype=torch.float32)
    g = F.logsigmoid(g)
    k = F.normalize(k, dim=-1, p=2)
    w = F.normalize(w, dim=-1, p=2)
    g = ref_chunk_cumsum(g, C)

    workspace_1 = torch.zeros((B * H * BV_NUM, C, DV), device="npu", dtype=torch.float16)
    workspace_2 = torch.zeros((B * H * BV_NUM, C, DK), device="npu", dtype=torch.float16)
    workspace_3 = torch.zeros((B * H * BV_NUM, DK, DV), device="npu", dtype=torch.float16)
    workspace_4 = torch.zeros((B * H * BV_NUM, DK, DV), device="npu", dtype=torch.float16)
    s = torch.zeros((B, H, CHUNK_NUM, DK, DV), device="npu", dtype=torch.float16)
    v_out = torch.empty((B, H, L, DV), device="npu", dtype=torch.float16)
    fs_out = torch.empty((B, H, DK, DV), device="npu", dtype=torch.float16)

    run_chunk_h(k, w, u, g, workspace_1, workspace_2, workspace_3, workspace_4, s, v_out, fs_out)
    torch.npu.synchronize()

    ref_s, ref_new_v, ref_final_s = ref_chunk_h(k, w, u, g, C)

    torch.testing.assert_close(s.cpu(), ref_s.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_out.cpu(), ref_new_v.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fs_out.cpu(), ref_final_s.cpu(), rtol=1e-5, atol=1e-5)
    print("chunk_h static kernel matches PyTorch reference.")


if __name__ == "__main__":
    main()
