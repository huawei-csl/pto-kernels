"""Static PTO wy_fast: compile + PyTorch reference check."""
from __future__ import annotations

import ctypes
import os

import torch

import pto_static_common  # noqa: F401
from pto_static_common import compile_pto_kernel

torch_npu = torch.npu  # noqa: F401

B, H, L, DK, DV, C = 2, 16, 16384, 128, 128, 128


def ref_wy_fast(k, v, beta, g, a, C_):
    B_, H_, L_, DK_ = k.shape
    DV_ = v.shape[-1]
    chunk_num = (L_ + C_ - 1) // C_
    w = torch.zeros((B_, H_, L_, DK_), device=k.device, dtype=torch.float16)
    u = torch.zeros((B_, H_, L_, DV_), device=k.device, dtype=torch.float16)
    g_e = torch.exp(g)
    beta = beta.float()

    for i in range(chunk_num):
        a_c = a[:, :, i * C_ : (i + 1) * C_, :].to(torch.float)
        k_c = k[:, :, i * C_ : (i + 1) * C_, :]
        v_c = v[:, :, i * C_ : (i + 1) * C_, :]
        beta_c = beta[:, :, i * C_ : (i + 1) * C_]
        g_c = g_e[:, :, i * C_ : (i + 1) * C_]
        g_c = g_c * beta_c
        a2_c = torch.einsum("bhlc,bhc->bhlc", a_c, beta_c).to(torch.float16)
        a1_c = torch.einsum("bhlc,bhc->bhlc", a_c, g_c).to(torch.float16)
        w[:, :, i * C_ : (i + 1) * C_, :] = torch.matmul(a1_c, k_c)
        u[:, :, i * C_ : (i + 1) * C_, :] = torch.matmul(a2_c, v_c)

    return w, u


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    lib_path = compile_pto_kernel("wy_fast_kernel.cpp", "wy_fast_static.so")
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_void_p]
    lib.call.restype = None

    k = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    v = torch.randn((B, H, L, DV), device="npu", dtype=torch.float16)
    beta = torch.rand((B, H, L), device="npu", dtype=torch.float16)
    g = torch.randn((B, H, L), device="npu", dtype=torch.float32)
    a = torch.randn((B, H, L, C), device="npu", dtype=torch.float16)
    workspace_a1 = torch.zeros((B, H, L, C), device="npu", dtype=torch.float16)
    workspace_a2 = torch.zeros((B, H, L, C), device="npu", dtype=torch.float16)
    w_out = torch.empty((B, H, L, DK), device="npu", dtype=torch.float16)
    u_out = torch.empty((B, H, L, DV), device="npu", dtype=torch.float16)

    stream = torch.npu.current_stream()._as_parameter_
    lib.call(
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(beta.data_ptr()),
        ctypes.c_void_p(g.data_ptr()),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(workspace_a1.data_ptr()),
        ctypes.c_void_p(workspace_a2.data_ptr()),
        ctypes.c_void_p(w_out.data_ptr()),
        ctypes.c_void_p(u_out.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    ref_w, ref_u = ref_wy_fast(k, v, beta, g, a, C)
    torch.testing.assert_close(w_out.cpu(), ref_w.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(u_out.cpu(), ref_u.cpu(), rtol=1e-5, atol=1e-5)
    print("wy_fast static kernel matches PyTorch reference.")


if __name__ == "__main__":
    main()
