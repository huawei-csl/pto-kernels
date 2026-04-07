"""Static PTO scaled-dot KKT block: compile + PyTorch reference check."""
from __future__ import annotations

import ctypes
import os

import torch

import pto_static_common  # noqa: F401
from pto_static_common import compile_pto_kernel

torch_npu = torch.npu  # noqa: F401

B, H, L, DK, C = 2, 16, 16384, 128, 128


def ref_kkt(k, beta, g, C_):
    B_, H_, L_, DK_ = k.shape
    chunk_num = (L_ + C_ - 1) // C_
    a = torch.zeros((B_, H_, L_, C_), device=k.device, dtype=torch.float32)
    beta = beta.float()

    for i in range(chunk_num):
        k_c = k[:, :, i * C_ : (i + 1) * C_, :]
        beta_c = beta[:, :, i * C_ : (i + 1) * C_]
        g_c = g[:, :, i * C_ : (i + 1) * C_]
        kkt = torch.einsum("bhid,bhjd->bhij", k_c, k_c).float()
        gamma = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
        gamma = torch.exp(gamma)
        a_c = (kkt * beta_c.unsqueeze(-1) * gamma).tril(-1)
        a[:, :, i * C_ : (i + 1) * C_, :] = a_c

    return a.to(torch.float16)


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    lib_path = compile_pto_kernel("scaled_dot_kkt_kernel.cpp", "scaled_dot_kkt_static.so")
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_void_p]
    lib.call.restype = None

    k = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    beta = torch.rand((B, H, L), device="npu", dtype=torch.float16)
    g = torch.randn((B, H, L), device="npu", dtype=torch.float32)
    msk = torch.tril(torch.ones((C, C), device="npu"), diagonal=-1).to(torch.float32)
    workspace = torch.zeros((B, H, L, C), device="npu", dtype=torch.float16)
    a_out = torch.empty((B, H, L, C), device="npu", dtype=torch.float16)

    stream = torch.npu.current_stream()._as_parameter_
    lib.call(
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(beta.data_ptr()),
        ctypes.c_void_p(g.data_ptr()),
        ctypes.c_void_p(msk.data_ptr()),
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_void_p(a_out.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    ref_a = ref_kkt(k, beta, g, C)
    torch.testing.assert_close(a_out.cpu(), ref_a.cpu(), rtol=1e-3, atol=1e-3)
    print("scaled_dot_kkt static kernel matches PyTorch reference.")


if __name__ == "__main__":
    main()
