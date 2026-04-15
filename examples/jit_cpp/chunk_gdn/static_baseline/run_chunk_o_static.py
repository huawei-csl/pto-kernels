"""Static PTO chunk_o: compile + PyTorch reference check."""
from __future__ import annotations

import ctypes
import os

import torch
import torch.nn.functional as F

import pto_static_common  # noqa: F401
from pto_static_common import compile_pto_kernel

torch_npu = torch.npu  # noqa: F401

B, H, L, DK, DV, C = 16, 16, 16384, 128, 128, 128
CHUNK_NUM = (L + C - 1) // C


def ref_chunk_o(q, k, v, s, g, C_):
    B_, H_, L_, DK_ = k.shape
    DV_ = v.shape[-1]
    chunk_num = (L_ + C_ - 1) // C_
    o = torch.zeros((B_, H_, L_, DV_), device=k.device, dtype=torch.float32)
    M = torch.tril(torch.ones((C_, C_), device=k.device, dtype=torch.float32))

    for i in range(chunk_num):
        q_c = q[:, :, i * C_ : (i + 1) * C_, :]
        k_c = k[:, :, i * C_ : (i + 1) * C_, :].transpose(-2, -1)
        v_c = v[:, :, i * C_ : (i + 1) * C_, :]
        s_c = s[:, :, i, :, :]
        g_c = g[:, :, i * C_ : (i + 1) * C_]
        gamma = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
        g_c = torch.exp(g_c)
        gamma = torch.exp(gamma)
        term1 = torch.matmul(q_c, s_c).float()
        term1 = g_c.unsqueeze(-1) * term1
        qkt = torch.matmul(q_c, k_c).float()
        qkt = (qkt * gamma * M.view(1, 1, C_, C_)).to(torch.float16)
        term2 = torch.matmul(qkt, v_c).float()
        o_t = term1 + term2
        o[:, :, i * C_ : (i + 1) * C_, :] = o_t

    return o.to(torch.float16)


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    stream = torch.npu.current_stream()._as_parameter_

    lib_path = compile_pto_kernel("chunk_o_kernel.cpp", "chunk_o_static.so")
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_void_p]
    lib.call.restype = None

    q = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    k = torch.randn((B, H, L, DK), device="npu", dtype=torch.float16)
    v = torch.randn((B, H, L, DV), device="npu", dtype=torch.float16)
    s = torch.randn((B, H, CHUNK_NUM, DK, DV), device="npu", dtype=torch.float16)
    g = torch.randn((B, H, L), device="npu", dtype=torch.float32)
    msk = torch.tril(torch.ones((C, C), device="npu"), diagonal=0).to(torch.float32)

    q = F.normalize(q, dim=-1, p=2)
    k = F.normalize(k, dim=-1, p=2)

    nblk = B * H * CHUNK_NUM
    workspace_1 = torch.zeros((nblk, C, C), device="npu", dtype=torch.float16)
    workspace_2 = torch.zeros((nblk, C, DV), device="npu", dtype=torch.float16)
    workspace_3 = torch.zeros((nblk, C, C), device="npu", dtype=torch.float16)
    o = torch.empty((B, H, L, DV), device="npu", dtype=torch.float16)

    lib.call(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(v.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
        ctypes.c_void_p(g.data_ptr()),
        ctypes.c_void_p(msk.data_ptr()),
        ctypes.c_void_p(workspace_1.data_ptr()),
        ctypes.c_void_p(workspace_2.data_ptr()),
        ctypes.c_void_p(workspace_3.data_ptr()),
        ctypes.c_void_p(o.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    ref_o = ref_chunk_o(q, k, v, s, g, C)
    torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=1e-5, atol=1e-5)
    print("chunk_o static kernel matches PyTorch reference.")


if __name__ == "__main__":
    main()
