"""Static PTO chunk cumsum: compile + PyTorch reference check."""
from __future__ import annotations

import ctypes
import os

import torch

import pto_static_common  # noqa: F401 — env validation
from pto_static_common import compile_pto_kernel

torch_npu = torch.npu  # noqa: F401

B, H, L, C = 2, 16, 16384, 128


def ref_chunk_cumsum(g, C_):
    B_, H_, L_ = g.shape
    chunk_num = (L_ + C_ - 1) // C_
    g = g.view(B_, H_, chunk_num, C_)
    g_sum = torch.cumsum(g, dim=-1)
    return g_sum.view(B_, H_, L_)


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    lib_path = compile_pto_kernel("chunk_cumsum_kernel.cpp", "chunk_cumsum_static.so")
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.call.restype = None

    g = torch.randn((B, H, L), device="npu", dtype=torch.float32)
    s_out = torch.empty_like(g)
    stream = torch.npu.current_stream()._as_parameter_
    lib.call(
        ctypes.c_void_p(g.data_ptr()),
        ctypes.c_void_p(s_out.data_ptr()),
        stream,
    )
    torch.npu.synchronize()

    ref = ref_chunk_cumsum(g, C)
    torch.testing.assert_close(s_out.cpu(), ref.cpu(), rtol=1e-5, atol=1e-5)
    print("chunk_cumsum static kernel matches PyTorch reference.")


if __name__ == "__main__":
    main()
