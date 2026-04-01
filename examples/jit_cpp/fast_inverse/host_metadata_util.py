# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

from __future__ import annotations

import ctypes
import os
import subprocess

import torch


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "host_chunk_metadata.cpp")
_LIB = os.path.join(_THIS_DIR, "host_chunk_metadata.so")
_HOST_LIB = None


def _torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def compile_host_metadata_cpp(timeout: int = 60) -> str:
    compiler = os.environ.get("CXX", "g++")
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        _SRC,
        "-o",
        _LIB,
    ]
    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as exc:
        raise RuntimeError(f"Host metadata compilation failed: {exc}") from exc
    return _LIB


def load_host_metadata_lib():
    global _HOST_LIB
    if _HOST_LIB is not None:
        return _HOST_LIB

    lib_path = compile_host_metadata_cpp()
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.count_varlen_chunks_host_cpp.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.count_varlen_chunks_host_cpp.restype = ctypes.c_uint32
    lib.build_varlen_chunk_metadata_host_cpp.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.build_varlen_chunk_metadata_host_cpp.restype = None
    _HOST_LIB = lib
    return lib


def build_varlen_chunk_metadata_cpp(
    cu_seqlens: torch.Tensor | list[int],
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lib = load_host_metadata_lib()
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.int32).contiguous()
    else:
        cu_seqlens_cpu = torch.tensor(cu_seqlens, dtype=torch.int32)

    if cu_seqlens_cpu.numel() < 2:
        raise ValueError("cu_seqlens must contain at least 2 entries.")

    num_sequences = cu_seqlens_cpu.numel() - 1
    num_chunks = int(
        lib.count_varlen_chunks_host_cpp(
            _torch_to_ctypes(cu_seqlens_cpu),
            num_sequences,
            chunk_size,
        )
    )
    chunk_indices = torch.empty(num_chunks, dtype=torch.int32)
    chunk_valid_sizes = torch.empty(num_chunks, dtype=torch.int32)
    lib.build_varlen_chunk_metadata_host_cpp(
        _torch_to_ctypes(cu_seqlens_cpu),
        num_sequences,
        chunk_size,
        _torch_to_ctypes(chunk_indices),
        _torch_to_ctypes(chunk_valid_sizes),
    )
    return chunk_indices, chunk_valid_sizes
