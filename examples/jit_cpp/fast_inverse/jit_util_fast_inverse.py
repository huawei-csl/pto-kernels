# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------

import ctypes
import os
import subprocess

import torch

# ---------------------------------------------------------------------------
# Environment / paths
# ---------------------------------------------------------------------------
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", os.environ["ASCEND_TOOLKIT_HOME"])

# Directory of this file  →  repo-root/examples/jit_cpp/fast_inverse
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# csrc/kernel lives three levels up from this file
_CSRC_KERNEL_DIR = os.path.abspath(os.path.join(_THIS_DIR, "../../../csrc/kernel"))

BLOCK_DIM = int(getattr(torch.npu.get_device_properties("npu:0"), "cube_core_num", 20))


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_cpp(kernel_cpp: str, verbose: bool = False, timeout: int = 180) -> str:
    """Compile *kernel_cpp* with bisheng and return the path to the .so."""
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "fast_inverse_jit.so")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        # Resolve kernel_utils.h (included by kernel_tri_inv_rec_unroll.cpp)
        f"-I{_CSRC_KERNEL_DIR}",
        # PTO-ISA headers
        f"-I{PTO_LIB_PATH}/include",
        # Target the Ascend 910B cube core
        "--cce-soc-version=Ascend910B4",
        "--cce-soc-core-type=CubeCore",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("Compiling fast_inverse kernel:")
        print(" ", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as exc:
        raise RuntimeError(f"Compilation failed: {exc}") from exc

    if verbose:
        print(f"Generated: {lib_path}")
    return lib_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path: str):
    """Load the compiled .so and return a Python callable for the kernel."""
    lib = ctypes.CDLL(os.path.abspath(lib_path))

    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # tensor_out  (fp32)
        ctypes.c_void_p,  # tensor_in   (fp16)
        ctypes.c_void_p,  # minus_identity_in (fp16)
        ctypes.c_uint32,  # matrix_size
        ctypes.c_uint32,  # num_matrices
        ctypes.c_uint32,  # num_bsnd_heads
        ctypes.c_void_p,  # cu_seqlens (optional int32 metadata)
        ctypes.c_void_p,  # chunk_indices (optional int32 metadata)
        ctypes.c_void_p,  # chunk_valid_sizes (optional int32 metadata)
    ]
    lib.call_kernel.restype = None

    def tri_inv_func(
        tensor_out: torch.Tensor,
        tensor_in: torch.Tensor,
        minus_identity: torch.Tensor,
        matrix_size: int,
        num_matrices: int,
        num_bsnd_heads: int = 0,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_valid_sizes: torch.Tensor | None = None,
        block_dim: int = BLOCK_DIM,
        stream_ptr=None,
    ):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa
        if cu_seqlens is not None:
            if cu_seqlens.dtype != torch.int32:
                raise TypeError("cu_seqlens must be int32.")
            if not cu_seqlens.is_contiguous():
                raise ValueError("cu_seqlens must be contiguous.")
        if chunk_indices is not None:
            if chunk_indices.dtype != torch.int32:
                raise TypeError("chunk_indices must be int32.")
            if not chunk_indices.is_contiguous():
                raise ValueError("chunk_indices must be contiguous.")
        if chunk_valid_sizes is not None:
            if chunk_valid_sizes.dtype != torch.int32:
                raise TypeError("chunk_valid_sizes must be int32.")
            if not chunk_valid_sizes.is_contiguous():
                raise ValueError("chunk_valid_sizes must be contiguous.")
        if (chunk_indices is None) != (chunk_valid_sizes is None):
            raise ValueError("chunk_indices and chunk_valid_sizes must be provided together.")
        effective_block_dim = min(block_dim, num_matrices)
        lib.call_kernel(
            effective_block_dim,
            stream_ptr,
            _torch_to_ctypes(tensor_out),
            _torch_to_ctypes(tensor_in),
            _torch_to_ctypes(minus_identity),
            matrix_size,
            num_matrices,
            num_bsnd_heads,
            _torch_to_ctypes(cu_seqlens)
            if cu_seqlens is not None
            else ctypes.c_void_p(),
            _torch_to_ctypes(chunk_indices)
            if chunk_indices is not None
            else ctypes.c_void_p(),
            _torch_to_ctypes(chunk_valid_sizes)
            if chunk_valid_sizes is not None
            else ctypes.c_void_p(),
        )

    return tri_inv_func


# ---------------------------------------------------------------------------
# Convenience: compile + load in one call
# ---------------------------------------------------------------------------

def jit_compile(src_path: str, verbose: bool = True, clean_up: bool = False):
    """Compile *src_path* and return the kernel callable."""
    lib_path = compile_cpp(src_path, verbose=verbose)
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func
