# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
"""Standalone runner for the tri_inv_rec_unroll kernel.

Loads the single-kernel shared library built via `make compile_tri_inv_rec_unroll`
(or `make compile_a5_tri_inv_rec_unroll`) and drives the low-level
`pto_launch_tri_inv_rec_unroll_fp16` launch shim directly, replicating the host
wrapper in csrc/host/torch_tri_inv_rec_unroll.h. The reference and input matrices
follow tests/test_tri_inv_rec_unroll.py.
"""
import os
import ctypes

import numpy as np
import torch

# Importing the built extension initializes the ACL runtime / mix-kernel
# environment. Without it, the raw ctypes launch of this cube (mix) kernel
# lands on the vector core and the `__DAV_CUBE__` body never runs (all-zero
# output). The symbol itself is not called here — the import side effect is
# what matters.
import pto_kernels  # noqa: F401

# Select device "cpu" or "npu"
DEVICE = "npu:0"

NUM_AI_CORES = int(getattr(torch.npu.get_device_properties("npu"), "cube_core_num", 20))


def random_tri_matrix(n, block_dim_x, block_dim_y, scale=0.1, is_lower=False):
    """Strictly (upper/lower) triangular random matrices; see the test module."""
    if is_lower:
        return scale * torch.tril(
            torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=-1
        )
    return scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)


def linalg_inv(U: torch.Tensor) -> torch.Tensor:
    """CPU fp64 golden: inverse of (U + I) for each matrix on the last two dims."""
    n = U.shape[-1]
    identity = np.eye(n, dtype=np.double)
    golden = np.zeros(U.shape)
    for x in range(U.shape[0]):
        for y in range(U.shape[1]):
            golden[x, y] = np.linalg.inv(
                U[x, y].double().numpy().astype(np.double) + identity
            )
    return torch.from_numpy(golden)


if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        lib_path = "build/lib/libkernel_tri_inv_rec_unroll.so"
        lib_path = os.path.abspath(lib_path)
        lib = ctypes.CDLL(lib_path)
        print(f"Loaded library from {lib_path}")

        lib.pto_launch_tri_inv_rec_unroll_fp16.restype = None
        lib.pto_launch_tri_inv_rec_unroll_fp16.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # tensor_out
            ctypes.c_void_p,  # tensor_in
            ctypes.c_void_p,  # minus_eye_in
            ctypes.c_uint32,  # matrix_size
            ctypes.c_uint32,  # num_matrices
            ctypes.c_uint32,  # num_bsnd_heads
            ctypes.c_uint32,  # is_lower
            ctypes.c_void_p,  # cu_seqlens (nullptr => non-strided path)
        ]
        stream_ptr = torch.npu.current_stream()._as_parameter_  # noqa

        def torch_to_ctypes(tensor):
            return ctypes.c_void_p(tensor.data_ptr())

        # Problem sizes (non-BSND path). matrix_size == n, num_matrices == bx*by.
        n = 64
        block_dim_x = 4
        block_dim_y = 8
        is_lower = False
        input_dtype = torch.float16

        # Build input the same way the test does.
        U = random_tri_matrix(n, block_dim_x, block_dim_y)
        if is_lower:
            U = U.transpose(-1, -2).contiguous().to(input_dtype)
        else:
            U = U.contiguous().to(input_dtype)

        golden_cpu = linalg_inv(U)

        matrix_size = n
        num_matrices = block_dim_x * block_dim_y

        M_in = U.to(DEVICE)
        M_out = torch.zeros_like(M_in)

        # -I on the diagonal, matching the host wrapper (I_neg.fill_diagonal_(-1)).
        I_neg = torch.zeros(
            (matrix_size, matrix_size), dtype=input_dtype, device=DEVICE
        )
        I_neg.fill_diagonal_(-1)

        block_dim = min(NUM_AI_CORES, num_matrices)

        def launch(out):
            lib.pto_launch_tri_inv_rec_unroll_fp16(
                block_dim,
                stream_ptr,
                torch_to_ctypes(out),
                torch_to_ctypes(M_in),
                torch_to_ctypes(I_neg),
                matrix_size,
                num_matrices,
                0,  # num_bsnd_heads (0 => non-strided)
                int(is_lower),
                None,  # cu_seqlens
            )
            torch.npu.synchronize()

        # The very first launch of this cube (mix) kernel on a fresh runtime
        # returns zeros — a one-time on-device warmup is needed before results
        # are valid. Discard a warmup launch, then run for real.
        launch(torch.zeros_like(M_in))
        launch(M_out)

        actual_cpu = M_out.cpu().to(torch.float64)
        frob_error = torch.sqrt(
            torch.sum((golden_cpu - actual_cpu) ** 2) / torch.sum(golden_cpu**2)
        )

        is_close = np.allclose(
            actual_cpu.numpy(), golden_cpu.numpy(), atol=5e-5, rtol=0.1
        )
        print(f"Input shape: {tuple(M_in.shape)}  is_lower={is_lower}")
        print(f"Is all close? {is_close}")
        print(f"Frobenius rel. error: {frob_error.item():.3e}")
        print("actual :", actual_cpu[0, 0, 0, :6])
        print("golden :", golden_cpu[0, 0, 0, :6])
    finally:
        del lib  # triggers dlclose in CPython
