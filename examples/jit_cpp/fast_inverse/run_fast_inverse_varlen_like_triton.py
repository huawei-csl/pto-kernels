"""
Standalone varlen BSND correctness runner that mirrors the Triton unit tests:
https://github.com/fla-org/flash-linear-attention/blob/v0.4.2/tests/ops/test_solve_tril.py

But changes:
1. uses fp16 inputs because the PTO kernel currently supports fp16 only
2. emulates `chunk_scaled_dot_kkt_fwd` in PyTorch because Triton is unavailable

Run from the fast_inverse/ directory:

    export PTO_LIB_PATH=/sources/pto-isa
    python run_fast_inverse_varlen_like_triton.py
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa

from jit_util_fast_inverse import jit_compile


torch.manual_seed(42)
np.random.seed(42)


def _make_minus_identity(matrix_size: int, device: torch.device) -> torch.Tensor:
    minus_identity = torch.zeros(
        (matrix_size, matrix_size),
        dtype=torch.float16,
        device=device,
    )
    minus_identity.fill_diagonal_(-1)
    return minus_identity


def _count_varlen_chunks(cu_seqlens: torch.Tensor, chunk_size: int) -> int:
    return sum(
        (int(eos) - int(bos) + chunk_size - 1) // chunk_size
        for bos, eos in zip(
            cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
        )
    )


def _chunk_scaled_dot_kkt_fwd_emulated(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    t_total = int(cu_seqlens[-1].item())
    num_heads = k.shape[2]
    A = torch.zeros((1, t_total, num_heads, chunk_size), dtype=k.dtype, device=k.device)

    for bos, eos in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
    ):
        for chunk_start in range(bos, eos, chunk_size):
            chunk_end = min(chunk_start + chunk_size, eos)
            actual_size = chunk_end - chunk_start
            k_chunk = k[:, chunk_start:chunk_end].transpose(1, 2).to(torch.float32)
            beta_chunk = (
                beta[:, chunk_start:chunk_end]
                .transpose(1, 2)
                .unsqueeze(-1)
                .to(torch.float32)
            )
            scores = torch.matmul(k_chunk, k_chunk.transpose(-1, -2))
            scores = torch.tril(scores * beta_chunk, diagonal=-1).to(k.dtype)
            A[:, chunk_start:chunk_end, :, :actual_size] = scores.transpose(1, 2)

    return A


def _reference_inverse(
    A: torch.Tensor, cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    A_cpu = A.cpu().to(torch.float64)
    ref = torch.zeros_like(A_cpu, dtype=torch.float64)
    for bos, eos in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
    ):
        for chunk_start in range(bos, eos, chunk_size):
            actual_size = min(chunk_size, eos - chunk_start)
            ref[:, chunk_start : chunk_start + actual_size, :, :actual_size] = (
                torch.inverse(
                    A_cpu[
                        :, chunk_start : chunk_start + actual_size, :, :actual_size
                    ].transpose(1, 2)
                    + torch.eye(actual_size, dtype=torch.float64)[None, None, ...]
                ).transpose(1, 2)
            )
    return ref


def _transpose_valid_chunks(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    transposed = torch.zeros_like(A)
    for bos, eos in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
    ):
        for chunk_start in range(bos, eos, chunk_size):
            actual_size = min(chunk_size, eos - chunk_start)
            chunk = A[:, chunk_start : chunk_start + actual_size, :, :actual_size]
            transposed[:, chunk_start : chunk_start + actual_size, :, :actual_size] = (
                chunk.transpose(1, 3)
            )
    return transposed


def _run_pto_varlen(
    tri_inv_func, A: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    chunk_size = A.shape[-1]
    num_heads = A.shape[-2]
    num_matrices = _count_varlen_chunks(cu_seqlens, chunk_size) * num_heads
    tensor_out = torch.zeros_like(A, dtype=torch.float32)
    minus_identity = _make_minus_identity(chunk_size, A.device)

    torch.npu.synchronize()
    tri_inv_func(
        tensor_out,
        A,
        minus_identity,
        chunk_size,
        num_matrices,
        num_heads,
        cu_seqlens=cu_seqlens,
    )
    torch.npu.synchronize()
    return tensor_out.cpu().to(torch.float64)


def _run_case(
    tri_inv_func,
    H: int,
    D: int,
    chunk_size: int,
    cu_seqlens_list: list[int],
    atol: float = 5e-4,
    rtol: float = 5e-2,
    ftol: float = 1e-4,
) -> None:
    device = torch.device("npu:0")
    T = cu_seqlens_list[-1]
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

    # Match the Triton varlen test structure, using fp16 instead of bf16.
    k = F.normalize(
        torch.randn((1, T, H, D), dtype=torch.float16, device=device), dim=-1
    )
    beta = torch.randn((1, T, H), dtype=torch.float16, device=device).sigmoid()
    A = _chunk_scaled_dot_kkt_fwd_emulated(
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    ref = _reference_inverse(A, cu_seqlens, chunk_size)
    tri = _run_pto_varlen(
        tri_inv_func,
        _transpose_valid_chunks(A, cu_seqlens, chunk_size),
        cu_seqlens,
    )
    tri = _transpose_valid_chunks(tri, cu_seqlens, chunk_size)

    frob = torch.sqrt(torch.sum((ref - tri) ** 2) / torch.sum(ref**2)).item()
    torch.testing.assert_close(tri, ref, atol=atol, rtol=rtol)
    assert frob <= ftol, f"Frobenius error {frob:.2e} > {ftol:.2e}"


def main() -> int:
    if "PTO_LIB_PATH" not in os.environ:
        fallback = "/sources/pto-isa"
        if os.path.exists(fallback):
            os.environ["PTO_LIB_PATH"] = fallback

    torch.npu.set_device("npu:0")

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fast_inverse.cpp")
    print(f"Compiling {src} ...")
    tri_inv_func = jit_compile(src)
    print("Compilation successful.\n")

    cases = [
        (4, 64, 16, [0, 15]),
        (4, 64, 32, [0, 256, 500, 1000]),
        (4, 100, 64, [0, 15, 100, 300, 1200, 2000]),
        (4, 64, 16, [0, 1, 100, 300, 1200, 2048]),
        (4, 128, 32, [0, 200, 512, 1200, 2048]),
    ]

    total = 0
    passed = 0
    print("=== Varlen Like Triton ===")
    for H, D, chunk_size, cu_seqlens in cases:
        total += 1
        label = f"H={H} D={D} chunk_size={chunk_size} cu_seqlens={cu_seqlens}"
        try:
            _run_case(tri_inv_func, H, D, chunk_size, cu_seqlens)
            print(f"  PASS  {label}")
            passed += 1
        except Exception as err:
            print(f"  FAIL  {label}: {err}")

    print(f"\n{passed}/{total} cases passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
