from __future__ import annotations

import math

import torch

import pto_dynamic_common  # noqa: F401
from dynamic_kernel_libs import (
    pack_bsh_tensor,
    pack_bshd_tensor,
    run_wy_fast_kernel,
)
from run_chunk_cumsum_dynamic_bsnd import benchmark_ms, total_chunks_from_cu


torch_npu = torch.npu  # noqa: F401
CHUNK = 128
RTOL = 1e-3
ATOL = 1e-3


def ref_wy_fast_bsnd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_packed: torch.Tensor,
    a_packed: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_packed = pack_bshd_tensor(k, chunk_size=chunk_size, cu_seqlens=cu_seqlens).float()
    v_packed = pack_bshd_tensor(v, chunk_size=chunk_size, cu_seqlens=cu_seqlens).float()
    beta_packed = pack_bsh_tensor(beta, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    a_float = a_packed.float()
    a2 = (a_float * beta_packed.unsqueeze(-1)).to(torch.float16)
    a1 = (a_float * (beta_packed * torch.exp(g_packed.float())).unsqueeze(-1)).to(torch.float16)
    w = torch.matmul(a1.float(), k_packed).to(torch.float16)
    u = torch.matmul(a2.float(), v_packed).to(torch.float16)
    return w, u


def run_case(label: str, *, shape: tuple[int, int, int, int], cu_seqlens: list[int] | None):
    k = torch.randn(shape, device="npu", dtype=torch.float16)
    v = torch.randn(shape, device="npu", dtype=torch.float16)
    beta = torch.rand(shape[:-1], device="npu", dtype=torch.float16)
    g_packed = None
    cu_tensor = (
        torch.tensor(cu_seqlens, device="npu", dtype=torch.int32)
        if cu_seqlens is not None
        else None
    )
    batch_override = (len(cu_seqlens) - 1) if cu_seqlens is not None else None
    total_chunks = (
        total_chunks_from_cu(cu_seqlens, CHUNK)
        if cu_seqlens is not None
        else shape[0] * math.ceil(shape[1] / CHUNK)
    )
    g_packed = torch.randn((total_chunks, shape[2], CHUNK), device="npu", dtype=torch.float32)
    a_packed = torch.randn((total_chunks, shape[2], CHUNK, CHUNK), device="npu", dtype=torch.float16)
    w_out = torch.zeros((total_chunks, shape[2], CHUNK, shape[3]), device="npu", dtype=torch.float16)
    u_out = torch.zeros_like(w_out)
    ref_w, ref_u = ref_wy_fast_bsnd(
        k,
        v,
        beta,
        g_packed,
        a_packed,
        chunk_size=CHUNK,
        cu_seqlens=cu_tensor,
    )

    def launch():
        run_wy_fast_kernel(
            k,
            v,
            beta,
            g_packed,
            a_packed,
            w_out,
            u_out,
            chunk_size=CHUNK,
            cu_seqlens=cu_tensor,
            batch_size_override=batch_override,
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(w_out.cpu(), ref_w.cpu(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(u_out.cpu(), ref_u.cpu(), rtol=RTOL, atol=ATOL)

    ms = benchmark_ms(launch, warmup=10, repeat=50)
    total_flops = 4.0 * total_chunks * shape[2] * CHUNK * CHUNK * shape[3]
    tflops = total_flops / (ms * 1e-3) / 1e12
    print(f"{label}: passed, {ms:.3f} ms, {tflops:.2f} TFLOP/s")


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    run_case("fixed-bsnd-wy", shape=(2, 256, 2, 128), cu_seqlens=None)
    run_case("packed-varlen-bsnd-wy", shape=(1, 161, 2, 128), cu_seqlens=[0, 17, 96, 161])
    print("Dynamic BSND wy_fast checks passed.")


if __name__ == "__main__":
    main()
