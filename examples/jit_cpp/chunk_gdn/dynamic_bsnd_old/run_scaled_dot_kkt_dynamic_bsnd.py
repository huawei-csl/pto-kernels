from __future__ import annotations

import math

import torch

import pto_dynamic_common  # noqa: F401
from dynamic_kernel_libs import run_chunk_cumsum_kernel, run_scaled_dot_kkt_kernel
from run_chunk_cumsum_dynamic_bsnd import benchmark_ms, total_chunks_from_cu


torch_npu = torch.npu  # noqa: F401
CHUNK = 128
RTOL = 1e-3
ATOL = 1e-3


def ref_kkt_bsnd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_packed: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, total_t, num_heads, _ = k.shape
    if cu_seqlens is None:
        spans = [(b, 0, total_t) for b in range(batch)]
        total_chunks = batch * math.ceil(total_t / chunk_size)
    else:
        spans = [(i, int(cu_seqlens[i]), int(cu_seqlens[i + 1])) for i in range(len(cu_seqlens) - 1)]
        total_chunks = total_chunks_from_cu(cu_seqlens.tolist(), chunk_size)
    out = torch.zeros((total_chunks, num_heads, chunk_size, chunk_size), device=k.device, dtype=torch.float16)
    chunk_offset = 0
    for seq_idx, bos, eos in spans:
        batch_idx = seq_idx if cu_seqlens is None else 0
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            valid = end - start
            k_c = k[batch_idx, start:end].transpose(0, 1).contiguous().float()
            beta_c = beta[batch_idx, start:end].transpose(0, 1).contiguous().float()
            g_c = g_packed[chunk_offset, :, :valid].float()
            kkt = torch.matmul(k_c, k_c.transpose(-1, -2))
            gamma = torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))
            block = (kkt * beta_c.unsqueeze(-1) * gamma).tril(-1)
            out[chunk_offset, :, :valid, :valid] = block.to(torch.float16)
            chunk_offset += 1
    return out


def run_case(label: str, *, shape: tuple[int, int, int, int], cu_seqlens: list[int] | None):
    k = torch.randn(shape, device="npu", dtype=torch.float16)
    beta = torch.rand(shape[:-1], device="npu", dtype=torch.float16)
    g = torch.randn(shape[:-1], device="npu", dtype=torch.float32)
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
    g_packed = torch.zeros((total_chunks, shape[2], CHUNK), device="npu", dtype=torch.float32)
    run_chunk_cumsum_kernel(
        g,
        g_packed,
        chunk_size=CHUNK,
        cu_seqlens=cu_tensor,
        batch_size_override=batch_override,
    )
    workspace = torch.zeros((total_chunks, shape[2], CHUNK, CHUNK), device="npu", dtype=torch.float16)
    out = torch.zeros_like(workspace)
    mask = torch.tril(torch.ones((CHUNK, CHUNK), device="npu", dtype=torch.float32), diagonal=-1)
    ref = ref_kkt_bsnd(k, beta, g_packed, chunk_size=CHUNK, cu_seqlens=cu_tensor)

    def launch():
        run_scaled_dot_kkt_kernel(
            k,
            beta,
            g_packed,
            mask,
            workspace,
            out,
            chunk_size=CHUNK,
            cu_seqlens=cu_tensor,
            batch_size_override=batch_override,
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=RTOL, atol=ATOL)

    ms = benchmark_ms(launch, warmup=10, repeat=50)
    total_flops = 2.0 * total_chunks * shape[2] * CHUNK * CHUNK * shape[3]
    tflops = total_flops / (ms * 1e-3) / 1e12
    print(f"{label}: passed, {ms:.3f} ms, {tflops:.2f} TFLOP/s")


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    run_case("fixed-bsnd-kkt", shape=(2, 256, 2, 128), cu_seqlens=None)
    run_case("packed-varlen-bsnd-kkt", shape=(1, 161, 2, 128), cu_seqlens=[0, 17, 96, 161])
    print("Dynamic BSND scaled_dot_kkt checks passed.")


if __name__ == "__main__":
    main()
