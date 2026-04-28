from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import pto_dynamic_common  # noqa: F401
from dynamic_kernel_libs import run_chunk_o_kernel
from run_chunk_cumsum_dynamic_bsnd import benchmark_ms, total_chunks_from_cu


torch_npu = torch.npu  # noqa: F401
CHUNK = 128
RTOL = 7e-2
ATOL = 7e-2


def ref_chunk_o_bsnd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s_packed: torch.Tensor,
    g_packed: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.zeros_like(v)
    batch, total_t, _, _ = q.shape
    if cu_seqlens is None:
        spans = [(b, 0, total_t) for b in range(batch)]
    else:
        spans = [(i, int(cu_seqlens[i]), int(cu_seqlens[i + 1])) for i in range(len(cu_seqlens) - 1)]
    chunk_offset = 0
    for seq_idx, bos, eos in spans:
        batch_idx = seq_idx if cu_seqlens is None else 0
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            valid = end - start
            q_c = q[batch_idx, start:end].permute(1, 0, 2).contiguous().float()
            k_c = k[batch_idx, start:end].permute(1, 0, 2).contiguous().float()
            v_c = v[batch_idx, start:end].permute(1, 0, 2).contiguous().float()
            g_c = g_packed[chunk_offset, :, :valid].float()
            s_c = s_packed[chunk_offset].float()
            term1 = torch.matmul(q_c.to(torch.float16), s_c.to(torch.float16)).to(torch.float16).float()
            term1 = term1 * torch.exp(g_c).unsqueeze(-1)
            qkt = torch.matmul(q_c.to(torch.float16), k_c.transpose(-1, -2).to(torch.float16)).to(torch.float16).float()
            gamma = torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))
            qkt = (qkt * gamma).to(torch.float16).float()
            qkt = torch.tril(qkt, diagonal=0)
            term2 = torch.matmul(qkt.to(torch.float16).float(), v_c.to(torch.float16).float())
            out[batch_idx, start:end] = (term1 + term2).permute(1, 0, 2).to(out.dtype)
            chunk_offset += 1
    return out


def run_case(label: str, *, shape: tuple[int, int, int, int], cu_seqlens: list[int] | None):
    q = torch.randn(shape, device="npu", dtype=torch.float16)
    k = torch.randn(shape, device="npu", dtype=torch.float16)
    v = torch.randn(shape, device="npu", dtype=torch.float16)
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
    s_packed = torch.randn((total_chunks, shape[2], shape[3], shape[3]), device="npu", dtype=torch.float16)
    g_base = F.logsigmoid(torch.randn((total_chunks, shape[2], CHUNK), device="npu", dtype=torch.float32))
    g_packed = torch.cumsum(g_base, dim=-1)
    out = torch.zeros_like(v)
    ref = ref_chunk_o_bsnd(
        q,
        k,
        v,
        s_packed,
        g_packed,
        chunk_size=CHUNK,
        cu_seqlens=cu_tensor,
    )

    def launch():
        run_chunk_o_kernel(
            q,
            k,
            v,
            s_packed,
            g_packed,
            out,
            chunk_size=CHUNK,
            cu_seqlens=cu_tensor,
            batch_size_override=batch_override,
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=RTOL, atol=ATOL)

    ms = benchmark_ms(launch, warmup=3, repeat=20)
    total_flops = 4.0 * total_chunks * shape[2] * CHUNK * CHUNK * shape[3]
    tflops = total_flops / (ms * 1e-3) / 1e12
    print(f"{label}: passed, {ms:.3f} ms, {tflops:.2f} TFLOP/s")


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    run_case("fixed-bsnd-chunk-o", shape=(2, 256, 2, 128), cu_seqlens=None)
    run_case("packed-varlen-bsnd-chunk-o", shape=(1, 161, 2, 128), cu_seqlens=[0, 17, 96, 161])
    print("Dynamic BSND chunk_o checks passed.")


if __name__ == "__main__":
    main()
