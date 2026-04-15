from __future__ import annotations

import math

import torch

import pto_dynamic_common  # noqa: F401
from dynamic_kernel_libs import run_chunk_h_kernel
from run_chunk_cumsum_dynamic_bsnd import benchmark_ms, total_chunks_from_cu


torch_npu = torch.npu  # noqa: F401
CHUNK = 128
RTOL = 1e-3
ATOL = 1e-3
FS_RTOL = 5e-2
FS_ATOL = 64.0


def ref_chunk_h_bsnd(
    k: torch.Tensor,
    w_packed: torch.Tensor,
    u_packed: torch.Tensor,
    g_packed: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, total_t, num_heads, hidden = k.shape
    if cu_seqlens is None:
        spans = [(b, 0, total_t) for b in range(batch)]
        num_seqs = batch
    else:
        spans = [(i, int(cu_seqlens[i]), int(cu_seqlens[i + 1])) for i in range(len(cu_seqlens) - 1)]
        num_seqs = len(spans)
    total_chunks = w_packed.shape[0]
    s = torch.zeros((total_chunks, num_heads, hidden, hidden), device=k.device, dtype=torch.float16)
    new_v = torch.zeros((total_chunks, num_heads, chunk_size, hidden), device=k.device, dtype=torch.float16)
    final_s = torch.zeros((num_seqs, num_heads, hidden, hidden), device=k.device, dtype=torch.float16)
    chunk_offset = 0
    for seq_idx, bos, eos in spans:
        state = torch.zeros((num_heads, hidden, hidden), device=k.device, dtype=torch.float32)
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            valid = end - start
            s[chunk_offset] = state.to(torch.float16)
            ws = torch.matmul(w_packed[chunk_offset], state.to(torch.float16)).float()
            nv = u_packed[chunk_offset, :, :valid].float() - ws[:, :valid]
            new_v[chunk_offset, :, :valid] = nv.to(torch.float16)
            g_chunk = g_packed[chunk_offset, :, :valid].float()
            g_last = g_chunk[:, valid - 1].view(num_heads, 1, 1)
            coeff = torch.exp(g_last - g_chunk.view(num_heads, valid, 1))
            k_chunk = k[seq_idx if cu_seqlens is None else 0, start:end].permute(1, 0, 2).contiguous().float()
            k_scaled = (k_chunk * coeff).to(torch.float16)
            kv = torch.matmul(k_scaled.transpose(-1, -2), nv.to(torch.float16)).float()
            state = state * torch.exp(g_last) + kv
            chunk_offset += 1
        final_s[seq_idx] = state.to(torch.float16)
    return s, new_v, final_s


def run_case(label: str, *, shape: tuple[int, int, int, int], cu_seqlens: list[int] | None):
    k = torch.randn(shape, device="npu", dtype=torch.float16)
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
    w_packed = torch.randn((total_chunks, shape[2], CHUNK, shape[3]), device="npu", dtype=torch.float16)
    u_packed = torch.randn_like(w_packed)
    g_packed = torch.randn((total_chunks, shape[2], CHUNK), device="npu", dtype=torch.float32)
    seq_count = batch_override if batch_override is not None else shape[0]
    s_out = torch.zeros((total_chunks, shape[2], shape[3], shape[3]), device="npu", dtype=torch.float16)
    nv_out = torch.zeros_like(w_packed)
    fs_out = torch.zeros((seq_count, shape[2], shape[3], shape[3]), device="npu", dtype=torch.float16)
    ref_s, ref_nv, ref_fs = ref_chunk_h_bsnd(
        k,
        w_packed,
        u_packed,
        g_packed,
        chunk_size=CHUNK,
        cu_seqlens=cu_tensor,
    )

    def launch():
        run_chunk_h_kernel(
            k,
            w_packed,
            u_packed,
            g_packed,
            s_out,
            nv_out,
            fs_out,
            chunk_size=CHUNK,
            cu_seqlens=cu_tensor,
            batch_size_override=batch_override,
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(s_out.cpu(), ref_s.cpu(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(nv_out.cpu(), ref_nv.cpu(), rtol=RTOL, atol=ATOL)
    fs_cpu = torch.nan_to_num(fs_out.cpu(), nan=0.0, posinf=65504.0, neginf=-65504.0)
    ref_fs_cpu = torch.nan_to_num(ref_fs.cpu(), nan=0.0, posinf=65504.0, neginf=-65504.0)
    torch.testing.assert_close(fs_cpu, ref_fs_cpu, rtol=FS_RTOL, atol=FS_ATOL)

    ms = benchmark_ms(launch, warmup=3, repeat=10)
    print(f"{label}: passed, {ms:.3f} ms")


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    run_case("fixed-bsnd-chunk-h", shape=(2, 256, 2, 128), cu_seqlens=None)
    run_case("packed-varlen-bsnd-chunk-h", shape=(1, 161, 2, 128), cu_seqlens=[0, 17, 96, 161])
    print("Dynamic BSND chunk_h checks passed.")


if __name__ == "__main__":
    main()
