from __future__ import annotations

import math

import torch

import pto_dynamic_common  # noqa: F401
from dynamic_kernel_libs import run_chunk_cumsum_kernel


torch_npu = torch.npu  # noqa: F401
CHUNK = 128
RTOL = 1e-5
ATOL = 1e-5


def total_chunks_from_cu(cu_seqlens: list[int], chunk_size: int) -> int:
    return sum(math.ceil((e - s) / chunk_size) for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False))


def ref_chunk_cumsum_bsnd(
    g: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    _, total_t, num_heads = g.shape
    if cu_seqlens is None:
        spans = [(b, 0, total_t) for b in range(g.shape[0])]
        total_chunks = g.shape[0] * math.ceil(total_t / chunk_size)
    else:
        spans = [(i, int(cu_seqlens[i]), int(cu_seqlens[i + 1])) for i in range(len(cu_seqlens) - 1)]
        total_chunks = total_chunks_from_cu(cu_seqlens.tolist(), chunk_size)
    out = torch.zeros((total_chunks, num_heads, chunk_size), device=g.device, dtype=g.dtype)
    chunk_offset = 0
    for seq_idx, bos, eos in spans:
        batch_idx = seq_idx if cu_seqlens is None else 0
        for start in range(bos, eos, chunk_size):
            end = min(start + chunk_size, eos)
            seq_chunk = g[batch_idx, start:end].transpose(0, 1).contiguous()
            out[chunk_offset, :, : end - start] = torch.cumsum(seq_chunk, dim=-1)
            chunk_offset += 1
    return out


def benchmark_ms(fn, warmup: int = 5, repeat: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / repeat


def run_case(label: str, *, shape: tuple[int, int, int], cu_seqlens: list[int] | None):
    g = torch.randn(shape, device="npu", dtype=torch.float32)
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
    out = torch.zeros((total_chunks, shape[2], CHUNK), device="npu", dtype=torch.float32)
    ref = ref_chunk_cumsum_bsnd(g, chunk_size=CHUNK, cu_seqlens=cu_tensor)

    def launch():
        run_chunk_cumsum_kernel(
            g,
            out,
            chunk_size=CHUNK,
            cu_seqlens=cu_tensor,
            batch_size_override=batch_override,
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=RTOL, atol=ATOL)

    ms = benchmark_ms(launch)
    moved_bytes = g.numel() * g.element_size() + out.numel() * out.element_size()
    gib_per_s = moved_bytes / (ms * 1e-3) / (1024**3)
    print(f"{label}: passed, {ms:.3f} ms, {gib_per_s:.1f} GiB/s")


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")
    run_case("fixed-bsnd", shape=(2, 256, 2), cu_seqlens=None)
    run_case("packed-varlen-bsnd", shape=(1, 161, 2), cu_seqlens=[0, 17, 96, 161])
    print("Dynamic BSND chunk_cumsum checks passed.")


if __name__ == "__main__":
    main()
