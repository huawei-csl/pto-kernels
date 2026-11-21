"""Run KDA output tail scenarios on the default GDN_H=16, GDN_D=GDN_C=128 build.

Usage: python tests/scenarios/kda_output_tails.py
"""

import torch
import torch_npu  # noqa: F401

from pto_kernels import pto_kda_chunk_o


def run_case(lengths, gate_scale, varlen):
    heads, dim, chunk_size = 16, 128, 128
    tokens = sum(lengths)
    chunks = sum((length + chunk_size - 1) // chunk_size for length in lengths)
    q, k, v = [(torch.randn(tokens, heads, dim) / 16).half() for _ in range(3)]
    snapshots = (torch.randn(chunks, heads, dim, dim) / 16).half()
    steps = -torch.rand(tokens, heads, dim) * gate_scale
    gates = torch.empty_like(steps)
    expected = torch.empty(tokens, heads, dim, dtype=torch.float64)
    boundaries = [0]
    chunk_index = 0
    for length in lengths:
        begin = boundaries[-1]
        end = begin + length
        for start in range(begin, end, chunk_size):
            stop = min(start + chunk_size, end)
            gates[start:stop] = steps[start:stop].cumsum(0)
            state = snapshots[chunk_index].double().clone()
            previous_gate = torch.zeros(heads, dim, dtype=torch.float64)
            for token in range(start, stop):
                gate = gates[token].double()
                state *= (gate - previous_gate).exp().unsqueeze(-1)
                state += k[token].double().unsqueeze(-1) * v[token].double().unsqueeze(
                    -2
                )
                expected[token] = (q[token].double().unsqueeze(-1) * state).sum(-2)
                previous_gate = gate
            chunk_index += 1
        boundaries.append(end)

    metadata = {}
    if varlen:
        metadata["cu_seqlens"] = torch.tensor(boundaries, dtype=torch.int32).npu()
    actual = pto_kda_chunk_o(
        q.permute(1, 0, 2).contiguous().npu(),
        k.permute(1, 0, 2).contiguous().npu(),
        v.npu(),
        snapshots.npu(),
        gates.permute(1, 0, 2).contiguous().npu(),
        torch.ones(chunk_size, chunk_size).tril().npu(),
        batch_size=len(lengths),
        seq_len=0 if varlen else lengths[0],
        total_chunks=chunks,
        **metadata,
    ).cpu()
    if not torch.allclose(actual.double(), expected, rtol=1e-2, atol=1e-4):
        error = (actual.double() - expected).abs().max().item()
        raise RuntimeError(
            f"lengths={lengths}, gate_scale={gate_scale}, varlen={varlen}: "
            f"max absolute error {error}"
        )
    print(f"passed lengths={lengths}, gate_scale={gate_scale}, varlen={varlen}")


def main():
    torch.manual_seed(23)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    lengths = [1, 63, 64, 65, 127, 128, 129]
    for gate_scale in (0.0, 0.01, 4.0):
        for length in lengths:
            run_case([length], gate_scale, varlen=False)
        run_case(lengths, gate_scale, varlen=True)


if __name__ == "__main__":
    main()
