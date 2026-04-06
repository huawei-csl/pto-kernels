import math
import os
from functools import lru_cache

import torch
import torch_npu  # noqa: F401

from jit_util_linear_attention import BLOCK_DIM, get_causal_mask, jit_compile

DTYPE = torch.float16
RTOL = 1e-2


def _to_seq_first(x: torch.Tensor, head_first: bool) -> torch.Tensor:
    return x.transpose(1, 2).contiguous() if head_first else x.contiguous()


def _from_seq_first(x: torch.Tensor, head_first: bool) -> torch.Tensor:
    return x.transpose(1, 2).contiguous() if head_first else x.contiguous()


def _apply_gating(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor | None,
    *,
    head_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if g is None:
        return q, k
    gate = torch.exp(g.float()).to(q.dtype)
    inv_gate = torch.exp(-g.float()).to(k.dtype)
    if head_first:
        return q * gate.unsqueeze(-1), k * inv_gate.unsqueeze(-1)
    return q * gate.unsqueeze(-1), k * inv_gate.unsqueeze(-1)


def _build_precomputed_h(
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    head_first: bool,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    k_seq = _to_seq_first(k, head_first).float()
    v_seq = _to_seq_first(v, head_first).float()
    _, total_t, num_heads, hidden = k_seq.shape
    states = []

    if cu_seqlens is None:
        batch = k_seq.shape[0]
        state = torch.zeros((batch, num_heads, hidden, hidden), device=k.device, dtype=torch.float32)
        chunk_num = math.ceil(k_seq.shape[1] / chunk_size)
        for i in range(chunk_num):
            states.append(state.to(DTYPE))
            start = i * chunk_size
            end = min(start + chunk_size, k_seq.shape[1])
            state = state + torch.einsum(
                "bthd,bthe->bhde",
                k_seq[:, start:end],
                v_seq[:, start:end],
            )
        return torch.stack(states, dim=1).contiguous().view(batch * chunk_num, num_heads, hidden, hidden)

    if head_first:
        raise ValueError("cu_seqlens is only supported with seq-first inputs.")

    state = torch.zeros((num_heads, hidden, hidden), device=k.device, dtype=torch.float32)
    for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
        state.zero_()
        for start in range(bos, eos, chunk_size):
            states.append(state.to(DTYPE))
            end = min(start + chunk_size, eos)
            state = state + torch.einsum(
                "thd,the->hde",
                k_seq[0, start:end],
                v_seq[0, start:end],
            )
    return torch.stack(states, dim=0).contiguous()


def ref_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    g: torch.Tensor | None = None,
    head_first: bool = True,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    q_scaled, k_scaled = _apply_gating(q, k, g, head_first=head_first)
    q_seq = _to_seq_first(q_scaled, head_first).float()
    k_seq = _to_seq_first(k_scaled, head_first).float()
    v_seq = _to_seq_first(v, head_first).float()
    out = torch.zeros_like(v_seq, dtype=torch.float32)

    if cu_seqlens is None:
        batch, seq_len, num_heads, hidden = q_seq.shape
        for b in range(batch):
            h = torch.zeros((num_heads, hidden, hidden), device=q.device, dtype=torch.float32)
            for i in range(seq_len):
                k_i = k_seq[b, i]
                v_i = v_seq[b, i]
                h = h + torch.einsum("hd,he->hde", k_i, v_i)
                out[b, i] = torch.einsum("hd,hde->he", q_seq[b, i], h)
    else:
        _, _, num_heads, hidden = q_seq.shape
        for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False):
            h = torch.zeros((num_heads, hidden, hidden), device=q.device, dtype=torch.float32)
            for i in range(bos, eos):
                k_i = k_seq[0, i]
                v_i = v_seq[0, i]
                h = h + torch.einsum("hd,he->hde", k_i, v_i)
                out[0, i] = torch.einsum("hd,hde->he", q_seq[0, i], h)

    return _from_seq_first(out.to(DTYPE), head_first)


@lru_cache(maxsize=None)
def _compiled_kernel(src: str, h: int, d: int, c: int):
    return jit_compile(src, num_heads=h, hidden_size=d, chunk_size=c)


def run_kernel(
    src: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    g: torch.Tensor | None = None,
    head_first: bool = True,
    cu_seqlens: torch.Tensor | None = None,
):
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have identical shapes.")
    if cu_seqlens is not None and head_first:
        raise ValueError("cu_seqlens is only supported with seq-first inputs.")

    num_heads = q.shape[1] if head_first else q.shape[2]
    hidden = q.shape[-1]
    linear_attention_func = _compiled_kernel(src, num_heads, hidden, chunk_size)
    causal_mask = get_causal_mask(chunk_size, DTYPE, q.device.index or 0)

    if g is None and head_first and cu_seqlens is None and q.shape[2] % chunk_size == 0:
        b, _, l, d = q.shape
        workspace_1 = torch.zeros((BLOCK_DIM, 2, chunk_size, chunk_size), device=q.device, dtype=DTYPE)
        workspace_2 = torch.zeros((BLOCK_DIM, 2, d, d), device=q.device, dtype=DTYPE)
        o = torch.zeros((b, num_heads, l, d), device=q.device, dtype=DTYPE)
        linear_attention_func(
            q,
            k,
            v,
            workspace_1,
            workspace_2,
            causal_mask,
            o,
            block_dim=BLOCK_DIM,
        )
        torch.npu.synchronize()
        return o

    q_scaled, k_scaled = _apply_gating(q, k, g, head_first=head_first)
    h_states = _build_precomputed_h(
        k_scaled,
        v,
        chunk_size,
        head_first=head_first,
        cu_seqlens=cu_seqlens,
    )
    workspace_1 = torch.zeros((BLOCK_DIM, 2, chunk_size, chunk_size), device=q.device, dtype=DTYPE)
    o = torch.zeros_like(v)
    linear_attention_func(
        q_scaled.contiguous(),
        k_scaled.contiguous(),
        v.contiguous(),
        workspace_1,
        h_states.contiguous(),
        causal_mask,
        o,
        cu_seqlens=cu_seqlens.contiguous() if cu_seqlens is not None else None,
        seq_first=not head_first,
        use_precomputed_h=True,
        batch_size_override=(len(cu_seqlens) - 1) if cu_seqlens is not None else None,
        block_dim=BLOCK_DIM,
    )
    torch.npu.synchronize()
    return o


def _make_normalized(shape: tuple[int, ...]) -> torch.Tensor:
    x = torch.randn(shape, device="npu", dtype=DTYPE)
    return x / (x.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-6)


def main():
    torch.manual_seed(0)
    torch.npu.set_device("npu:0")

    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_attention.cpp")
    test_configs = [
        {
            "label": "head_first fixed",
            "shape": (4, 2, 128, 128),
            "chunk": 64,
            "head_first": True,
            "g": None,
            "cu_seqlens": None,
        },
        {
            "label": "seq_first fixed",
            "shape": (4, 128, 2, 128),
            "chunk": 64,
            "head_first": False,
            "g": None,
            "cu_seqlens": None,
        },
        {
            "label": "seq_first gated",
            "shape": (4, 128, 2, 128),
            "chunk": 64,
            "head_first": False,
            "g": "random",
            "cu_seqlens": None,
        },
        {
            "label": "seq_first uniform-zero gated",
            "shape": (4, 128, 2, 128),
            "chunk": 64,
            "head_first": False,
            "g": "zeros",
            "cu_seqlens": None,
        },
        {
            "label": "seq_first varlen gated",
            "shape": (1, 161, 2, 128),
            "chunk": 64,
            "head_first": False,
            "g": "random",
            "cu_seqlens": [0, 17, 96, 161],
        },
    ]

    for cfg in test_configs:
        shape = cfg["shape"]
        chunk = cfg["chunk"]
        head_first = cfg["head_first"]
        cu_seqlens = cfg["cu_seqlens"]
        print(f"Testing {cfg['label']} shape={shape} C={chunk}")

        q = _make_normalized(shape)
        k = _make_normalized(shape)
        v = torch.randn(shape, device="npu", dtype=DTYPE)
        g = None
        if cfg["g"] == "random":
            g = torch.randn(shape[:-1], device="npu", dtype=torch.float32)
        elif cfg["g"] == "zeros":
            g = torch.zeros(shape[:-1], device="npu", dtype=torch.float32)
        cu_tensor = (
            torch.tensor(cu_seqlens, device="npu", dtype=torch.int32)
            if cu_seqlens is not None
            else None
        )

        o = run_kernel(
            src,
            q,
            k,
            v,
            chunk,
            g=g,
            head_first=head_first,
            cu_seqlens=cu_tensor,
        )
        ref_o = ref_linear_attention(
            q,
            k,
            v,
            g=g,
            head_first=head_first,
            cu_seqlens=cu_tensor,
        )

        total_t = shape[2] if head_first else shape[1]
        if total_t >= 4096:
            atol = 4e-2
        elif total_t >= 2048:
            atol = 2e-2
        else:
            atol = 1e-2

        torch.testing.assert_close(o.cpu(), ref_o.cpu(), rtol=RTOL, atol=atol)
        print("  passed!")

    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
