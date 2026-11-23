import math
from typing import Literal

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

# adapted from https://github.com/vllm-project/vllm-ascend/blob/v0.18.0rc1/vllm_ascend/ops/triton/fla/chunk_o.py


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(lens, chunk_size)]).cumsum(-1)


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["chunk_offsets", "scale", "T", "H", "Hg", "K", "V"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = T

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V

    for i_t in range(NT):
        i_tg = boh + i_t
        h_base = h + (i_tg * H + i_h).to(tl.int64) * K * V
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))

            b_o += tl.dot(b_q, b_h)
            b_A += tl.dot(b_q, b_k)

        if USE_G:
            offs_t = i_t * BT + tl.arange(0, BT)
            mask_t = offs_t < T
            g_ptr = g + bos + i_h * T_max
            b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

            b_o = b_o * tl.exp(b_g)[:, None]
            b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])

        o_i = tl.arange(0, BT).to(tl.float32)
        m_A = o_i[:, None] >= o_i[None, :]
        b_A = tl.where(m_A, b_A, 0)

        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    b, t, hg, k_dim, v_dim = *q.shape, v.shape[-1]
    h_dim = v.shape[-2]
    bt = chunk_size

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    if cu_seqlens is None:
        n, chunk_offsets = b, None
    else:
        n, chunk_offsets = len(cu_seqlens) - 1, prepare_chunk_offsets(cu_seqlens, bt)
    bk = min(64, triton.next_power_of_2(k_dim))
    bv = min(64, triton.next_power_of_2(v_dim))

    def grid(meta):
        return (triton.cdiv(v_dim, meta["BV"]), n * h_dim)

    if g is not None:
        g = g.transpose(1, 2).contiguous()

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=t,
        H=h_dim,
        Hg=hg,
        K=k_dim,
        V=v_dim,
        BT=bt,
        BK=bk,
        BV=bv,
        num_warps=4,
        num_stages=2,
    )
    return o


def build_chunk_states(
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    b, h, t, d_k = k.shape
    d_v = v.shape[-1]
    nt = math.ceil(t / chunk_size)
    state = torch.zeros((b, h, d_k, d_v), device=k.device, dtype=torch.float32)
    states = []
    for i_t in range(nt):
        states.append(state.to(v.dtype))
        start = i_t * chunk_size
        end = min(start + chunk_size, t)
        state = state + torch.matmul(
            k[:, :, start:end, :].float().transpose(-1, -2),
            v[:, :, start:end, :].float(),
        )
    return torch.stack(states, dim=1).contiguous().view(b * nt, h, d_k, d_v)


def prepare_vllm_equivalent_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    g_mode: Literal["none", "uniform_zero"],
    varlen_mode: Literal["static", "varlen_equiv"],
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must all be rank-4 tensors")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on (B, H, T)")

    b, h, t, d = q.shape
    q_seq = q.transpose(1, 2).contiguous()
    k_seq = k.transpose(1, 2).contiguous()
    v_seq = v.transpose(1, 2).contiguous()
    h_states = build_chunk_states(k, v, chunk_size)

    g = None
    if g_mode == "uniform_zero":
        g = torch.zeros((b, t, h), device=q.device, dtype=torch.float32)
    elif g_mode != "none":
        raise ValueError(f"Unsupported g_mode: {g_mode}")

    if varlen_mode == "static":
        return {
            "q": q_seq,
            "k": k_seq,
            "v": v_seq,
            "h": h_states,
            "g": g,
            "cu_seqlens": None,
            "restore": lambda o: o.transpose(1, 2).contiguous(),
        }

    if varlen_mode != "varlen_equiv":
        raise ValueError(f"Unsupported varlen_mode: {varlen_mode}")

    total_t = b * t
    cu_seqlens = torch.arange(0, total_t + 1, t, device=q.device, dtype=torch.long)
    q_flat = q_seq.reshape(1, total_t, h, d).contiguous()
    k_flat = k_seq.reshape(1, total_t, h, d).contiguous()
    v_flat = v_seq.reshape(1, total_t, h, v.shape[-1]).contiguous()
    if g is not None:
        g = g.reshape(1, total_t, h).contiguous()

    return {
        "q": q_flat,
        "k": k_flat,
        "v": v_flat,
        "h": h_states,
        "g": g,
        "cu_seqlens": cu_seqlens,
        "restore": lambda o: o.view(b, t, h, v.shape[-1]).transpose(1, 2).contiguous(),
    }


def chunk_o_vllm_adapted(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    scale: float = 1.0,
    g_mode: Literal["none", "uniform_zero"] = "none",
    varlen_mode: Literal["static", "varlen_equiv"] = "static",
) -> torch.Tensor:
    prepared = prepare_vllm_equivalent_inputs(
        q, k, v, chunk_size, g_mode=g_mode, varlen_mode=varlen_mode
    )
    out = chunk_fwd_o(
        q=prepared["q"],
        k=prepared["k"],
        v=prepared["v"],
        h=prepared["h"],
        g=prepared["g"],
        scale=scale,
        cu_seqlens=prepared["cu_seqlens"],
        chunk_size=chunk_size,
    )
    return prepared["restore"](out)
