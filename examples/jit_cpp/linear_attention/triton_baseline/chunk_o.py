import math
from functools import lru_cache
from typing import Optional

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "NT", "total_bh", "scale"])
def _chunk_o_fwd_kernel(
    q,
    k,
    v,
    h,
    mask,
    o,
    scale,
    T,
    NT,
    total_bh,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_PRECOMPUTED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    NV: tl.constexpr = tl.cdiv(V, BV)
    i_bh = pid // NV
    i_v = pid % NV

    if i_bh >= total_bh:
        return

    q += i_bh * T * K
    k += i_bh * T * K
    v += i_bh * T * V
    o += i_bh * T * V

    for i_c in range(NT):
        chunk_start = i_c * C
        h_base = h + ((i_bh * NT + i_c).to(tl.int64) * K * V)
        p_v = tl.make_block_ptr(
            v, (T, V), (V, 1), (chunk_start, i_v * BV), (C, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))

        for i_t in range(tl.cdiv(C, BT)):
            row_start = chunk_start + i_t * BT
            p_o = tl.make_block_ptr(
                o, (T, V), (V, 1), (row_start, i_v * BV), (BT, BV), (1, 0)
            )
            p_mask = tl.make_block_ptr(
                mask, (C, C), (C, 1), (i_t * BT, 0), (BT, C), (1, 0)
            )
            b_o = tl.zeros([BT, BV], dtype=tl.float32)
            b_a = tl.zeros([BT, C], dtype=tl.float32)

            for i_k in range(tl.cdiv(K, BK)):
                p_q = tl.make_block_ptr(
                    q, (T, K), (K, 1), (row_start, i_k * BK), (BT, BK), (1, 0)
                )
                p_k = tl.make_block_ptr(
                    k, (K, T), (1, K), (i_k * BK, chunk_start), (BK, C), (0, 1)
                )
                p_h = tl.make_block_ptr(
                    h_base, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
                )
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_o += tl.dot(b_q, b_h)
                b_a += tl.dot(b_q, b_k)

            if USE_PRECOMPUTED_MASK:
                b_mask = tl.load(p_mask, boundary_check=(0, 1))
                b_a *= b_mask.to(b_a.dtype)
            else:
                row_offsets = (i_t * BT + tl.arange(0, BT)).to(tl.float32)
                col_offsets = tl.arange(0, C).to(tl.float32)
                b_a = tl.where(row_offsets[:, None] >= col_offsets[None, :], b_a, 0)
            b_o += tl.dot(b_a.to(b_v.dtype), b_v)
            b_o *= scale

            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _require_head_first(x: torch.Tensor, name: str) -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} must be rank-4, got {tuple(x.shape)}")


def ref_chunk_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    _require_head_first(q, "q")
    _require_head_first(k, "k")
    _require_head_first(v, "v")

    b, h, t, d_k = q.shape
    d_v = v.shape[-1]
    qf = q.float()
    kf = k.float()
    vf = v.float()

    state = torch.zeros((b, h, d_k, d_v), device=q.device, dtype=torch.float32)
    if initial_state is not None:
        state.copy_(initial_state.float())

    out = torch.zeros((b, h, t, d_v), device=q.device, dtype=torch.float32)
    nt = math.ceil(t / chunk_size)
    for i_t in range(nt):
        start = i_t * chunk_size
        end = min(start + chunk_size, t)
        q_tile = qf[:, :, start:end, :]
        k_tile = kf[:, :, start:end, :]
        v_tile = vf[:, :, start:end, :]
        attn = torch.matmul(q_tile, k_tile.transpose(-1, -2)).tril()
        out[:, :, start:end, :] = (
            torch.matmul(q_tile, state) + torch.matmul(attn, v_tile)
        ) * scale
        state = state + torch.matmul(k_tile.transpose(-1, -2), v_tile)

    if output_final_state:
        return out.to(v.dtype), state
    return out.to(v.dtype)


def build_chunk_states(
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    _require_head_first(k, "k")
    _require_head_first(v, "v")

    b, h, t, d_k = k.shape
    d_v = v.shape[-1]
    nt = math.ceil(t / chunk_size)
    state = torch.zeros((b, h, d_k, d_v), device=k.device, dtype=torch.float32)
    if initial_state is not None:
        state.copy_(initial_state.float())

    states = []
    for i_t in range(nt):
        states.append(state.to(v.dtype))
        start = i_t * chunk_size
        end = min(start + chunk_size, t)
        state = state + torch.matmul(
            k[:, :, start:end, :].float().transpose(-1, -2),
            v[:, :, start:end, :].float(),
        )

    stacked = torch.stack(states, dim=2).contiguous()
    if output_final_state:
        return stacked, state
    return stacked


@lru_cache(maxsize=None)
def get_causal_mask(chunk_size: int, dtype: torch.dtype, device_index: int) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    mask = torch.ones(
        (chunk_size, chunk_size),
        device=f"npu:{device_index}",
        dtype=dtype,
    )
    return torch.tril(mask).contiguous()


def _normalize_precomputed_h(
    h: torch.Tensor,
    b: int,
    heads: int,
    nt: int,
    d_k: int,
    d_v: int,
) -> torch.Tensor:
    expected_5d = (b, heads, nt, d_k, d_v)
    expected_4d = (b * nt, heads, d_k, d_v)
    if tuple(h.shape) == expected_5d:
        return h.contiguous().view(b * nt, heads, d_k, d_v)
    if tuple(h.shape) == expected_4d:
        return h.contiguous()
    raise ValueError(
        f"precomputed_h must have shape {expected_5d} or {expected_4d}, got {tuple(h.shape)}"
    )


def chunk_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    precomputed_h: Optional[torch.Tensor] = None,
    precomputed_mask: Optional[torch.Tensor] = None,
    use_cached_mask: bool = True,
):
    # TODO: support seq_first layout: (B, T, H, D).
    # TODO: support gated and varlen variants when the baseline is proven out.
    _require_head_first(q, "q")
    _require_head_first(k, "k")
    _require_head_first(v, "v")

    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} vs {k.shape}")
    if q.shape[:3] != v.shape[:3]:
        raise ValueError(
            f"q/k and v must agree on (B, H, T), got {q.shape[:3]} vs {v.shape[:3]}"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if q.device.type != "npu":
        raise ValueError(f"expected NPU tensors, got {q.device}")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    b, h, t, d_k = q.shape
    d_v = v.shape[-1]
    total_bh = b * h
    nt = math.ceil(t / chunk_size)

    if initial_state is not None:
        expected = (b, h, d_k, d_v)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"initial_state must have shape {expected}, got {tuple(initial_state.shape)}"
            )
        initial_state = initial_state.contiguous()

    if precomputed_h is None:
        built = build_chunk_states(
            k,
            v,
            chunk_size,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        if output_final_state:
            h_states, final_state = built
        else:
            h_states, final_state = built, None
        precomputed_h = h_states
    else:
        if output_final_state:
            raise ValueError(
                "output_final_state=True is not supported together with externally supplied precomputed_h"
            )
        final_state = None

    precomputed_h = _normalize_precomputed_h(precomputed_h, b, h, nt, d_k, d_v)
    if precomputed_mask is not None:
        if not use_cached_mask:
            raise ValueError("precomputed_mask requires use_cached_mask=True")
        expected_mask = (chunk_size, chunk_size)
        if tuple(precomputed_mask.shape) != expected_mask:
            raise ValueError(
                f"precomputed_mask must have shape {expected_mask}, got {tuple(precomputed_mask.shape)}"
            )
        if precomputed_mask.device != q.device:
            raise ValueError(
                f"precomputed_mask must be on {q.device}, got {precomputed_mask.device}"
            )
        precomputed_mask = precomputed_mask.contiguous()
    elif use_cached_mask:
        precomputed_mask = get_causal_mask(chunk_size, q.dtype, q.device.index or 0)
    else:
        precomputed_mask = torch.empty((chunk_size, chunk_size), device=q.device, dtype=q.dtype)
    out = torch.empty_like(v)
    tile_rows = min(64, chunk_size)
    bk = min(64, triton.next_power_of_2(d_k))
    bv = min(64, triton.next_power_of_2(d_v))
    grid = (total_bh * triton.cdiv(d_v, bv),)
    _chunk_o_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        h=precomputed_h,
        mask=precomputed_mask,
        o=out,
        scale=scale,
        T=t,
        NT=nt,
        total_bh=total_bh,
        K=d_k,
        V=d_v,
        C=chunk_size,
        BT=tile_rows,
        BK=bk,
        BV=bv,
        USE_PRECOMPUTED_MASK=use_cached_mask,
        num_warps=4,
        num_stages=2,
    )
    return (out, final_state) if output_final_state else out


chunk_linear_attention = chunk_o
