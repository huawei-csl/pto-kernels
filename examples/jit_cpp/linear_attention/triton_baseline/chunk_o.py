import math
from typing import Optional

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.heuristics(
    {
        "HAS_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "STORE_FINAL_STATE": lambda args: args["final_state"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "total_bh", "scale"])
def _chunk_o_fwd_kernel(
    q,
    k,
    v,
    initial_state,
    o,
    final_state,
    scale,
    T,
    total_bh,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
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

    state = tl.zeros([K, BV], dtype=tl.float16)
    if HAS_INITIAL_STATE:
        p_init = tl.make_block_ptr(
            initial_state + i_bh * K * V,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (K, BV),
            (1, 0),
        )
        state = tl.load(p_init, boundary_check=(0, 1)).to(tl.float16)

    for i_c in range(tl.cdiv(T, C)):
        chunk_start = i_c * C
        p_k = tl.make_block_ptr(
            k, (K, T), (1, K), (0, chunk_start), (K, C), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v, (T, V), (V, 1), (chunk_start, i_v * BV), (C, BV), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        for i_t in range(tl.cdiv(C, BT)):
            row_start = chunk_start + i_t * BT
            p_q = tl.make_block_ptr(
                q, (T, K), (K, 1), (row_start, 0), (BT, K), (1, 0)
            )
            p_o = tl.make_block_ptr(
                o, (T, V), (V, 1), (row_start, i_v * BV), (BT, BV), (1, 0)
            )

            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_a = tl.dot(b_q, b_k)
            row_offsets = (i_t * BT + tl.arange(0, BT)).to(tl.float32)
            col_offsets = tl.arange(0, C).to(tl.float32)
            b_a = tl.where(row_offsets[:, None] >= col_offsets[None, :], b_a, 0)

            b_o = tl.dot(b_q, state)
            b_o += tl.dot(b_a.to(b_v.dtype), b_v)
            b_o *= scale

            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        state += tl.dot(b_k, b_v).to(state.dtype)

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(
            final_state + i_bh * K * V,
            (K, V),
            (V, 1),
            (0, i_v * BV),
            (K, BV),
            (1, 0),
        )
        tl.store(p_final, state.to(p_final.dtype.element_ty), boundary_check=(0, 1))


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


def chunk_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    *,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
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

    if initial_state is not None:
        expected = (b, h, d_k, d_v)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                f"initial_state must have shape {expected}, got {tuple(initial_state.shape)}"
            )
        initial_state = initial_state.contiguous()

    out = torch.empty_like(v)
    final_state = None
    if output_final_state:
        state_dtype = (
            initial_state.dtype if initial_state is not None else torch.float32
        )
        final_state = torch.empty(
            (b, h, d_k, d_v), device=q.device, dtype=state_dtype
        )

    tile_rows = min(64, chunk_size)
    bv = min(16, triton.next_power_of_2(d_v))
    grid = (total_bh * triton.cdiv(d_v, bv),)
    _chunk_o_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        initial_state=initial_state,
        o=out,
        final_state=final_state,
        scale=scale,
        T=t,
        total_bh=total_bh,
        K=d_k,
        V=d_v,
        C=chunk_size,
        BT=tile_rows,
        BV=bv,
        num_warps=4,
        num_stages=2,
    )
    return (out, final_state) if output_final_state else out


chunk_linear_attention = chunk_o
