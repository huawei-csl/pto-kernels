"""
Compile (bisheng) and run the static varlen chunk_gated_delta_rule PTO kernels,
then compare to a pure PyTorch reference (no TileLang).

The dumped ``*_H32.cpp`` / ``*_H48.cpp`` kernels bake in ``T_total_pad``,
``NT_max``, and ``N * H`` launch geometry. Constants below match the copies in
this directory (generated from ``tilelang_codegen/kernels``).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys

import torch
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import pto_static_common  # noqa: F401 — env validation

from static_kernel_libs import lib_chunk_gated_delta_rule_varlen_h32, lib_chunk_gated_delta_rule_varlen_h48

torch_npu = torch.npu  # noqa: F401 — register NPU

BT = 64

# Baked into the dumped AICore code (strides / bounds / launch grid).
KERNEL_META = {
    "H48": {
        "lib_fn": lib_chunk_gated_delta_rule_varlen_h48,
        "H": 48,
        "Hg": 16,
        "N": 5,
        "T_pad": 568,
        "NT_max": 4,
        "default_seqlens": (7, 32, 159, 256, 50),
    },
    "H32": {
        "lib_fn": lib_chunk_gated_delta_rule_varlen_h32,
        "H": 32,
        "Hg": 16,
        "N": 2,
        "T_pad": 1056,
        "NT_max": 16,
        # Strides in H32 dump match ``T_total = 992`` (not 1024); use this for exact GM layout.
        "default_seqlens": (496, 496),
        "alt_seqlens_512": (512, 512),
    },
}


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_offsets = []
    offset = 0
    cu_seqlens_np = cu_seqlens.cpu().numpy()
    for i in range(len(cu_seqlens_np) - 1):
        t_len = int(cu_seqlens_np[i + 1] - cu_seqlens_np[i])
        nt = (t_len + chunk_size - 1) // chunk_size
        chunk_offsets.append(offset)
        offset += nt
    return torch.tensor(chunk_offsets, dtype=torch.int32, device=cu_seqlens.device)


def ref_chunk_gated_delta_rule_varlen(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Varlen-only reference (same math as ``chunk_gated_delta_rule_varlen.ref_chunk_gated_delta_rule``)."""
    kf = k.float()
    wf = w.float()
    uf = u.float()
    gf = g.float() if g is not None else None
    init_f = initial_state.float() if initial_state is not None else None

    _, t_total, hg, kk = k.shape
    _, _, h, v = u.shape
    n = len(cu_seqlens) - 1

    nt_total = sum(
        (int(cu_seqlens[i + 1].item()) - int(cu_seqlens[i].item()) + chunk_size - 1) // chunk_size
        for i in range(n)
    )

    h_out = torch.zeros(1, nt_total, h, kk, v, dtype=torch.float32, device=k.device)
    v_new = torch.zeros(1, t_total, h, v, dtype=torch.float32, device=k.device)
    final_state = (
        torch.zeros(1, n, h, kk, v, dtype=torch.float32, device=k.device) if output_final_state else None
    )

    chunk_offset = 0
    for i_n in range(n):
        bos, eos = int(cu_seqlens[i_n].item()), int(cu_seqlens[i_n + 1].item())
        t_len = eos - bos
        nt = (t_len + chunk_size - 1) // chunk_size

        for i_h in range(h):
            h_state = (
                init_f[0, i_n, i_h].clone()
                if init_f is not None
                else torch.zeros(kk, v, dtype=torch.float32, device=k.device)
            )
            k_head = i_h // (h // hg)

            for i_t in range(nt):
                t_start = i_t * chunk_size
                t_end = min((i_t + 1) * chunk_size, t_len)

                h_out[0, chunk_offset + i_t, i_h] = h_state
                k_chunk = kf[0, bos + t_start : bos + t_end, k_head, :]
                w_chunk = wf[0, bos + t_start : bos + t_end, i_h, :]
                v_chunk = uf[0, bos + t_start : bos + t_end, i_h, :]

                v_n = v_chunk - torch.matmul(w_chunk, h_state)
                v_new[0, bos + t_start : bos + t_end, i_h, :] = v_n

                if gf is not None:
                    g_chunk = gf[0, bos + t_start : bos + t_end, i_h]
                    g_last = g_chunk[-1].item()
                    v_n = v_n * torch.exp(g_last - g_chunk)[:, None]
                    h_state = h_state * torch.exp(torch.tensor(g_last, device=k.device, dtype=torch.float32))

                h_state = h_state + torch.matmul(k_chunk.transpose(-1, -2), v_n)

            if output_final_state and final_state is not None:
                final_state[0, i_n, i_h] = h_state
        chunk_offset += nt

    return h_out.half(), v_new.half(), final_state.half() if final_state is not None else None


def pack_h_ret(
    h_work: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int,
    nt_max: int,
    h_: int,
    kk: int,
    v: int,
) -> torch.Tensor:
    """Match ``chunk_gated_delta_rule_fwd_h`` varlen packing: ``(1, NT_total, H, K, V)``."""
    n = len(cu_seqlens) - 1
    nt_total = int(
        sum(
            (int(cu_seqlens[i + 1].item()) - int(cu_seqlens[i].item()) + chunk_size - 1) // chunk_size
            for i in range(n)
        )
    )
    h_ret = torch.zeros(1, nt_total, h_, kk, v, dtype=torch.float16, device=h_work.device)
    cu_np = cu_seqlens.cpu().numpy()
    for i in range(n):
        nt_i = (int(cu_np[i + 1]) - int(cu_np[i]) + chunk_size - 1) // chunk_size
        offset = int(chunk_offsets[i].item())
        h_ret[0, offset : offset + nt_i] = h_work[i, :nt_i]
    return h_ret


def run_varlen_kernel(
    lib,
    h_out: torch.Tensor,
    k_pad: torch.Tensor,
    u_pad: torch.Tensor,
    w_pad: torch.Tensor,
    g_pad: torch.Tensor,
    v_new_pad: torch.Tensor,
    h0: torch.Tensor,
    ht: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ws_wh: torch.Tensor,
    ws_vnew: torch.Tensor,
    ws_hupd: torch.Tensor,
    ws_h: torch.Tensor,
    stream,
):
    lib.call(
        ctypes.c_void_p(h_out.data_ptr()),
        ctypes.c_void_p(k_pad.data_ptr()),
        ctypes.c_void_p(u_pad.data_ptr()),
        ctypes.c_void_p(w_pad.data_ptr()),
        ctypes.c_void_p(g_pad.data_ptr()),
        ctypes.c_void_p(v_new_pad.data_ptr()),
        ctypes.c_void_p(h0.data_ptr()),
        ctypes.c_void_p(ht.data_ptr()),
        ctypes.c_void_p(cu_seqlens.data_ptr()),
        ctypes.c_void_p(ws_wh.data_ptr()),
        ctypes.c_void_p(ws_vnew.data_ptr()),
        ctypes.c_void_p(ws_hupd.data_ptr()),
        ctypes.c_void_p(ws_h.data_ptr()),
        stream,
    )


def main():
    parser = argparse.ArgumentParser(description="Static PTO varlen chunk_gated_delta_rule vs PyTorch ref")
    parser.add_argument(
        "--profile",
        choices=("H32", "H48"),
        default="H48",
        help="Which dumped kernel (must match head count / launch geometry).",
    )
    parser.add_argument(
        "--seqlens",
        type=str,
        default=None,
        help="Comma-separated sequence lengths (default: profile-specific layout-safe tuple).",
    )
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    meta = KERNEL_META[args.profile]
    h, hg = meta["H"], meta["Hg"]
    n_expect = meta["N"]
    t_pad = meta["T_pad"]
    nt_max = meta["NT_max"]

    if args.seqlens is not None:
        seqlens = tuple(int(x.strip()) for x in args.seqlens.split(",") if x.strip())
    else:
        seqlens = meta["default_seqlens"]

    if len(seqlens) != n_expect:
        raise ValueError(f"Profile {args.profile} expects N={n_expect} sequences, got {len(seqlens)}.")

    t_total = sum(seqlens)
    if t_total + BT != t_pad:
        print(
            f"WARNING: sum(seqlens)+BT = {t_total + BT} != baked T_pad={t_pad}; "
            "GM strides in the dump may not match (e.g. use default seqlens for H32).",
            file=sys.stderr,
        )

    torch.manual_seed(args.seed)
    torch.npu.set_device("npu:0")
    stream = torch.npu.current_stream()._as_parameter_

    cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), dim=0)), dtype=torch.int32, device="npu")
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    kk, v = 128, 128
    k = torch.randn(1, t_total, hg, kk, device="npu", dtype=torch.float16) * 0.01
    w = torch.randn(1, t_total, h, kk, device="npu", dtype=torch.float16) * 0.01
    u = torch.randn(1, t_total, h, v, device="npu", dtype=torch.float16) * 0.01
    g = torch.randn(1, t_total, h, device="npu", dtype=torch.float32) * 0.01
    initial_state = torch.randn(1, n_expect, h, kk, v, device="npu", dtype=torch.float16) * 0.01

    def pad_tensor(t: torch.Tensor) -> torch.Tensor:
        # ``t`` is ``[1, T, ...]`` (batch 1); pad the time axis like ``torch.cat`` on dim 0 of flattened ``[T, ...]``.
        z = torch.zeros((t.shape[0], BT) + t.shape[2:], dtype=t.dtype, device=t.device)
        return torch.cat([t, z], dim=1)

    k_pad = pad_tensor(k)
    w_pad = pad_tensor(w)
    u_pad = pad_tensor(u)
    g_pad = pad_tensor(g.float()).contiguous()
    v_new_pad = torch.empty(1, t_pad, h, v, device="npu", dtype=torch.float16)
    v_new_pad.zero_()

    h_work = torch.zeros(n_expect, nt_max, h, kk, v, device="npu", dtype=torch.float16)
    h0 = torch.zeros(n_expect, h, kk, v, device="npu", dtype=torch.float16)
    h0.copy_(initial_state.squeeze(0))
    ht = torch.zeros(n_expect, h, kk, v, device="npu", dtype=torch.float16)

    ws_wh = torch.zeros(n_expect, h, BT, v, device="npu", dtype=torch.float32)
    ws_vnew = torch.zeros(n_expect, h, BT, v, device="npu", dtype=torch.float16)
    ws_hupd = torch.zeros(n_expect, h, kk, v, device="npu", dtype=torch.float16)
    ws_h = torch.zeros(n_expect, h, kk, v, device="npu", dtype=torch.float16)

    lib = meta["lib_fn"]()
    run_varlen_kernel(
        lib,
        h_work,
        k_pad.squeeze(0),
        u_pad.squeeze(0),
        w_pad.squeeze(0),
        g_pad.squeeze(0),
        v_new_pad.squeeze(0),
        h0,
        ht,
        cu_seqlens,
        ws_wh,
        ws_vnew,
        ws_hupd,
        ws_h,
        stream,
    )
    torch.npu.synchronize()

    v_new_out = v_new_pad[:, :t_total].contiguous()
    h_packed = pack_h_ret(h_work, cu_seqlens, chunk_offsets, BT, nt_max, h, kk, v)

    ref_h, ref_v_new, ref_ht = ref_chunk_gated_delta_rule_varlen(
        k.cpu(),
        w.cpu(),
        u.cpu(),
        g.cpu(),
        initial_state.cpu(),
        True,
        cu_seqlens.cpu(),
    )

    torch.testing.assert_close(h_packed.cpu(), ref_h.cpu(), rtol=args.rtol, atol=args.atol)
    torch.testing.assert_close(v_new_out.cpu(), ref_v_new.cpu(), rtol=args.rtol, atol=args.atol)
    torch.testing.assert_close(ht.cpu(), ref_ht.squeeze(0).cpu(), rtol=args.rtol, atol=args.atol)
    print(f"chunk_gated_delta_rule varlen static ({args.profile}) matches PyTorch reference.")


if __name__ == "__main__":
    main()
