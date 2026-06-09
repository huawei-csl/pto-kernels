# Parts of this code were imported from the original implementations
# found in: https://github.com/fla-org/flash-linear-attention/
# with the corresponding Copyright notice:
#
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of the source tree:
#   https://github.com/fla-org/flash-linear-attention/

import numpy as np
import torch


def _seq_ranges(T: int, cu_seqlens=None) -> list[tuple[int, int]]:
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


class RefKDA:
    """CPU reference implementations for each KDA stage.

    KDA pipeline stages:
    gate_cumsum → kkt → inversion → wy → chunk_h_kda (snapshots + v_corr) → chunk_o_kda
    """

    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def gate_cumsum(
        self, g: torch.Tensor, chunk_size: int, cu_seqlens=None
    ) -> torch.Tensor:
        B, T, HV, Kd = g.shape
        out = torch.zeros(B, T, HV, Kd, dtype=self.dtype)
        gf = g.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, chunk_size):
                s, e = bos + j, min(bos + j + chunk_size, eos)
                out[:, s:e] = gf[:, s:e].cumsum(dim=1)
        return out

    def kkt_kda(self, k, g_cs, beta_sig, chunk_size, cu_seqlens=None):
        B, T, HV, Kd = k.shape
        L_out = torch.zeros(B, T, HV, chunk_size, dtype=self.dtype)
        kf = k.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, chunk_size):
                s, e = bos + j, min(bos + j + chunk_size, eos)
                c_len = e - s
                for h in range(HV):
                    kc = kf[0, s:e, h, :]
                    gc = g_cs[0, s:e, h, :]
                    bc = beta_sig[0, s:e, h]
                    A = kc * torch.exp(gc)
                    B_ = kc * torch.exp(-gc)
                    L_full = A @ B_.T
                    L_out[0, s:e, h, :c_len] = torch.tril(
                        L_full * bc.unsqueeze(-1), diagonal=-1
                    )
        return L_out

    def inversion_kda(self, A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        B, T, H, _ = A.shape
        out = torch.zeros(B, T, H, cs, dtype=self.dtype)
        Af = A.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                for h in range(H):
                    Ac = Af[0, s:e, h, :v]
                    M = np.linalg.inv((np.identity(v) + Ac.numpy()).astype(np.double))
                    out[0, s:e, h, :v] = torch.from_numpy(M).to(self.dtype)
        return out

    def wy_kda(self, k, v, g_cs, beta_sig, A_inv, chunk_size, cu_seqlens=None):
        B, T, HV, Kd = k.shape
        Vd = v.shape[-1]
        u_out = torch.zeros(B, T, HV, Vd, dtype=self.dtype)
        w_out = torch.zeros(B, T, HV, Kd, dtype=self.dtype)
        kf, vf = k.to(self.dtype), v.to(self.dtype)
        for bos, eos in _seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, chunk_size):
                s, e = bos + j, min(bos + j + chunk_size, eos)
                c_len = e - s
                for h in range(HV):
                    kc = kf[0, s:e, h, :]
                    gc = g_cs[0, s:e, h, :]
                    vc = vf[0, s:e, h, :]
                    bc = beta_sig[0, s:e, h]
                    A_invc = A_inv[0, s:e, h, :c_len]
                    beta_col = bc.unsqueeze(-1)
                    u_out[0, s:e, h, :] = A_invc @ (vc * beta_col)
                    w_out[0, s:e, h, :] = A_invc @ (kc * torch.exp(gc) * beta_col)
        return u_out, w_out

    def chunk_h_kda(self, k, u, w, g_cs, chunk_size, cu_seqlens=None):
        """Sequential state pass: snapshot S entering each chunk, compute v_corr.

        Returns:
            s_snapshots: [total_chunks, HV, K, V]
            v_corr:      [B, T, HV, V]
        """
        B, T, HV, Kd = k.shape
        Vd = u.shape[-1]
        ranges = _seq_ranges(T, cu_seqlens)
        n_chunks = sum(
            (eos - bos + chunk_size - 1) // chunk_size for bos, eos in ranges
        )
        s_snapshots = torch.zeros(n_chunks, HV, Kd, Vd, dtype=self.dtype)
        v_corr_out = torch.zeros(B, T, HV, Vd, dtype=self.dtype)
        ci_base = 0
        for bos, eos in ranges:
            nc = (eos - bos + chunk_size - 1) // chunk_size
            for h in range(HV):
                S = torch.zeros(Kd, Vd, dtype=self.dtype)
                for ci in range(nc):
                    s = bos + ci * chunk_size
                    e = min(s + chunk_size, eos)
                    gc = g_cs[0, s:e, h, :].to(self.dtype)
                    g_total = gc[-1]
                    kc = k[0, s:e, h, :].to(self.dtype)
                    uc = u[0, s:e, h, :].to(self.dtype)
                    wc = w[0, s:e, h, :].to(self.dtype)
                    s_snapshots[ci_base + ci, h] = S.clone()
                    v_corr = uc - wc @ S
                    v_corr_out[0, s:e, h, :] = v_corr
                    k_rest = kc * torch.exp(g_total.unsqueeze(0) - gc)
                    S = torch.exp(g_total).unsqueeze(-1) * S + k_rest.T @ v_corr
            ci_base += nc
        return s_snapshots, v_corr_out

    def chunk_o_kda(self, q, k, v_corr, s_snapshots, g_cs, chunk_size, cu_seqlens=None):
        B, T, HV, Kd = q.shape
        Vd = v_corr.shape[-1]
        o = torch.zeros(B, T, HV, Vd, dtype=self.dtype)
        ci_base = 0
        for bos, eos in _seq_ranges(T, cu_seqlens):
            nc = (eos - bos + chunk_size - 1) // chunk_size
            for h in range(HV):
                for ci in range(nc):
                    s = bos + ci * chunk_size
                    e = min(s + chunk_size, eos)
                    gc = g_cs[0, s:e, h, :].to(self.dtype)
                    qc = q[0, s:e, h, :].to(self.dtype)
                    kc = k[0, s:e, h, :].to(self.dtype)
                    vc = v_corr[0, s:e, h, :].to(self.dtype)
                    S = s_snapshots[ci_base + ci, h].to(self.dtype)
                    q_eff = qc * torch.exp(gc)
                    k_eff = kc * torch.exp(-gc)
                    inter = q_eff @ S
                    Aqk = torch.tril(q_eff @ k_eff.T, diagonal=0)
                    o[0, s:e, h, :] = inter + Aqk @ vc
            ci_base += nc
        return o
