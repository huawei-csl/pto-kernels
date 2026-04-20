"""
Educational PyTorch emulation of ``triton_baseline/fla_vendor`` GDN kernels.

API mirrors the Triton entry points (same argument lists and tensor layouts).
"""

from ._common import relative_rmse, tensor_r2_score
from .chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_fwd_h_explained
from .chunk_o import chunk_fwd_o, chunk_fwd_o_explained
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .solve_tril import solve_tril
from .wy_fast import recompute_w_u_fwd

__all__ = [
    "tensor_r2_score",
    "relative_rmse",
    "chunk_local_cumsum",
    "chunk_scaled_dot_kkt_fwd",
    "recompute_w_u_fwd",
    "solve_tril",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_gated_delta_rule_fwd_h_explained",
    "chunk_fwd_o",
    "chunk_fwd_o_explained",
]
