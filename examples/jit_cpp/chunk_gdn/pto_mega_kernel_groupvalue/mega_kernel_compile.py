"""mega_kernel_compile.py — compile, load, and run the group-value GDN mega-kernel."""
from __future__ import annotations

import ctypes
import os
import subprocess
from functools import lru_cache

import torch

ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get(
    "ASCEND_HOME_PATH", ""
)
if not ASCEND_TOOLKIT_HOME:
    raise RuntimeError("Set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH")

PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)
_pto_inc = os.path.join(PTO_LIB_PATH, "include")
if not os.path.isdir(_pto_inc):
    raise RuntimeError(f"PTO include directory missing: {_pto_inc!r}")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
_CSRC_KERNEL = os.path.join(_REPO_ROOT, "csrc", "kernel")
_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"

_npu_dev = os.environ.get("GDN_NPU_DEVICE", "npu:0")
try:
    BLOCK_DIM = int(
        getattr(torch.npu.get_device_properties(_npu_dev), "cube_core_num", 20)
    )
except RuntimeError:
    BLOCK_DIM = 24

COMPILED_DIR = os.path.join(_HERE, "compiled_lib")


def _vp(t: torch.Tensor | None) -> ctypes.c_void_p:
    if t is None:
        return ctypes.c_void_p()
    return ctypes.c_void_p(t.data_ptr())


@lru_cache(maxsize=None)
def compile_mega_kernel(
    *,
    num_heads: int = 16,
    key_heads: int | None = None,
    hidden_size: int = 128,
    chunk_size: int = 128,
    cpp_mtime_ns: int = 0,
) -> str:
    hg = key_heads if key_heads is not None else num_heads
    os.makedirs(COMPILED_DIR, exist_ok=True)
    cpp_path = os.path.join(_HERE, "mega_kernel.cpp")
    stem = (
        f"mega_kernel_groupvalue_H{num_heads}_Hg{hg}"
        f"_D{hidden_size}_C{chunk_size}"
    )
    lib_path = os.path.join(COMPILED_DIR, f"{stem}.so")

    extra = os.environ.get("PTO_DYNAMIC_EXTRA_FLAGS", "").split()
    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        f"-I{_pto_inc}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
        f"-I{_CSRC_KERNEL}",
        f"-DGDN_H={num_heads}",
        f"-DGDN_HG={hg}",
        f"-DGDN_D={hidden_size}",
        f"-DGDN_C={chunk_size}",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")
    flags.extend(extra)

    cmd = ["bisheng", *flags, cpp_path, "-o", lib_path]
    if os.environ.get("VERBOSE_COMPILE"):
        print("compile:", " ".join(cmd))
    print(f"[mega_kernel_groupvalue] Compiling {cpp_path} ...")
    subprocess.run(cmd, check=True, timeout=600)
    print(f"[mega_kernel_groupvalue] Compiled → {lib_path}")
    return lib_path


@lru_cache(maxsize=None)
def load_mega_kernel(
    *,
    num_heads: int = 16,
    key_heads: int | None = None,
    hidden_size: int = 128,
    chunk_size: int = 128,
):
    mtime = os.stat(os.path.join(_HERE, "mega_kernel.cpp")).st_mtime_ns
    lib_path = compile_mega_kernel(
        num_heads=num_heads,
        key_heads=key_heads,
        hidden_size=hidden_size,
        chunk_size=chunk_size,
        cpp_mtime_ns=mtime,
    )
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
    ] + [ctypes.c_void_p] * 28 + [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_uint32,
    ]
    lib.call_kernel.restype = None
    return lib


def _count_varlen_chunks(cu_seqlens: torch.Tensor, chunk_size: int) -> int:
    return sum(
        (int(eos) - int(bos) + chunk_size - 1) // chunk_size
        for bos, eos in zip(
            cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False
        )
    )


def total_chunks(batch_size, seq_len, chunk_size, cu_seqlens=None):
    if cu_seqlens is None:
        return batch_size * ((seq_len + chunk_size - 1) // chunk_size)
    return _count_varlen_chunks(cu_seqlens, chunk_size)


def run_mega_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_in: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    stream,
    chunk_size: int = 128,
    scale: float = 1.0,
    block_dim: int | None = None,
    key_heads: int | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the group-value mega-kernel.

    ``q``, ``k``: ``[B, T, Hg, D]``; ``v``, ``β``, ``g``: ``[B, T, H]`` with
    ``H % Hg == 0``. Returns ``O * scale`` (and optionally final state like the
    per-stage pipeline).
    """
    dev = q.device
    hg = q.shape[2]
    kh = key_heads if key_heads is not None else hg
    H = v.shape[2]
    D = q.shape[3]
    C = chunk_size
    assert k.shape[2] == hg == kh, "q/k head dim must match key_heads"
    assert H % kh == 0, f"H={H} must be divisible by Hg={kh}"
    assert v.shape[3] == D and beta.shape[2] == H and g_in.shape[2] == H
    T = q.shape[1]
    N_seq = len(cu_seqlens) - 1
    bd = block_dim or BLOCK_DIM

    if cu_seqlens.dtype != torch.int32:
        cu_seqlens = cu_seqlens.to(torch.int32)

    msk_lower = torch.tril(
        torch.ones(C, C, device=dev), diagonal=-1
    ).float()
    msk_full = torch.tril(
        torch.ones(C, C, device=dev), diagonal=0
    ).float()
    minus_identity = torch.zeros(C, C, device=dev, dtype=torch.float16)
    minus_identity.fill_diagonal_(-1)

    g_sum = torch.empty(1, T, H, device=dev, dtype=torch.float32)
    g_t = torch.empty(H, T, device=dev, dtype=torch.float32)
    beta_t = torch.empty(H, T, device=dev, dtype=torch.float16)
    A = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    tc = total_chunks(N_seq, T, C, cu_seqlens)
    num_matrices = tc * H
    A_inv_f32 = torch.zeros(1, T, H, C, device=dev, dtype=torch.float32)
    A_inv = torch.zeros(1, T, H, C, device=dev, dtype=torch.float16)
    w = torch.empty_like(v)
    u = torch.empty_like(v)
    s = torch.zeros(tc * H, D, D, device=dev, dtype=torch.float16)
    v_new = torch.empty_like(v)
    fs = torch.zeros(N_seq * H, D, D, device=dev, dtype=torch.float16)

    kkt_ws = torch.zeros(bd * 2, C, C, device=dev, dtype=torch.float16)
    wy_ws_a1 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    wy_ws_a2 = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    h_ws = torch.zeros(bd * 4, D, D, device=dev, dtype=torch.float16)
    o_ws_qk = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)
    o_ws_qs = torch.zeros(bd, C, D, device=dev, dtype=torch.float16)
    o_ws_gated = torch.zeros(bd, C, C, device=dev, dtype=torch.float16)

    o_out = torch.empty_like(v)

    lib = load_mega_kernel(
        num_heads=H,
        key_heads=kh,
        hidden_size=D,
        chunk_size=C,
    )
    lib.call_kernel(
        bd, stream,
        _vp(q), _vp(k), _vp(v), _vp(g_in), _vp(beta),
        _vp(msk_lower), _vp(msk_full), _vp(minus_identity), _vp(cu_seqlens),
        _vp(o_out),
        _vp(g_sum), _vp(g_t), _vp(beta_t),
        _vp(A), _vp(A_inv_f32), _vp(A_inv),
        _vp(w), _vp(u), _vp(s), _vp(v_new), _vp(fs),
        _vp(kkt_ws), _vp(wy_ws_a1), _vp(wy_ws_a2), _vp(h_ws),
        _vp(o_ws_qk), _vp(o_ws_qs), _vp(o_ws_gated),
        N_seq, T, T, num_matrices,
    )

    o_scaled = o_out * scale
    if return_final_state:
        return o_scaled, fs.view(N_seq, H, D, D)
    return o_scaled
