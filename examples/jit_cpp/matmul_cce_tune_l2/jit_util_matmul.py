import ctypes
import os
import subprocess

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)

DEFAULT_MAX_BLOCK_DIM = int(os.environ.get("PTO_MATMUL_MAX_BLOCK_DIM", "20"))

M_TILE = 128
N_TILE = 256
K_TILE = 512


def compile_cpp(kernel_cpp: str, verbose: bool = False, timeout: int = 120) -> str:
    so_dir = os.path.join(os.path.dirname(kernel_cpp), "outputs", "so")
    os.makedirs(so_dir, exist_ok=True)
    lib_path = os.path.join(so_dir, "matmul_abt_jit.so")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "--cce-soc-version=Ascend910B4",
        "--cce-soc-core-type=CubeCore",
        f"-I{PTO_LIB_PATH}/include",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _round_up(v: int, tile: int) -> int:
    return ((v + tile - 1) // tile) * tile


def _choose_block_dim(m: int, n: int, max_block_dim: int) -> int:
    m_loop = m // M_TILE
    n_loop = n // N_TILE
    core_loop = m_loop * n_loop
    if core_loop <= 0:
        return 1
    return max(1, min(core_loop, max_block_dim))


def load_lib(lib_path):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    # call_kernel(blockDim, stream, x, y, z, M, N, K)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x [M, K]
        ctypes.c_void_p,  # y [N, K]
        ctypes.c_void_p,  # z [M, N]
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
    ]
    lib.call_kernel.restype = None

    def _launch_kernel_f16(a, b, c, m, n, k, block_dim, stream_ptr):
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            torch_to_ctypes(c),
            m,
            n,
            k,
        )

    def _matmul_single(a, b, max_block_dim, stream_ptr):
        m = int(a.shape[0])
        k = int(a.shape[1])
        n = int(b.shape[0])

        m_pad = _round_up(m, M_TILE)
        n_pad = _round_up(n, N_TILE)
        k_pad = _round_up(k, K_TILE)

        # Fast path: if N/K are aligned, avoid padded-M launch for the tail.
        if n == n_pad and k == k_pad:
            m_main = (m // M_TILE) * M_TILE
            if m_main == m and m_main > 0:
                out = torch.empty((m, n), device=a.device, dtype=a.dtype)
                block_dim = _choose_block_dim(m, n, max_block_dim)
                _launch_kernel_f16(a, b, out, m, n, k, block_dim, stream_ptr)
                return out

            if m_main == 0:
                return torch.matmul(a, b.transpose(0, 1))

            out = torch.empty((m, n), device=a.device, dtype=a.dtype)
            a_main = a[:m_main, :]
            out_main = out[:m_main, :]
            block_dim = _choose_block_dim(m_main, n, max_block_dim)
            _launch_kernel_f16(a_main, b, out_main, m_main, n, k, block_dim, stream_ptr)

            out_tail = out[m_main:, :]
            tail = torch.matmul(a[m_main:, :], b.transpose(0, 1))
            out_tail.copy_(tail)
            return out

        # General padded path (simple, no caching/partial-zero optimization).
        if m_pad != m or n_pad != n or k_pad != k:
            a_work = torch.zeros((m_pad, k_pad), device=a.device, dtype=a.dtype)
            b_work = torch.zeros((n_pad, k_pad), device=b.device, dtype=b.dtype)
            a_work[:m, :k] = a
            b_work[:n, :k] = b
            c_work = torch.empty((m_pad, n_pad), device=a.device, dtype=a.dtype)
        else:
            a_work = a
            b_work = b
            c_work = torch.empty((m, n), device=a.device, dtype=a.dtype)

        block_dim = _choose_block_dim(m_pad, n_pad, max_block_dim)
        _launch_kernel_f16(
            a_work, b_work, c_work, m_pad, n_pad, k_pad, block_dim, stream_ptr
        )
        return c_work[:m, :n]

    def matmul_abt(
        a,
        b,
        max_block_dim=DEFAULT_MAX_BLOCK_DIM,
        stream_ptr=None,
    ):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul_abt expects 2D tensors: a[M,K], b[N,K]")
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"K mismatch: a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
            )
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError("matmul_abt currently supports float16 inputs only")

        if stream_ptr is None:
            stream = torch.npu.current_stream()
            stream_ptr = getattr(stream, "_as_parameter_", None)

        return _matmul_single(a, b, max_block_dim, stream_ptr)

    return matmul_abt


def jit_compile(src_path, verbose=True, clean_up=False):
    lib_path = compile_cpp(src_path, verbose=verbose)
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func
