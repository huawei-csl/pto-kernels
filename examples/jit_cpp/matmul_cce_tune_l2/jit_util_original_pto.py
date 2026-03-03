import ctypes
import os
import subprocess

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
PTO_LIB_PATH = os.environ.get("PTO_LIB_PATH", ASCEND_TOOLKIT_HOME)

DEFAULT_MAX_BLOCK_DIM = int(os.environ.get("ORIG_PTO_MATMUL_BLOCK_DIM", "20"))
M_TILE = 128
N_TILE = 256
K_TILE = 64


def _round_up(v: int, tile: int) -> int:
    return ((v + tile - 1) // tile) * tile


def _divisors(v: int):
    out = []
    for i in range(1, int(v**0.5) + 1):
        if v % i == 0:
            out.append(i)
            if i * i != v:
                out.append(v // i)
    return sorted(out)


def _choose_partition(m: int, n: int, max_block_dim: int):
    max_block_dim = max(1, int(max_block_dim))
    m_tiles = m // M_TILE
    n_tiles = n // N_TILE

    best_m_iter = 1
    best_n_iter = 1
    best_blocks = 1

    for m_iter in _divisors(m_tiles):
        for n_iter in _divisors(n_tiles):
            blocks = m_iter * n_iter
            if best_blocks < blocks <= max_block_dim:
                best_m_iter = m_iter
                best_n_iter = n_iter
                best_blocks = blocks
    return best_m_iter, best_n_iter, best_blocks


def compile_cpp(
    kernel_cpp: str,
    m: int,
    n: int,
    k: int,
    m_iter: int,
    n_iter: int,
    block_dim: int,
    *,
    verbose: bool = False,
    timeout: int = 120,
) -> str:
    so_dir = os.path.join(os.path.dirname(kernel_cpp), "outputs", "so")
    os.makedirs(so_dir, exist_ok=True)
    lib_path = os.path.join(
        so_dir,
        f"matmul_original_pto_m{m}_n{n}_k{k}_mi{m_iter}_ni{n_iter}.so",
    )

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
        f"-DORIG_PTO_M={m}",
        f"-DORIG_PTO_N={n}",
        f"-DORIG_PTO_K={k}",
        f"-DORIG_PTO_BASE_M={M_TILE}",
        f"-DORIG_PTO_BASE_N={N_TILE}",
        f"-DORIG_PTO_BASE_K={K_TILE}",
        f"-DORIG_PTO_M_ITER={m_iter}",
        f"-DORIG_PTO_N_ITER={n_iter}",
        f"-DORIG_PTO_BLOCK_DIM={block_dim}",
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


def _as_ctypes_ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _load_lib(lib_path):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    # call_kernel(blockDim, stream, out, src0, src1)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.call_kernel.restype = None
    return lib


def jit_compile(src_path, verbose=True, clean_up=False):
    # Keep per-shape compiled libs in-memory for this process.
    compiled = {}
    compiled_paths = []

    def _ensure_shape_lib(m: int, n: int, k: int, max_block_dim: int):
        key = (m, n, k, max_block_dim)
        if key in compiled:
            return compiled[key]

        m_iter, n_iter, block_dim = _choose_partition(m, n, max_block_dim)
        lib_path = compile_cpp(
            src_path,
            m,
            n,
            k,
            m_iter,
            n_iter,
            block_dim,
            verbose=verbose,
        )
        lib = _load_lib(lib_path)
        compiled[key] = (lib, block_dim)
        compiled_paths.append(lib_path)
        return compiled[key]

    def matmul_abt(a, b, max_block_dim=DEFAULT_MAX_BLOCK_DIM, stream_ptr=None):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul_abt expects 2D tensors: a[M,K], b[N,K]")
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"K mismatch: a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
            )
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError("original_pto kernel supports float16 inputs only")

        m = int(a.shape[0])
        n = int(b.shape[0])
        k = int(a.shape[1])
        m_pad = _round_up(m, M_TILE)
        n_pad = _round_up(n, N_TILE)
        k_pad = _round_up(k, K_TILE)

        if m_pad != m or n_pad != n or k_pad != k:
            a_work = torch.zeros((m_pad, k_pad), device=a.device, dtype=a.dtype)
            b_work = torch.zeros((n_pad, k_pad), device=b.device, dtype=b.dtype)
            a_work[:m, :k] = a
            b_work[:n, :k] = b
        else:
            a_work = a
            b_work = b

        # Kernel writes float output.
        out_work = torch.empty((m_pad, n_pad), device=a.device, dtype=torch.float32)

        if stream_ptr is None:
            stream = torch.npu.current_stream()
            stream_ptr = getattr(stream, "_as_parameter_", None)

        lib, block_dim = _ensure_shape_lib(m_pad, n_pad, k_pad, int(max_block_dim))
        lib.call_kernel(
            int(block_dim),
            stream_ptr,
            _as_ctypes_ptr(out_work),
            _as_ctypes_ptr(a_work),
            _as_ctypes_ptr(b_work),
        )
        return out_work[:m, :n]

    if clean_up:
        # Keep API parity with other wrappers; callers can request cleanup,
        # but only after all cached shape builds are loaded.
        def wrapped(*args, **kwargs):
            result = matmul_abt(*args, **kwargs)
            for p in compiled_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            compiled_paths.clear()
            return result

        return wrapped

    return matmul_abt
