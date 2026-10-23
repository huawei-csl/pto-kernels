import ctypes
import math
import os
import subprocess
from pathlib import Path

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]


def _resolve_pto_lib_path() -> str:
    """Pick the PTO-ISA header root for bisheng's -isystem flag.

    Resolution order:
      1. ``$PTO_LIB_PATH`` (explicit override).
      2. The pto-kernels CMake FetchContent mirror at
         ``<repo>/build/_deps/libpto_isa_headers-src`` (pinned in
         top-level CMakeLists.txt; populated by any cmake build).
      3. ``$ASCEND_TOOLKIT_HOME`` (CANN default; may lack newer
         instructions such as TCOLEXPANDDIV on CANN 8.5.0).
    """
    env = os.environ.get("PTO_LIB_PATH")
    if env:
        return env
    # examples/jit_cpp/<example>/jit_util_common.py → parents[3] is the repo root
    repo_root = Path(__file__).resolve().parents[3]
    vendored = repo_root / "build" / "_deps" / "libpto_isa_headers-src"
    if (vendored / "include" / "pto" / "pto-inst.hpp").is_file():
        return str(vendored)
    return ASCEND_TOOLKIT_HOME


PTO_LIB_PATH = _resolve_pto_lib_path()
DEFAULT_DEVICE = "npu:0"
DEFAULT_BLOCK_DIM = 20
MAX_HADAMARD_N = 16384
FUSED_HADAMARD_QUANT_ARGTYPES = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # x
    ctypes.c_void_p,  # y
    ctypes.c_void_p,  # group_scales
    ctypes.c_void_p,  # group_offsets
    ctypes.c_uint32,  # scale_group_stride
    ctypes.c_uint32,  # offset_group_stride
    ctypes.c_uint32,  # batch
    ctypes.c_uint32,  # n
    ctypes.c_uint32,  # log2_n
    ctypes.c_float,  # scale
    ctypes.c_uint32,  # group_size
    ctypes.c_float,  # q_offset
]


def normalize_npu_device(device: str | int) -> str:
    """Normalize NPU device inputs to canonical 'npu:<index>' format."""
    text = str(device).strip().strip('"').strip("'")
    if text.lower().startswith("npu:"):
        index = text.split(":", 1)[1].strip()
    else:
        index = text

    if not index.isdigit():
        raise ValueError(
            f"Invalid NPU device '{device}'. Expected values like 0 or npu:0."
        )
    return f"npu:{int(index)}"


def get_cube_block_dim(device: str | int = DEFAULT_DEVICE) -> int:
    resolved_device = normalize_npu_device(device)
    return int(
        getattr(
            torch.npu.get_device_properties(resolved_device),
            "cube_core_num",
            DEFAULT_BLOCK_DIM,
        )
    )


try:
    BLOCK_DIM = get_cube_block_dim(DEFAULT_DEVICE)
except Exception:
    BLOCK_DIM = DEFAULT_BLOCK_DIM


def chmod_output_path(path: str | Path) -> None:
    try:
        os.chmod(path, 0o777)
    except OSError:
        pass


def compile_cpp(
    kernel_cpp: str,
    verbose: bool = False,
    timeout: int = 120,
    so_dir: str | None = None,
) -> str:
    kernel_path = Path(kernel_cpp).resolve()
    out_dir = (
        Path(so_dir) if so_dir is not None else kernel_path.parent / "outputs" / "so"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chmod_output_path(out_dir)
    lib_path = out_dir / f"{kernel_path.stem}_jit.so"

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "-DMEMORY_BASE",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        "--cce-aicore-arch=dav-c220-vec",
        "-isystem",
        f"{PTO_LIB_PATH}/include",
    ]
    command = ["bisheng", *flags, str(kernel_path), "-o", str(lib_path)]
    if verbose:
        print("compile command:", " ".join(command))

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    chmod_output_path(lib_path)
    return str(lib_path)


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def optional_torch_to_ctypes(tensor):
    if tensor is None:
        return None
    return torch_to_ctypes(tensor)


def validate_group_param_tensor(params, name, x, batch, groups_per_row):
    if params is None:
        return 0
    if params.dtype != torch.float16:
        raise TypeError(f"{name} must use torch.float16.")
    if params.device != x.device:
        raise ValueError(f"{name} must be on the same device as x.")
    if not params.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")

    if params.dim() == 1:
        if params.shape[0] != groups_per_row:
            raise ValueError(f"1D {name} must have shape [num_groups].")
        return 0

    if params.dim() == 2:
        if params.shape[1] != groups_per_row:
            raise ValueError(f"2D {name} must have shape [batch|1, num_groups].")
        if params.shape[0] == 1:
            return 0
        if params.shape[0] == batch:
            return groups_per_row
        raise ValueError(f"2D {name} must have a leading dim of 1 or batch.")

    raise ValueError(f"{name} must be 1D or 2D.")


def infer_group_size(n, group_size, q_scales):
    if group_size is not None:
        resolved = int(group_size)
        if resolved <= 0 or n % resolved != 0:
            raise ValueError("group_size must be a positive divisor of n.")
        return resolved

    if q_scales is None:
        return n

    groups_per_row = int(q_scales.shape[-1])
    if groups_per_row <= 0 or n % groups_per_row != 0:
        raise ValueError("Could not infer a valid group_size from q_scales.")
    return n // groups_per_row


def validate_hadamard_shape(n, log2_n):
    if n <= 0:
        raise ValueError(f"n must be a positive integer, but got n={n}.")
    if n & (n - 1):
        raise ValueError(f"n must be a power of two, but got n={n}.")
    if n > MAX_HADAMARD_N:
        raise ValueError(
            f"n must be <= {MAX_HADAMARD_N} (kernel limit), but got n={n}."
        )
    expected_log2_n = int(math.log2(n))
    if log2_n != expected_log2_n:
        raise ValueError(f"log2_n must equal int(log2(n)); got n={n}, log2_n={log2_n}.")


def validate_matching_2d_tensors(x, y):
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError("x and y must be 2D tensors.")
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes.")


def resolve_hadamard_call_shape(x, *, batch=None, n=None, log2_n=None):
    resolved_batch = x.shape[0] if batch is None else int(batch)
    resolved_n = x.shape[1] if n is None else int(n)
    if resolved_batch != x.shape[0] or resolved_n != x.shape[1]:
        raise ValueError("batch and n must match the input tensor shape.")
    resolved_log2_n = int(math.log2(resolved_n)) if log2_n is None else int(log2_n)
    validate_hadamard_shape(resolved_n, resolved_log2_n)
    return resolved_batch, resolved_n, resolved_log2_n


def resolve_grouped_quant_config(x, batch, n, group_size, q_scales, q_offsets):
    resolved_group_size = infer_group_size(n, group_size, q_scales)
    groups_per_row = n // resolved_group_size
    scale_group_stride = validate_group_param_tensor(
        q_scales, "q_scales", x, batch, groups_per_row
    )
    offset_group_stride = validate_group_param_tensor(
        q_offsets, "q_offsets", x, batch, groups_per_row
    )
    return resolved_group_size, scale_group_stride, offset_group_stride


def make_fused_hadamard_quant_func(kernel, resolved_block_dim: int):
    def fused_func(
        x,
        y,
        batch=None,
        n=None,
        log2_n=None,
        scale=1.0,
        *,
        group_size=None,
        q_offset=0.0,
        q_scales=None,
        q_offsets=None,
        block_dim=resolved_block_dim,
        stream_ptr=None,
    ):
        validate_matching_2d_tensors(x, y)
        batch, n, log2_n = resolve_hadamard_call_shape(
            x,
            batch=batch,
            n=n,
            log2_n=log2_n,
        )
        resolved_group_size, scale_group_stride, offset_group_stride = (
            resolve_grouped_quant_config(
                x,
                batch,
                n,
                group_size,
                q_scales,
                q_offsets,
            )
        )

        kernel(
            resolve_launch_block_dim(block_dim, resolved_block_dim),
            resolve_stream_ptr(stream_ptr),
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            optional_torch_to_ctypes(q_scales),
            optional_torch_to_ctypes(q_offsets),
            scale_group_stride,
            offset_group_stride,
            batch,
            n,
            log2_n,
            float(scale),
            resolved_group_size,
            float(q_offset),
        )

    fused_func.block_dim = resolved_block_dim
    fused_func.supports_grouped = True
    return fused_func


def get_current_stream_ptr():
    stream = torch.npu.current_stream()
    stream_ptr = getattr(stream, "_as_parameter_", None)
    if stream_ptr is None:
        raise RuntimeError("Could not access the current NPU stream pointer.")
    return stream_ptr


def resolve_block_dim(
    device: str | int = DEFAULT_DEVICE,
    block_dim: int | None = None,
) -> int:
    if block_dim is not None:
        return max(1, int(block_dim))
    return get_cube_block_dim(device)


def resolve_launch_block_dim(block_dim, default_block_dim: int) -> int:
    return default_block_dim if block_dim is None else int(block_dim)


def resolve_stream_ptr(stream_ptr=None):
    if stream_ptr is None:
        return get_current_stream_ptr()
    return stream_ptr


def load_cdll(lib_path: str):
    return ctypes.CDLL(os.path.abspath(lib_path))


def load_required_symbol(lib, symbol_name: str, argtypes):
    if not hasattr(lib, symbol_name):
        raise AttributeError(f"Could not find {symbol_name} in the compiled library.")
    kernel = getattr(lib, symbol_name)
    kernel.argtypes = argtypes
    kernel.restype = None
    return kernel


def jit_compile_with_loader(
    src_path,
    load_lib,
    *,
    verbose=True,
    clean_up=False,
    so_dir=None,
    device: str | int = DEFAULT_DEVICE,
    block_dim=None,
):
    resolved_device = normalize_npu_device(device)
    resolved_block_dim = resolve_block_dim(resolved_device, block_dim)
    lib_path = compile_cpp(src_path, verbose=verbose, so_dir=so_dir)
    func = load_lib(lib_path, block_dim=resolved_block_dim)
    if clean_up:
        os.remove(lib_path)
    return func
