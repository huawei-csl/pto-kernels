import torch  # noqa: F401
import torch_npu  # noqa: F401

from . import _C


def add(out, x, z, stream: int):
    _C.launch_static_add(out, x, z, stream)
    return out


def matmul(out, a, b, stream: int):
    _C.launch_static_matmul(out, a, b, stream)
    return out
