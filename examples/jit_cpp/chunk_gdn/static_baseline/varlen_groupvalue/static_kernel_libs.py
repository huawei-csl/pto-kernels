"""
Load compiled varlen chunk_gated_delta_rule PTO shared libraries (ctypes).
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache

from pto_static_common import compile_pto_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))


def _kernel_mtime(cpp_name: str) -> int:
    return os.stat(os.path.join(_HERE, cpp_name)).st_mtime_ns


@lru_cache(maxsize=4)
def _lib_varlen_h32_cached(cpp_mtime_ns: int):
    del cpp_mtime_ns
    p = compile_pto_kernel(
        "chunk_gated_delta_rule_varlen_H32_kernel.cpp",
        "chunk_gated_delta_rule_varlen_H32_static.so",
    )
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


def lib_chunk_gated_delta_rule_varlen_h32():
    return _lib_varlen_h32_cached(_kernel_mtime("chunk_gated_delta_rule_varlen_H32_kernel.cpp"))


@lru_cache(maxsize=4)
def _lib_varlen_h48_cached(cpp_mtime_ns: int):
    del cpp_mtime_ns
    p = compile_pto_kernel(
        "chunk_gated_delta_rule_varlen_H48_kernel.cpp",
        "chunk_gated_delta_rule_varlen_H48_static.so",
    )
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


def lib_chunk_gated_delta_rule_varlen_h48():
    return _lib_varlen_h48_cached(_kernel_mtime("chunk_gated_delta_rule_varlen_H48_kernel.cpp"))
