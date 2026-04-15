"""
Load compiled static PTO shared libraries for chunk_gdn kernels (ctypes).
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache

from pto_static_common import compile_pto_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))


def _kernel_mtime(cpp_name: str) -> int:
    return os.stat(os.path.join(_HERE, cpp_name)).st_mtime_ns


@lru_cache(maxsize=8)
def _lib_chunk_cumsum_cached(cpp_mtime_ns: int):
    del cpp_mtime_ns
    p = compile_pto_kernel("chunk_cumsum_kernel.cpp", "chunk_cumsum_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.call.restype = None
    return lib


def lib_chunk_cumsum():
    return _lib_chunk_cumsum_cached(_kernel_mtime("chunk_cumsum_kernel.cpp"))


@lru_cache(maxsize=8)
def _lib_scaled_dot_kkt_cached(cpp_mtime_ns: int):
    del cpp_mtime_ns
    p = compile_pto_kernel("scaled_dot_kkt_kernel.cpp", "scaled_dot_kkt_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


def lib_scaled_dot_kkt():
    return _lib_scaled_dot_kkt_cached(_kernel_mtime("scaled_dot_kkt_kernel.cpp"))


@lru_cache(maxsize=8)
def _lib_wy_fast_cached(cpp_mtime_ns: int):
    del cpp_mtime_ns
    p = compile_pto_kernel("wy_fast_kernel.cpp", "wy_fast_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


def lib_wy_fast():
    return _lib_wy_fast_cached(_kernel_mtime("wy_fast_kernel.cpp"))


@lru_cache(maxsize=8)
def _lib_chunk_h_cached(cpp_mtime_ns: int):
    del cpp_mtime_ns
    p = compile_pto_kernel("chunk_h_kernel.cpp", "chunk_h_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


def lib_chunk_h():
    return _lib_chunk_h_cached(_kernel_mtime("chunk_h_kernel.cpp"))


@lru_cache(maxsize=8)
def _lib_chunk_o_cached(cpp_mtime_ns: int):
    del cpp_mtime_ns
    p = compile_pto_kernel("chunk_o_kernel.cpp", "chunk_o_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


def lib_chunk_o():
    return _lib_chunk_o_cached(_kernel_mtime("chunk_o_kernel.cpp"))
