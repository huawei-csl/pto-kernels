"""
Load compiled static PTO shared libraries for chunk_gdn kernels (ctypes).
"""
from __future__ import annotations

import ctypes
import os
from functools import lru_cache

from pto_static_common import compile_pto_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=8)
def lib_chunk_cumsum():
    p = compile_pto_kernel("chunk_cumsum_kernel.cpp", "chunk_cumsum_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.call.restype = None
    return lib


@lru_cache(maxsize=8)
def lib_scaled_dot_kkt():
    p = compile_pto_kernel("scaled_dot_kkt_kernel.cpp", "scaled_dot_kkt_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


@lru_cache(maxsize=8)
def lib_wy_fast():
    p = compile_pto_kernel("wy_fast_kernel.cpp", "wy_fast_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


@lru_cache(maxsize=8)
def lib_chunk_h():
    p = compile_pto_kernel("chunk_h_kernel.cpp", "chunk_h_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib


@lru_cache(maxsize=8)
def lib_chunk_o():
    p = compile_pto_kernel("chunk_o_kernel.cpp", "chunk_o_static.so")
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.call.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_void_p]
    lib.call.restype = None
    return lib
